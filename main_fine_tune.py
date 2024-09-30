import os
import numpy as np
import warnings
import math
import shutil
import time
import torch
import torch.multiprocessing as mp
from lib.tools.visualization import patches3d_to_grid
from timm.layers.helpers import to_3tuple
import lib.models as models
import lib.networks as networks
from lib.utils import SmoothedValue
from collections import defaultdict, OrderedDict
# from monai.inferers import sliding_window_inference
from monai.data import decollate_batch, NibabelWriter
from lib.data.med_datasets import idx2label_all
from monai.metrics import compute_dice, compute_hausdorff_distance
from functools import partial
from itertools import cycle
from lib.data.psr_transform import FullAugmentor
from base.base_modules import TensorBuffer
import torch.nn.functional as F

import sys
sys.path.append('lib/')

# import lib.models
# import lib.networks
# torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
from monai import data
from monai.losses import DiceCELoss
from lib.data.med_transforms import get_post_transforms

from util.FedAvg_utils import Partial_Client_Selection_Fine_Tune, average_model
import util.misc as misc
from util.sliding_window_infer import sliding_window_inference
import sys
sys.path.append('lib/')
from lib.data.med_datasets import Sampler

from lib.utils import set_seed, dist_setup, get_conf

import torch.nn as nn

from packaging import version
_persistent_workers = False if version.parse(torch.__version__) < version.parse('1.8.2') else True

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'




def main():

    args = get_conf()

    args.test = False

    # set seed if required
    set_seed(args.seed)

    if not args.multiprocessing_distributed and args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                nprocs=ngpus_per_node,
                args=(args,))
    else:
        print("single process")
        main_worker(args.gpu, args)


def wrap_model(args, model):
    """
    1. Distribute model or not
    2. Rewriting batch size and workers
    """
    # args = self.args
    # model = self.model
    assert model is not None, "Please build model before wrapping model"

    if args.distributed:
        ngpus_per_node = args.ngpus_per_node
        # Apply SyncBN
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            print("=> Finish adapting batch size and workers according to gpu number")
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[args.gpu],
                                                        find_unused_parameters=True)
        else:
            model.cuda()
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("Must Specify GPU or use DistributeDataParallel.")

    return model


def deallocate_batch_dict(input_dict, batch_idx_start, batch_idx_end):
    """
    Deallocate the dict containing multiple batches into a dict with multiple items
    """
    out_dict = {}
    for key, value in input_dict.items():
        out_dict[key] = value[batch_idx_start:batch_idx_end]
    return out_dict

def dict_loss(loss_fn, inputs, targets, key_list=None, **kwargs):
    """
    Perform a certain loss function in a data dict
    Args:
        loss_fn: loss function
        inputs: input data dict
        targets: ground truth
        key_list: specify which keys should be taken into computation
        **kwargs: other args for loss function
    """
    loss = 0.0
    keys = key_list if key_list is not None else list(inputs.keys())
    for key in keys:
        loss += loss_fn(inputs[key], targets, **kwargs)
    return loss

def uncertainty_loss(inputs, targets):
    """
    Uncertainty estimation pseudo supervised loss
    """
    # detach from the computational graph
    pseudo_label = F.softmax(targets / 0.5, dim=1).detach() #args.TEMP = 0.5
    vanilla_loss = F.cross_entropy(inputs, pseudo_label, reduction='none')
    # uncertainty estimation
    kl_div = torch.sum(F.kl_div(F.log_softmax(inputs, dim=1), F.softmax(targets, dim=1).detach(), reduction='none'), dim=1)
    uncertainty_loss = (torch.exp(-kl_div) * vanilla_loss).mean() + kl_div.mean()
    return uncertainty_loss

def consist_loss(inputs, targets, key_list=None):
    """
    Consistency regularization between two augmented views
    """
    loss = 0.0
    keys = key_list if key_list is not None else list(inputs.keys())
    for key in keys:
        loss += (1.0 - F.cosine_similarity(inputs[key], targets[key], dim=1)).mean()
    return loss

def sigmoid_rampup(current):
    # args = self.args
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242
    Args:
        current: current epoch index
    """
    rampup_length = 200 #args.RAMPUP = 200
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def predictor(inputs, model):
    model = wrap_model(model)
    """
    Predictor function for `monai.inferers.sliding_window_inference`
    Args:
        inputs: input data

    Returns:
        output: network final output
    """
    output = model(inputs)['out']
    return output




def main_worker(gpu, args):

    args.gpu = gpu
    ngpus_per_node = args.ngpus_per_node
    dist_setup(ngpus_per_node, args)

    output_dir = args.output_dir
    ckpt_dir = {
        "GDPH": f"{output_dir}/ckpts/GDPH",
        "Lung1": f"{output_dir}/ckpts/Lung1",
        "Multi": f"{output_dir}/ckpts/Multi",
        "avg": f"{output_dir}/ckpts/avg",
        "best": f"{output_dir}/ckpts/best"
    }
    for dir_path in ckpt_dir.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    #if args.rank == 0 and not args.disable_wandb:
    if args.rank == 0 and not args.disable_wandb:
        if args.wandb_id is None:
            args.wandb_id = wandb.util.generate_id()

        run = wandb.init(project=f"{args.proj_name}_{args.dataset}",
                        name=args.run_name,
                        config=vars(args),
                        id=args.wandb_id,
                        resume='allow',
                        dir=args.output_dir)

    if args.test:
        metric_funcs = OrderedDict([
            ('Dice',
             compute_dice),
            ('HD',
             partial(compute_hausdorff_distance, percentile=95))
        ])
    else:
        metric_funcs = OrderedDict([
            ('Dice',
             compute_dice)
        ])

    # create model
    model = None
    #wrapped_model= None
    if args.model_name != 'Unknown' and model is None:

        print(f"=> creating model {args.model_name}")

        if args.dataset == 'lung':
            args.num_classes = 2
            loss_fn = DiceCELoss(to_onehot_y=True,
                                      softmax=True,
                                      squared_pred=True,
                                      smooth_nr=args.smooth_nr,
                                      smooth_dr=args.smooth_dr)
        else:
            raise ValueError(f"Unsupported dataset {args.dataset}")
        post_pred, post_label = get_post_transforms(args)

        # setup mixup and loss functions
        if args.mixup > 0:
            raise NotImplemented("Mixup for segmentation has not been implemented.")
        else:
            mixup_fn = None

        model = getattr(models, args.model_name)(encoder=getattr(networks, args.enc_arch),
                                                          decoder=getattr(networks, args.dec_arch),
                                                          args=args)

        # load pretrained weights
        if hasattr(args, 'test') and args.test and args.pretrain is not None and os.path.exists(args.pretrain):
            print(f"=> Start loading the model weights from {args.pretrain} for test")
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            state_dict = checkpoint['state_dict']
            msg = model.load_state_dict(state_dict, strict=False)
            print(f'Loading messages: \n {msg}')
            print(f"=> Finish loading pretrained weights from {args.pretrain}")
        elif args.pretrain is not None and os.path.exists(args.pretrain):
            print(f"=> Start loading pretrained weights from {args.pretrain}")
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # import pdb
            # pdb.set_trace()
            if args.model_name == 'UNETR3D':
                for key in list(state_dict.keys()):
                    if key.startswith('encoder.'):
                        state_dict[key[len('encoder.'):]] = state_dict[key]
                        del state_dict[key]
                    # need to concat and load pos embed. too
                    # TODO: unify the learning of pos embed of pretraining and finetuning
                    if key == 'encoder_pos_embed':
                        pe = torch.zeros([1, 1, state_dict[key].size(-1)])
                        state_dict['pos_embed'] = torch.cat([pe, state_dict[key]], dim=1)
                        del state_dict[key]
                    if key == 'patch_embed.proj.weight' and \
                            state_dict[
                                'patch_embed.proj.weight'].shape != model.encoder.patch_embed.proj.weight.shape:
                        del state_dict['patch_embed.proj.weight']
                        del state_dict['patch_embed.proj.bias']
                    if key == 'pos_embed' and \
                            state_dict['pos_embed'].shape != model.encoder.pos_embed.shape:
                        del state_dict[key]
                msg = model.encoder.load_state_dict(state_dict, strict=False)
            # self.model.load(state_dict)
            print(f'Loading messages: \n {msg}')
            print(f"=> Finish loading pretrained weights from {args.pretrain}")

        model= wrap_model(args, model)
    elif args.model_name == 'Unknown':
        raise ValueError("=> Model name is still unknown")
    else:
        raise ValueError("=> Model has been created. Do not create twice")
        #wrapped_model = wrap_model(model)



    model_all, optimizer_all, labeled_dataset, unlabeled_dataset, scaler_all, testset_all, metric_testset_all= Partial_Client_Selection_Fine_Tune(args, model)


    len_all = {}
    len_labeled_all = {}
    len_unlabeled_all = {}
    model_avg = model

    print("=============== Running fine-tuning ===============")
    tot_clients = args.dis_cvs_files
    print('total_clients: ', tot_clients)
    epoch = -1

    print(f"Start training for {args.epochs} epochs, distributed={args.distributed}")
    start_time = time.time()

    best_metric = 0

    project_l_negative = TensorBuffer(buffer_size=args.buffer_sizes, concat_dim=0)
    project_u_negative = TensorBuffer(buffer_size=args.buffer_sizes, concat_dim=0)
    map_l_negative = TensorBuffer(buffer_size=args.buffer_sizes, concat_dim=0)
    map_u_negative = TensorBuffer(buffer_size=args.buffer_sizes, concat_dim=0)

    ds_list = ['level3', 'level2', 'level1', 'out']

    # while True:
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch: ', epoch)

        # randomly select partial clients
        if args.num_local_clients == len(args.dis_cvs_files):
            # just use all the local clients
            cur_selected_clients = args.proxy_clients
        else:
            cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()

        # get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0
        for client in cur_selected_clients:
            cur_tot_client_Lens += args.clients_with_len[client]



        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            print('cur_single_client: ', cur_single_client)
            print('proxy_single_client: ', proxy_single_client)



            args.single_client = cur_single_client
            args.clients_weightes[proxy_single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens
            print(f"{cur_selected_clients}长度是{args.clients_with_len}")
            print(f"{proxy_single_client}的权重是：", args.clients_weightes)
            # ---- get dataset for each client for pretraining

            labeled_dataset_middle = labeled_dataset[proxy_single_client]
            unlabeled_dataset_middle = unlabeled_dataset[proxy_single_client]
            model = model_all[proxy_single_client]
            optimizer = optimizer_all[proxy_single_client]
            loss_scaler = scaler_all[proxy_single_client]

            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()

            print(f'=========client: {proxy_single_client} ==============')


            labeled_sampler = None
            data_loader_labeled = data.DataLoader(labeled_dataset_middle,
                                           batch_size=args.batch_size,
                                           shuffle=(labeled_sampler is None),
                                           num_workers=args.workers,
                                           sampler=labeled_sampler,
                                           pin_memory=True,
                                           persistent_workers=_persistent_workers)

            print(f"{proxy_single_client}的有标注数据总量:", len(data_loader_labeled))

            unlabeled_sampler = None
            data_loader_unlabeled = data.DataLoader(unlabeled_dataset_middle,
                                                batch_size=args.batch_size,
                                                shuffle=(unlabeled_sampler is None),
                                                num_workers=args.workers,
                                                sampler=unlabeled_sampler,
                                                pin_memory=True,
                                                persistent_workers=_persistent_workers)

            print(f"{proxy_single_client}的无标注数据总量:", len(data_loader_unlabeled))

            len_labeled_all[proxy_single_client] = len(data_loader_labeled)
            len_unlabeled_all[proxy_single_client] = len(data_loader_unlabeled)


            len_all[proxy_single_client] = len(data_loader_labeled) + len(data_loader_unlabeled)

            # ---- prepare model for a client

            total_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()


            if args.lr is None:  # only base_lr is specified
                args.lr = args.blr * total_batch_size / 256


            iters_per_epoch = len(data_loader_labeled)

            niters = args.start_epoch * iters_per_epoch


            for inner_epoch in range(args.E_epoch):
                if args.distributed:
                    args.dataloader.sampler.set_epoch(epoch)
                    torch.distributed.barrier()



                niters = epoch_train(args, data_loader_labeled, data_loader_unlabeled, model, optimizer, loss_scaler, loss_fn, epoch, niters,
                                     iters_per_epoch, proxy_single_client, project_l_negative, project_u_negative, map_l_negative, map_u_negative, ds_list)

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                    if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                        print(f"=> start saving checkpoint after epoch {epoch + 1}")
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scaler': loss_scaler.state_dict(),  # additional line compared with base imple
                        }, is_best=False, filename=f'{ckpt_dir[proxy_single_client]}/checkpoint_{epoch:04d}.pth.tar')
                        print("=> finish saving checkpoint")


        if (epoch + 1) % args.communication == 0:

            middle_weights = {}
            final_weights = {}
            sum_weight = 0

            print("未使用动态聚合权重策略之前的权重：", args.clients_weightes)

            dice_weights = get_dice(args, model_all, metric_testset_all, post_pred, post_label, metric_funcs)
            consisty_weights = get_consistency(args, model_all, metric_testset_all, post_pred)

            for client in cur_selected_clients:
                middle_weights[client] = args.clients_weightes[client] * consisty_weights[client] * dice_weights[client]
                sum_weight += middle_weights[client]

            for client in cur_selected_clients:
                final_weights[client] = middle_weights[client] / sum_weight

            print("使用动态聚合权重策略之后的权重：", final_weights)


            average_model(args, model_avg, model_all, final_weights)

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_avg.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': loss_scaler.state_dict(),  # additional line compared with base imple
                }, is_best=False, filename=f'{ckpt_dir["avg"]}/checkpoint_avg_{epoch:04d}.pth.tar')
                print("=> finish saving avg_checkpoint")




        if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
            sum_metric = {}
            len_all_list = 0
            sum_all_list = 0
            for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
                data_loader_test = testset_all[proxy_single_client]
                val_sampler = Sampler(data_loader_test, shuffle=False) if args.distributed else None
                val_loader = data.DataLoader(data_loader_test,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             persistent_workers=_persistent_workers)

                metric_list = evaluate(args,model_avg,val_loader,post_pred,post_label,metric_funcs,epoch=epoch, niters=niters)
                metric = metric_list[0]
                sum_metric[proxy_single_client] = metric * len_all[proxy_single_client]
                sum_all_list += sum_metric[proxy_single_client]
                len_all_list += len_all[proxy_single_client]

            avg_metric = sum_all_list / len_all_list if len_all_list else 0



            if avg_metric > best_metric:
                print(f"=> New val best metric: {avg_metric} | Old val best metric: {best_metric}!")
                best_metric = avg_metric
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scaler':  loss_scaler.state_dict(),  # additional line compared with base imple
                            'metric': avg_metric
                        },
                        is_best=False,
                        filename=f'{ckpt_dir["best"]}/best_model.pth.tar'
                    )
                    print("=> Finish saving best model.")
            else:
                print(f"=> Still old val best metric: {best_metric}")
    print("================End fine_tune! ================ ")


def epoch_train(args, labeled_loader, unlabeled_loader, model, optimizer, scaler, loss_fn, epoch, niters, iters_per_epoch, proxy_single_client, project_l_negative, project_u_negative, map_l_negative, map_u_negative, ds_list):
    init_lr = args.lr * args.batch_size / 256

    augmentor = FullAugmentor()

    print("model.parameters:",len(list(model.parameters())))
    print("optimizer.param_groups:",len(list(optimizer.param_groups[0]['params'])))

    # switch to train mode
    model.train()

    load_start_time = time.time()

    tbar = range(len(labeled_loader))

    data_loader = iter(zip(cycle(labeled_loader), unlabeled_loader))

    # compute output and loss
    forward_start_time = time.time()

    for step in tbar:
        # adjust learning at the beginning of each iteration
        adjust_learning_rate(args, epoch + args.clients_labeled_with_len[proxy_single_client] / iters_per_epoch, optimizer)
        batch_data_labeled, batch_data_unlabeled = next(data_loader)

        image_l = batch_data_labeled['image'].to(args.gpu)
        label_l = batch_data_labeled['label'].to(args.gpu)
        image_u = batch_data_unlabeled['image'].to(args.gpu)

        with torch.cuda.amp.autocast(True):
            middle_batch_size = image_l.shape[0]
            # Data Augmentation
            image_u_t1 = augmentor.forward_image(image_u)
            image_u_t2 = augmentor.forward_image(image_u)
            inputs = torch.cat([
                image_l, image_u, image_u_t1, image_u_t2],
                dim=0)
            outputs = model(inputs)
            out_l = deallocate_batch_dict(outputs, 0, middle_batch_size)
            out_u = deallocate_batch_dict(outputs, middle_batch_size, 2 * middle_batch_size)
            out_u_t1 = deallocate_batch_dict(outputs, 2 * middle_batch_size, 3 * middle_batch_size)
            out_u_t2 = deallocate_batch_dict(outputs, 3 * middle_batch_size, 4 * middle_batch_size)

            # define predictions
            pred_l = out_l['out']  # pseudo label for labeled data
            pred_u = out_u['out']  # pseudo label for unlabeled data

            seg_loss = loss_fn(pred_l, label_l)

            cps_u_loss = dict_loss(uncertainty_loss, out_u_t1, pred_u,
                                   key_list=ds_list) + \
                         dict_loss(uncertainty_loss, out_u_t2, pred_u,
                                   key_list=ds_list)
            cosine_u_loss = consist_loss(out_u_t1, out_u_t2, key_list=ds_list)

            tau = sigmoid_rampup(epoch)

            final_loss = args.SEG_RATIO * seg_loss + args.CPS_RATIO * tau * (
                    cps_u_loss + cosine_u_loss)

        forward_time = time.time() - forward_start_time

        # compute gradient and do SGD step
        bp_start_time = time.time()
        optimizer.zero_grad()
        scaler.scale(final_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bp_time = time.time() - bp_start_time
        if step % args.print_freq == 0:
            if 'lr_scale' in optimizer.param_groups[0]:
                last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
            else:
                last_layer_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch: {epoch:03d}/{args.epochs} | "
                  f"Iter: {step + 1:05d}/{args.clients_labeled_with_len[proxy_single_client]} | "
                  f"Init Lr: {init_lr:.05f} | "
                  f"Lr: {optimizer.param_groups[0]['lr']:.05f} | "
                  f"Forward Time: {forward_time:.03f}s | "
                  f"Backward Time: {bp_time:.03f}s | "
                  f"Dice Loss: {seg_loss.item():.03f} | "
                  f"u_u: {cps_u_loss.item():.03f} | "
                  f"cos_u: {cosine_u_loss.item():.03f} | "
                  f"Final Loss: {final_loss.item():.03f}")

        niters += 1
    return niters

def adjust_learning_rate(args, epoch, optimizer):
    """Base schedule: CosineDecay with warm-up."""
    init_lr = args.lr * args.batch_size / 256
    if epoch < args.warmup_epochs:
        cur_lr = init_lr * epoch / args.warmup_epochs
    else:
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
@torch.no_grad()
def evaluate(args,model,test_loader,post_pred,post_label,metric_funcs, epoch=0, niters=0):
    print("=> Start Evaluating")
    model = wrap_model(args,model)

    if args.spatial_dim == 3:
        roi_size = (args.roi_x, args.roi_y, args.roi_z)
    elif args.spatial_dim == 2:
        roi_size = (args.roi_x, args.roi_y)
    else:
        raise ValueError(f"Do not support this spatial dimension (={args.spatial_dim}) for now")

    meters = defaultdict(SmoothedValue)

    assert args.batch_size == 1, "Test mode requires batch size 1"
    ts_samples = int(len(test_loader))
    ts_meters = None
    print(f"test samples: {ts_samples}")

    # switch to evaluation mode
    model.eval()

    for i, batch_data in enumerate(test_loader):
        image, target = batch_data['image'], batch_data['label']
        if args.gpu is not None:
            image = image.to(args.gpu, non_blocking=True)
            target = target.to(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = sliding_window_inference(image,
                                              roi_size=roi_size,
                                              sw_batch_size=4,
                                              predictor=model,
                                              overlap=args.infer_overlap)
        target_convert = torch.stack([post_label(target_tensor) for target_tensor in decollate_batch(target)],
                                     dim=0)
        output_convert = torch.stack([post_pred(output_tensor) for output_tensor in decollate_batch(output)],
                                     dim=0)

        batch_size = image.size(0)
        idx2label = idx2label_all[args.dataset]
        for metric_name, metric_func in metric_funcs.items():
            if i < ts_samples:
                log_meters = meters
            else:
                log_meters = ts_meters
            metric = metric_func(y_pred=output_convert, y=target_convert,
                                 include_background=False if args.dataset == 'lung' else True)
            metric = metric.cpu().numpy()
            compute_avg_metric(metric, log_meters, metric_name, batch_size, args)
            for k in range(metric.shape[-1]):
                cls_metric = np.nanmean(metric, axis=0)[k]
                print("cls_metric:", cls_metric)
                if np.isnan(cls_metric) or np.isinf(cls_metric):
                    continue
                log_meters[f'{idx2label[k]}.{metric_name}'].update(value=cls_metric, n=batch_size)
        print(f'==> Evaluating on the {i + 1}th batch is finished.')

    # gather the stats from all processes
    if args.distributed:
        for k, v in meters.items():
            print(f'==> start synchronizing meter {k}...')
            v.synchronize_between_processes()
            print(f'==> finish synchronizing meter {k}...')
        if ts_meters is not None:
            for k, v in ts_meters.items():
                print(f'==> start synchronizing meter {k}...')
                v.synchronize_between_processes()
                print(f'==> finish synchronizing meter {k}...')
    # pdb.set_trace()
    log_string = f"==> Epoch {epoch:04d} val results: \n"
    for k, v in meters.items():
        if k == 'Dice':
            continue
        else:
            global_avg_metric = v.global_avg
            new_line = f"===> {k}: {global_avg_metric:.05f} \n"
            log_string += new_line
    print(log_string)
    if ts_meters is not None:
        log_string = f"==> Epoch {epoch:04d} test results: \n"
        for k, v in ts_meters.items():
            global_avg_metric = v.global_avg
            new_line = f"===> {k}: {global_avg_metric:.05f} \n"
            log_string += new_line
        print(log_string)

    print("=> Finish Evaluating")

    if args.dataset == 'lung':
        if ts_meters is None:
            return [meters['lung_tumor.Dice'].global_avg]

        else:
            return [meters['Dice'].global_avg, ts_meters['Dice'].global_avg]

def compute_avg_metric(metric, meters, metric_name, batch_size, args):
    assert len(metric.shape) == 2
    if args.dataset == 'lung':
        avg_metric = np.nanmean(np.nanmean(metric, axis=0))
        print("avg_metric:",avg_metric)
        meters[metric_name].update(value=avg_metric, n=batch_size)
    else:
        cls_avg_metric = np.nanmean(np.nanmean(metric, axis=0))
        meters[metric_name].update(value=cls_avg_metric, n=batch_size)


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss

@torch.no_grad()
def get_consistency(args, model_all, metric_dataset, post_pred):

    print("-------------开始计算 metric consistency-------------")


    augmentor = FullAugmentor()

    cur_selected_clients = args.proxy_clients

    roi_size = (args.roi_x, args.roi_y, args.roi_z)

    sum_pixel = 0
    con_pixel = 0
    sum_consistency_weight = 0

    consistency_weight = {}
    consistency_middle_weight = {}


    # get the quantity of clients joined in the FL train for updating the clients weights
    cur_tot_client_Lens = 0
    for client in cur_selected_clients:
        metric_dataset_middle = metric_dataset[client]
        metric_sampler = None
        data_loader_metric = data.DataLoader(metric_dataset_middle,
                                                batch_size=args.batch_size,
                                                shuffle=(metric_sampler is None),
                                                num_workers=args.workers,
                                                sampler=metric_sampler,
                                                pin_memory=True,
                                                persistent_workers=_persistent_workers)
        length = len(data_loader_metric)
        print(f"共有{length}个metric test data")

        sum = 0
        for i, batch_data in enumerate(data_loader_metric):


            image_u = batch_data['image'].to(args.gpu)
            image_u_t1 = augmentor.forward_image(image_u)
            image_u_t2 = augmentor.forward_image(image_u)
            current_model = model_all[client]
            # compute output
            with torch.cuda.amp.autocast():
                output_u_t1 = sliding_window_inference(image_u_t1,
                                                  roi_size=roi_size,
                                                  sw_batch_size=4,
                                                  predictor=current_model,
                                                  overlap=args.infer_overlap)

                output_u_t2 = sliding_window_inference(image_u_t2,
                                                       roi_size=roi_size,
                                                       sw_batch_size=4,
                                                       predictor=current_model,
                                                       overlap=args.infer_overlap)


            output_u_t1 = torch.softmax(output_u_t1, 1).cpu().numpy()
            output_convert_u_t1 = np.argmax(output_u_t1, axis=1).astype(np.uint8)[0]

            output_u_t2 = torch.softmax(output_u_t2, 1).cpu().numpy()
            output_convert_u_t2 = np.argmax(output_u_t2, axis=1).astype(np.uint8)[0]

            print("output_convert_u_t2:", output_convert_u_t2.shape)


            mask = (output_convert_u_t1 == output_convert_u_t2)
            single_same_sum_pixel = mask.sum()
            print(f"{client}的第{sum}个无标注数据有{single_same_sum_pixel}个相同像素")
            single_sum_pixel = output_convert_u_t1.size
            print(f"{client}的第{sum}个无标注数据总共有{single_sum_pixel}个像素")
            sum_pixel += output_convert_u_t1.size
            con_pixel += mask.sum()
            sum = sum + 1

            print(f"{client}的第{sum}个无标注数据推理完成")

        consistency_middle_weight[client] = (con_pixel / sum_pixel).item()
        middle_metric = consistency_middle_weight[client]
        print(f"=> Finish Evaluating on {client}, the consistency metric is {middle_metric}")
        sum_consistency_weight += consistency_middle_weight[client]


    for client in cur_selected_clients:
        consistency_weight[client] = consistency_middle_weight[client] / sum_consistency_weight

    print("一致性评估指标：", consistency_weight)



    return consistency_weight


@torch.no_grad()
def get_dice(args, model_all, metric_dataset, post_pred, post_label, metric_funcs):

    print("-------------开始计算 metric dice-------------")


    cur_selected_clients = args.proxy_clients

    roi_size = (args.roi_x, args.roi_y, args.roi_z)


    dice_weight = {}
    dice_middle_weight = {}
    sum_dice_weight = 0


    for client in cur_selected_clients:
        meters = defaultdict(SmoothedValue)
        metric_dataset_middle = metric_dataset[client]
        metric_sampler = None
        data_loader_metric = data.DataLoader(metric_dataset_middle,
                                                batch_size=args.batch_size,
                                                shuffle=(metric_sampler is None),
                                                num_workers=args.workers,
                                                sampler=metric_sampler,
                                                pin_memory=True,
                                                persistent_workers=_persistent_workers)
        length = len(data_loader_metric)
        print(f"共有{length}个metric test data")

        # print("------------finish----------")
        sum = 0
        for i, batch_data in enumerate(data_loader_metric):


            image = batch_data['image'].to(args.gpu)
            target = batch_data['label'].to(args.gpu)

            current_model = model_all[client]
            # compute output
            with torch.cuda.amp.autocast():
                output = sliding_window_inference(image,
                                                  roi_size=roi_size,
                                                  sw_batch_size=4,
                                                  predictor=current_model,
                                                  overlap=args.infer_overlap)

            target_convert = torch.stack([post_label(target_tensor) for target_tensor in decollate_batch(target)],
                                         dim=0)
            output_convert = torch.stack([post_pred(output_tensor) for output_tensor in decollate_batch(output)],
                                         dim=0)

            batch_size = image.size(0)
            idx2label = idx2label_all[args.dataset]
            for metric_name, metric_func in metric_funcs.items():
                log_meters = meters
                metric = metric_func(y_pred=output_convert, y=target_convert,
                                     include_background=False if args.dataset == 'lung' else True)
                metric = metric.cpu().numpy()
                compute_avg_metric(metric, log_meters, metric_name, batch_size, args)
                for k in range(metric.shape[-1]):
                    cls_metric = np.nanmean(metric, axis=0)[k]
                    print("cls_metric:", cls_metric)
                    if np.isnan(cls_metric) or np.isinf(cls_metric):
                        continue
                    log_meters[f'{idx2label[k]}.{metric_name}'].update(value=cls_metric, n=batch_size)
            print(f'==> Evaluating on the {i + 1}th batch is finished.')

        middle_metric = meters['lung_tumor.Dice'].global_avg

        print(f"=> Finish Evaluating on {client}, the mean dice is {middle_metric}")

        dice_middle_weight[client] = middle_metric

        sum_dice_weight += dice_middle_weight[client]


    for client in cur_selected_clients:
        dice_weight[client] = dice_middle_weight[client] / sum_dice_weight

    return dice_weight




if __name__ == '__main__':
    main()
