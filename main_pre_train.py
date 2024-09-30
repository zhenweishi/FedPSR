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

# torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
from copy import deepcopy
from monai import data
from monai.data import load_decathlon_datalist

from util.FedAvg_utils import Partial_Client_Selection, average_model
from lib.data.med_datasets import get_train_loader
import util.misc as misc
import sys
sys.path.append('lib/')
from lib.data.med_datasets import Sampler

from lib.utils import set_seed, dist_setup, get_conf
import lib.trainers as trainers
# from monai import data

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


def main_worker(gpu, args):

    args.gpu = gpu
    ngpus_per_node = args.ngpus_per_node
    dist_setup(ngpus_per_node, args)

    output_dir = args.output_dir
    ckpt_dir = {
        "center_1.json": f"{output_dir}/ckpts/center_1",
        "center_2.json": f"{output_dir}/ckpts/center_2",
        "center_3.json": f"{output_dir}/ckpts/center_3",
        "avg": f"{output_dir}/ckpts/avg"
    }
    for dir_path in ckpt_dir.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    # init trainer
    trainer_class = getattr(trainers, f'{args.trainer_name}', None)
    assert trainer_class is not None, f"Trainer class {args.trainer_name} is not defined"
    trainer = trainer_class(args)

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



    # create model
    model = None
    #wrapped_model= None
    if args.model_name != 'Unknown' and model is None:
        print(f"=> creating model {args.model_name} of arch {args.arch}")
        model = getattr(models, args.model_name)(
            encoder=getattr(networks, args.enc_arch),
            decoder=getattr(networks, args.dec_arch),
            args=args)
        #wrapped_model = wrap_model(model)

    model_all, optimizer_all, dataset_all, scaler_all= Partial_Client_Selection(args, model)
    model_avg = model

    # ---------- Train! (use different clients)
    print("=============== Running pre-training ===============")
    tot_clients = args.dis_cvs_files
    print('total_clients: ', tot_clients)

    print(f"Start training for {args.epochs} epochs, distributed={args.distributed}")

    # while True:
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch: ', epoch)
        epoch += 1

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
            print(f"{proxy_single_client}的权重是：",args.clients_weightes[proxy_single_client])
            # ---- get dataset for each client for pretraining
            dataset_train = dataset_all[proxy_single_client]
            model = model_all[proxy_single_client]
            optimizer = optimizer_all[proxy_single_client]
            loss_scaler = scaler_all[proxy_single_client]

            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_rank = global_rank
            num_training_steps_per_inner_epoch = args.clients_with_len[proxy_single_client] // args.batch_size // num_tasks

            print(f'=========client: {proxy_single_client} ==============')


            train_sampler = Sampler(dataset_train) if args.distributed else None
            data_loader_train = data.DataLoader(dataset_train,
                                           batch_size=args.batch_size,
                                           shuffle=(train_sampler is None),
                                           num_workers=args.workers,
                                           sampler=train_sampler,
                                           pin_memory=True,
                                           persistent_workers=_persistent_workers)

            # ---- prepare model for a client
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

            if args.lr is None:  # only base_lr is specified
                args.lr = args.lr * total_batch_size / 256

            print("base lr: %.2e" % (args.lr * args.batch_size / total_batch_size))
            print("actual lr: %.2e" % args.lr)
            print("accumulate grad iterations: %d" % args.accum_iter)
            print("effective batch size: %d" % total_batch_size)
            print("Number of training steps = %d" % num_training_steps_per_inner_epoch)
            print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_inner_epoch))

            iters_per_epoch = len(data_loader_train)

            niters = args.start_epoch * iters_per_epoch


            for inner_epoch in range(args.E_epoch):
                if args.distributed:
                    args.dataloader.sampler.set_epoch(epoch)
                    torch.distributed.barrier()

                niters = epoch_train(args, data_loader_train,model,optimizer,loss_scaler, epoch, niters, iters_per_epoch)

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
            print("--------------开始聚合----------------")
            average_model(args, model_avg, model_all)

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

    print("================End pre-training! ================ ")


def epoch_train(args,train_loader,model,optimizer,scaler, epoch, niters, iters_per_epoch):

    model = wrap_model(args,model)

    init_lr = args.lr * args.batch_size / 256


    # switch to train mode
    model.train()

    load_start_time = time.time()
    for i, batch_data in enumerate(train_loader):
        load_time = time.time() - load_start_time
        # adjust learning at the beginning of each iteration
        adjust_learning_rate(args, epoch + i / iters_per_epoch, optimizer)

        # For SSL pretraining, only image data is required for training
        image = batch_data['image']

        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        forward_start_time = time.time()
        with torch.cuda.amp.autocast(True):
            loss = model(image, return_image=False)
        forward_time = time.time() - forward_start_time

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError(f"Loss has NaN or Inf values: {loss.item()}")

        # compute gradient and do SGD step
        bp_start_time = time.time()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bp_time = time.time() - bp_start_time

        # Log to the screen
        if i % args.print_freq == 0:
            print(f"Epoch: {epoch:03d}/{args.epochs} | "
                  f"Iter: {i:05d}/{iters_per_epoch} | "
                  f"TotalIter: {niters:06d} | "
                  f"Init Lr: {init_lr:.05f} | "
                  f"Lr: {optimizer.param_groups[0]['lr']:.05f} | "
                  f"Load Time: {load_time:.03f}s | "
                  f"Forward Time: {forward_time:.03f}s | "
                  f"Backward Time: {bp_time:.03f}s | "
                  f"Loss: {loss.item():.03f}")
            if args.rank == 0:
                wandb.log(
                    {
                        "lr": optimizer.param_groups[0]['lr'],
                        "Loss": loss.item(),
                    },
                    step=niters,
                )

        niters += 1
        load_start_time = time.time()
    return niters

def vis_reconstruction(self, niters=0):
    args = self.args
    loader = self.val_dataloader
    model = self.wrapped_model

    model.eval()

    for batch_data in loader:
        image = batch_data['image']
        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        _, x, recon, masked_x = model(image, return_image=True)

        vis_tensor = torch.cat([x, masked_x, recon], dim=0)

        # visualize
        grid_size = []
        for pa_size, in_size in zip(to_3tuple(args.patch_size), to_3tuple(args.input_size)):
            grid_size.append(in_size // pa_size)
        vis_grid_hw = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='d')
        # import pdb
        # pdb.set_trace()
        # vis_grid_hd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='w')
        # vis_grid_wd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='h')

        print("wandb logging")
        vis_grid_hw = vis_grid_hw.cpu()
        vis_grid_hw_np = vis_grid_hw.numpy()
        vis_grid_hw_np = np.transpose(vis_grid_hw_np,(1, 2, 0))
        vis_grid_hw = wandb.Image(vis_grid_hw_np, caption=f"hw_iter{niters:06d}")
        # vis_grid_hd = wandb.Image(vis_grid_hd, caption=f"hd_iter{niters:06d}")
        # vis_grid_wd = wandb.Image(vis_grid_wd, caption=f"wd_iter{niters:06d}")

        wandb.log(
            {
            "vis_hw": vis_grid_hw,
            # "vis_hd": vis_grid_hd,
            # "vis_wd": vis_grid_wd
            },
            step=niters,
        )
        break
    print("finish wandb logging")

def wrap_model(args,model):
    """
    1. Distribute model or not
    2. Rewriting batch size and workers
    """
    #args = get_conf()
    assert model is not None, "Please build model before wrapping model"

    if args.distributed:
        ngpus_per_node = args.ngpus_per_node
        # Apply SyncBN
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = args.batch_size // ngpus_per_node
            args.workers = (args.workers + ngpus_per_node - 1) // ngpus_per_node
            print("=> Finish adapting batch size and workers according to gpu number")
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[args.gpu],
                                                        find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel
        raise NotImplementedError("Must Specify GPU or use DistributeDataParallel.")

    return model

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





if __name__ == '__main__':
    main()
