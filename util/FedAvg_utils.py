# --------------------------------------------------------
# Based on BEiT and MAE code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/facebookresearch/mae
# Author: Rui Yan
# --------------------------------------------------------

from __future__ import absolute_import, division, print_function
import os
import numpy as np
from copy import deepcopy
import torch
from lib.utils import get_conf
from functools import partial

from . import misc as misc
from .optim_factory import create_optimizer, add_weight_decay, get_parameter_groups
from lib.utils import LayerDecayValueAssigner
import lib.models as models
import lib.networks as networks

from timm.utils import accuracy
from lib.data.med_transforms import get_metric_val_transforms, get_mae_radom_transforms, get_scratch_labeled_transforms, get_scratch_unlabeled_transforms, get_val_transforms
from monai import data
from monai.data import load_decathlon_datalist




def Partial_Client_Selection(args, model,mode='pretrain'):
    
    device = torch.device(args.gpu)

    if args.split_type == 'central':
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, args.split_type))
    else:
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type))
        print("dis_cvs_files:", args.dis_cvs_files)
    
    # Select partial clients join in FL train
    if args.num_local_clients == -1: # all the clients joined in the train
        args.proxy_clients = args.dis_cvs_files
        args.num_local_clients =  len(args.dis_cvs_files)# update the true number of clients
    else:
        args.proxy_clients = ['train_' + str(i) for i in range(args.num_local_clients)]
    
    # Generate model for each client
    model_all = {}
    dataset_all = {}
    optimizer_all = {}
    scaler_all = {}
    criterion_all = {}
    lr_scheduler_all = {}
    wd_scheduler_all = {}
    loss_scaler_all = {}
    mixup_fn_all = {}
    args.learning_rate_record = {}
    args.t_total = {}


    
    # Load pretrained model if mode='finetune'

            
    for proxy_single_client in args.proxy_clients:
        
        global_rank = misc.get_rank()
        num_tasks = misc.get_world_size()
        
        #print('clients_with_len: ', args.clients_with_len[proxy_single_client])

        if args.model_name == 'MAE3D':
            total_batch_size = args.batch_size * args.accum_iter * num_tasks
            if args.lr is None:  # only base_lr is specified
                args.lr = args.blr * total_batch_size / 256


        
        # model_all
        # model_all[proxy_single_client] = deepcopy(model)
        #model_all[proxy_single_client] = model_all[proxy_single_client].to(device)

        if args.model_name != 'Unknown':
            print(f"=> creating model {args.model_name} of arch {args.arch}")
            model_all[proxy_single_client] = getattr(models, args.model_name)(
                encoder=getattr(networks, args.enc_arch),
                decoder=getattr(networks, args.dec_arch),
                args=args)

        model_all[proxy_single_client] = model_all[proxy_single_client].to(device)

        model_without_ddp = model_all[proxy_single_client]
        
        # optimizer_all

        # trainer_class = getattr(trainers, f'{args.trainer_name}', None)
        # assert trainer_class is not None, f"Trainer class {args.trainer_name} is not defined"
        # args.trainer_list[proxy_single_client] = trainer_class(args)


        # if mode == 'pretrain':
        #     if args.model_name == 'MAE3D':
        #         optim_params = get_parameter_groups(model)
        #         # print("888888888888888888888888", optim_params)
        #
        #         # TODO: create optimizer factory
        #         optimizer_all[proxy_single_client] = torch.optim.AdamW(optim_params,
        #                                            lr=args.lr,
        #                                            betas=(args.beta1, args.beta2),
        #                                            weight_decay=args.weight_decay)

        # build optimizer with layer-wise lr decay (lrd)
        if mode == 'pretrain':
            if args.model_name == 'beit':
                optimizer_all[proxy_single_client] = create_optimizer(args, model_without_ddp)
            elif args.model_name == 'MAE3D':
                param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
                optimizer_all[proxy_single_client] = torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))

        # dataset_all
        if mode == 'pretrain':

            if args.model_name == 'MAE3D':
                train_transform = get_mae_radom_transforms(args)
                data_dir = os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type)
                datalist_json = os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type,
                                             proxy_single_client)

                print("datalist_json:",datalist_json)

                datalist = load_decathlon_datalist(datalist_json,
                                                   True,
                                                   "training",
                                                   base_dir=data_dir)
                # print(datalist)
                train_ds = data.CacheDataset(
                    data=datalist,
                    transform=train_transform,
                    cache_num=len(datalist),
                    cache_rate=1,
                    num_workers=8,
                )
                dataset_all[proxy_single_client] = train_ds
                args.clients_with_len[proxy_single_client] = len(datalist)

            # //是只保留整数部分的除法规则
            num_training_steps_per_inner_epoch = args.clients_with_len[proxy_single_client] // total_batch_size

            # print("Batch size = %d" % total_batch_size)
            # print("Number of training steps = %d" % num_training_steps_per_inner_epoch)
            # print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_inner_epoch))




        # loss_scaler_all
        # scaler_all[proxy_single_client] = NativeScaler()
        scaler_all[proxy_single_client] = torch.cuda.amp.GradScaler()

        # get the total decay steps first
        args.t_total[proxy_single_client] = num_training_steps_per_inner_epoch * args.E_epoch * args.epochs

        args.learning_rate_record[proxy_single_client] = []
    
    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}


    # print("777777777777777777:",optimizer_all)


    if args.model_name == 'MAE3D':
        if mode == 'pretrain':
            return model_all, optimizer_all, dataset_all, scaler_all
        else:
            return model_all, optimizer_all, criterion_all, loss_scaler_all, mixup_fn_all


def Partial_Client_Selection_Fine_Tune(args, model, mode='fine_tune'):
    device = torch.device(args.gpu)

    if args.split_type == 'central':
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, args.split_type))
    else:
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type))
        print("dis_cvs_files:", args.dis_cvs_files)

    # Select partial clients join in FL train
    if args.num_local_clients == -1:  # all the clients joined in the train
        args.proxy_clients = args.dis_cvs_files
        args.num_local_clients = len(args.dis_cvs_files)  # update the true number of clients
    else:
        args.proxy_clients = ['train_' + str(i) for i in range(args.num_local_clients)]

    # Generate model for each client
    model_all = {}
    labeled_dataset = {}
    unlabeled_dataset = {}
    testset_all = {}
    metric_testset_all = {}
    optimizer_all = {}
    scaler_all = {}
    criterion_all = {}
    lr_scheduler_all = {}
    wd_scheduler_all = {}
    loss_scaler_all = {}
    mixup_fn_all = {}
    args.learning_rate_record = {}
    args.t_total = {}

    # if args.model_name == 'UNETR3D':
    #     test_transform = get_scratch_train_transforms(args)
    #     print("n_clients:",args.n_clients)
    #     data_dir = os.path.join(args.data_path, f'{args.n_clients}_clients', 'fine_tune')
    #     datalist_json = os.path.join(args.data_path, f'{args.n_clients}_clients', 'fine_tune',
    #                                  'test.json')
    #
    #     test_files = load_decathlon_datalist(datalist_json,
    #                                        True,
    #                                        "validation",
    #                                        base_dir=data_dir)
    #     # print(datalist)
    #     test_ds = data.Dataset(data=test_files, transform=test_transform)

    # Load pretrained model if mode='finetune'

    for proxy_single_client in args.proxy_clients:

        global_rank = misc.get_rank()
        num_tasks = misc.get_world_size()

        # print('clients_with_len: ', args.clients_with_len[proxy_single_client])

        if args.model_name == 'UNETR3D':
            total_batch_size = args.batch_size * args.accum_iter * num_tasks
            if args.lr is None:  # only base_lr is specified
                args.lr = args.blr * total_batch_size / 256

        # model_all
        model_all[proxy_single_client] = deepcopy(model)
        model_all[proxy_single_client] = model_all[proxy_single_client].to(device)

        # if args.model_name != 'Unknown':
        #     print(f"=> creating model {args.model_name} of arch {args.arch}")
        #     model_all[proxy_single_client] = getattr(models, args.model_name)(
        #         encoder=getattr(networks, args.enc_arch),
        #         decoder=getattr(networks, args.dec_arch),
        #         args=args)
        #
        # model_all[proxy_single_client] = model_all[proxy_single_client].to(device)

        model_without_ddp = model_all[proxy_single_client]

        # optimizer_all

        # trainer_class = getattr(trainers, f'{args.trainer_name}', None)
        # assert trainer_class is not None, f"Trainer class {args.trainer_name} is not defined"
        # args.trainer_list[proxy_single_client] = trainer_class(args)

        # if mode == 'pretrain':
        #     if args.model_name == 'MAE3D':
        #         optim_params = get_parameter_groups(model)
        #         # print("888888888888888888888888", optim_params)
        #
        #         # TODO: create optimizer factory
        #         optimizer_all[proxy_single_client] = torch.optim.AdamW(optim_params,
        #                                            lr=args.lr,
        #                                            betas=(args.beta1, args.beta2),
        #                                            weight_decay=args.weight_decay)

        # build optimizer with layer-wise lr decay (lrd)
        if mode == 'fine_tune':
            if args.model_name == 'beit':
                optimizer_all[proxy_single_client] = create_optimizer(args, model_without_ddp)
            elif args.model_name == 'UNETR3D':
                num_layers = model.get_num_layers()
                assigner = LayerDecayValueAssigner(
                    list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))

                # optim_params = self.group_params(model)
                optim_params = get_parameter_groups(model_without_ddp,get_layer_id=partial(assigner.get_layer_id, prefix='encoder.'),
                                                         get_layer_scale=assigner.get_scale,
                                                         verbose=True)
                # TODO: create optimizer factory
                optimizer_all[proxy_single_client] = torch.optim.AdamW(optim_params,
                                                   lr=args.lr,
                                                   betas=(args.beta1, args.beta2),
                                                   weight_decay=args.weight_decay)

        # dataset_all
        if mode == 'fine_tune':

            if args.model_name == 'UNETR3D':
                labeled_transform = get_scratch_labeled_transforms(args)
                unlabeled_transform = get_scratch_unlabeled_transforms(args)
                data_dir = os.path.join(args.data_path, f'{args.n_clients}_clients', 'fine_tune')
                labeled_datalist_json = os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type,
                                             proxy_single_client, 'labeled.json')
                unlabeled_datalist_json = os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type,
                                                     proxy_single_client, 'unlabeled.json')

                labeled_datalist = load_decathlon_datalist(labeled_datalist_json,
                                                   True,
                                                   "training",
                                                   base_dir=data_dir)
                # print(datalist)
                labeled_ds = data.CacheDataset(
                    data=labeled_datalist,
                    transform=labeled_transform,
                    cache_num=len(labeled_datalist),
                    cache_rate=0.8,
                    num_workers=8,
                )
                labeled_dataset[proxy_single_client] = labeled_ds

                unlabeled_datalist = load_decathlon_datalist(unlabeled_datalist_json,
                                                           True,
                                                           "training",
                                                           base_dir=data_dir)
                # print(datalist)
                unlabeled_ds = data.CacheDataset(
                    data=unlabeled_datalist,
                    transform=unlabeled_transform,
                    cache_num=len(unlabeled_datalist),
                    cache_rate=0.8,
                    num_workers=8,
                )
                labeled_dataset[proxy_single_client] = labeled_ds
                unlabeled_dataset[proxy_single_client] = unlabeled_ds


                args.clients_labeled_with_len[proxy_single_client] = len(labeled_datalist)
                args.clients_unlabeled_with_len[proxy_single_client] = len(unlabeled_datalist)
                args.clients_with_len[proxy_single_client] = len(labeled_datalist) + len(unlabeled_datalist)

            if args.model_name == 'UNETR3D':
                test_transform = get_val_transforms(args)
                metric_test_transform = get_metric_val_transforms(args)
                print("n_clients:", args.n_clients)
                data_dir = os.path.join(args.data_path, f'{args.n_clients}_clients', 'fine_tune')
                datalist_json = os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type,
                                             proxy_single_client, 'labeled.json')

                test_files = load_decathlon_datalist(datalist_json,
                                                     True,
                                                     "validation",
                                                     base_dir=data_dir)
                # print(datalist)
                test_ds = data.Dataset(data=test_files, transform=test_transform)
                testset_all[proxy_single_client] = test_ds

                datalist_metric_json = os.path.join(args.data_path, 'Metric', f'{proxy_single_client}.json')

                metric_test_files = load_decathlon_datalist(datalist_metric_json,
                                                     True,
                                                     "validation",
                                                     base_dir=data_dir)
                # print(datalist)
                metric_test_ds = data.Dataset(data=metric_test_files, transform=metric_test_transform)
                metric_testset_all[proxy_single_client] = metric_test_ds





            # //是只保留整数部分的除法规则
            num_training_steps_per_inner_epoch = len(labeled_datalist)// total_batch_size

            print("Batch size = %d" % total_batch_size)
            print("Number of training steps = %d" % num_training_steps_per_inner_epoch)
            print(
                "Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_inner_epoch))

        # loss_scaler_all
        # scaler_all[proxy_single_client] = NativeScaler()
        scaler_all[proxy_single_client] = torch.cuda.amp.GradScaler()

        # get the total decay steps first
        args.t_total[proxy_single_client] = num_training_steps_per_inner_epoch * args.E_epoch * args.epochs

        args.learning_rate_record[proxy_single_client] = []

        # if os.path.isfile(args.resume):
        #     print("=> loading checkpoint '{}'".format(args.resume))
        #     if args.gpu is None:
        #         checkpoint = torch.load(args.resume)
        #     else:
        #         # Map model to be loaded to specified single gpu.
        #         loc = 'cuda:{}'.format(args.gpu)
        #         checkpoint = torch.load(args.resume, map_location=loc)
        #     args.start_epoch = checkpoint['epoch']
        #     model_all[proxy_single_client].load_state_dict(checkpoint['state_dict'])
        #     optimizer_all[proxy_single_client].load_state_dict(checkpoint['optimizer'])
        #     scaler_all[proxy_single_client].load_state_dict(
        #         checkpoint['scaler'])  # additional line compared with base imple
        #     print("=> loaded checkpoint '{}' (epoch {})"
        #           .format(args.resume, checkpoint['epoch']))
        # else:
        #     print("=> no checkpoint found at '{}'".format(args.resume))



    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}

    # print("777777777777777777:",optimizer_all)

    if args.model_name == 'UNETR3D':
        if mode == 'fine_tune':
            return model_all, optimizer_all, labeled_dataset, unlabeled_dataset, scaler_all, testset_all, metric_testset_all
        else:
            return model_all, optimizer_all, criterion_all, loss_scaler_all, mixup_fn_all



def get_parameter_groups(model, get_layer_id=None, get_layer_scale=None, verbose=False):
    args = get_conf()
    weight_decay = args.weight_decay
    # model = self.model

    if hasattr(model, 'no_weight_decay'):
        skip_list = model.no_weight_decay()
        #print("skip_list:",skip_list)
    else:
        skip_list = {}

    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_id is not None:
            layer_id = get_layer_id(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    if verbose:
        import json
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    else:
        print("Param groups information is omitted...")
    return list(parameter_group_vars.values())


def average_model(args, model_avg, model_all,final_weight):
    model_avg.cpu()
    print('Calculate the model avg----')
    params = dict(model_avg.named_parameters())
        
    for name, param in params.items():
        for client in range(len(args.proxy_clients)):
            single_client = args.proxy_clients[client]
            
            single_client_weight = final_weight[single_client]
            single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()
            
            if client == 0:
                if args.distributed:
                    tmp_param_data = dict(model_all[single_client].module.named_parameters())[
                                         name].data * single_client_weight
                else:
                    tmp_param_data = dict(model_all[single_client].named_parameters())[
                                         name].data * single_client_weight
            else:
                if args.distributed:
                    tmp_param_data = tmp_param_data + \
                                     dict(model_all[single_client].module.named_parameters())[
                                         name].data * single_client_weight
                else:
                    tmp_param_data = tmp_param_data + \
                                     dict(model_all[single_client].named_parameters())[
                                         name].data * single_client_weight
        
        params[name].data.copy_(tmp_param_data)
        
    print('Update each client model parameters----')
        
    for single_client in args.proxy_clients:
        
        if args.distributed:
            tmp_params = dict(model_all[single_client].module.named_parameters())
        else:
            tmp_params = dict(model_all[single_client].named_parameters())

        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    client_name = os.path.basename(args.single_client).split('.')[0]
    model_checkpoint = os.path.join(args.output_dir, "%s_%s_checkpoint.bin" % (args.name, client_name))
    
    torch.save(model_to_save.state_dict(), model_checkpoint)
    # print("Saved model checkpoint to [DIR: %s]", args.output_dir)


def valid(args, model, data_loader):
    # eval_losses = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    
    print("++++++ Running Validation ++++++")
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1,losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def metric_evaluation(args, eval_result):
    if args.nb_classes == 1:
        if args.best_acc[args.single_client] < eval_result:
            Flag = False
        else:
            Flag = True
    else:
        if args.best_acc[args.single_client] < eval_result:
            Flag = True
        else:
            Flag = False
    return Flag