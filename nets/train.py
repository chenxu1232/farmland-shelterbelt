

import argparse
import datetime
import os
import time
import csv
import copy
import pandas as pd
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader


try:
    from thop import profile
except ImportError:
    print("pip install thop")
    profile = None

from nets.segformer import SegFormer
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch


def get_model_complexity(model, input_channels, input_shape):
    if profile is None:
        return 0, 0, 0

    try:
        model_copy = copy.deepcopy(model).cpu().eval()
    except Exception as e:
        print(f"计算 FLOPs 时模型深拷贝失败: {e}")
        return 0, 0, 0

    dummy_input = torch.randn(1, input_channels, input_shape[0], input_shape[1]).cpu()

    try:
        flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
    except Exception as e:
        print(f"计算 FLOPs 失败: {e}")
        return 0, 0, 0
    finally:
        del model_copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    flops_g = flops / 1e9
    params_m = params / 1e6
    params_mb = params * 4 / (1024 * 1024)

    return flops_g, params_m, params_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11, help="random seed")
    parser.add_argument("--save_dir", type=str, default="logs", help="directory to save logs/checkpoints")
    args = parser.parse_args()

    
    Cuda            = True
    seed            = args.seed
    distributed     = False
    sync_bn         = False
    fp16            = False
    
  
    num_classes     = 2
    
   
    input_channels  = 8

    phi             = "b5"
    pretrained      = True
    model_path      = "model_data/segformer_b5_backbone_weights.pth"
    
    input_shape     = [512, 512]
    
    Init_Epoch      = 0
    Freeze_Epoch    = 50
    Freeze_batch_size = 8
    UnFreeze_Epoch  = 100
    Unfreeze_batch_size = 4
    
    Freeze_Train    = False

    Init_lr         = 1e-4
    Min_lr          = Init_lr * 0.01
    optimizer_type  = "adamw"
    momentum        = 0.9
    weight_decay    = 1e-2
    lr_decay_type   = 'cos'
    save_period     = 5
    save_dir        = args.save_dir
    
   
    eval_flag       = True 
    eval_period     = 5

    VOCdevkit_path  = 'VOCdevkit'
    dice_loss       = True
    focal_loss      = False
    cls_weights     = np.ones([num_classes], np.float32)
    num_workers     = 4

    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ngpus_per_node  = torch.cuda.device_count()
    
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
    else:
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank  = 0
        rank        = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)
            dist.barrier()
        else:
            download_weights(phi)

   
    model = SegFormer(num_classes=num_classes, phi=phi, pretrained=pretrained, in_chans=input_channels)
    
    # -------------------------------------------------------------------#
    #   计算静态指标 (FLOPs, Params)
    # -------------------------------------------------------------------#
    flops_g, params_m, params_mb = 0, 0, 0
    if local_rank == 0:
        print("\n (FLOPs & Params)...")
        flops_g, params_m, params_mb = get_model_complexity(model, input_channels, input_shape)
        print(f" FLOPs={flops_g:.3f}G, Params={params_m:.3f}M ({params_mb:.2f}MB)\n")

    if not pretrained:
        weights_init(model)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}

        if isinstance(pretrained_dict, dict):
            if "state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["state_dict"]
            elif "model" in pretrained_dict:
                pretrained_dict = pretrained_dict["model"]

        cleaned_dict = {}
        for k, v in pretrained_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned_dict[nk] = v

        for k, v in cleaned_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict, strict=False)
        
        if local_rank == 0:
            print("\nSuccessful Load Key Num:", len(load_key))
            print("Fail To Load Key Num:", len(no_load_key))

    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = False
            model_train = model_train.cuda()
    
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, phi=phi, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    # -------------------------------------------------------------------#
    #   初始化详细指标 CSV (包含 clDice 和 CC-F1)
    # -------------------------------------------------------------------#
    metrics_csv_path = ""
    if local_rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metrics_csv_path = os.path.join(save_dir, "training_metrics_every_5_epochs.csv")
        
        if not os.path.exists(metrics_csv_path):
            with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = [
                    'Epoch', 
                    'Train Time (min)', 
                    'Params Size (MB)', 
                    'Total Params (10^6)', 
                    'FLOPs (G)', 
                    'mIoU (%)', 
                    'mPA (%)', 
                    'Accuracy (%)', 
                    'F1 Score',
                    'clDice (%)',
                    'CC-F1 (%)'
                ]
                writer.writerow(header)

    UnFreeze_flag = False
        
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    nbs             = 16
    lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
    lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
    epoch_step      = num_train // batch_size
    epoch_step_val  = max(1, num_val // batch_size)
        
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("")
        
    train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset   = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size    = batch_size // ngpus_per_node
        train_shuffle = False
        val_shuffle   = False
    else:
        train_sampler = None
        val_sampler   = None
        train_shuffle = True
        val_shuffle   = False

    gen = DataLoader(
        train_dataset,
        shuffle=train_shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_dataset_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
    )

    gen_val = DataLoader(
        val_dataset,
        shuffle=val_shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=seg_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
    )

    if local_rank == 0:
        eval_callback = EvalCallback(
            model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
            eval_flag=eval_flag, period=eval_period
        )
    else:
        eval_callback = None
        
   
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        epoch_start_time = time.time()

        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            nbs             = 16
            lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
            lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
            Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
            for param in model.backbone.parameters():
                param.requires_grad = True
                            
            epoch_step      = num_train // batch_size
            epoch_step_val  = max(1, num_val // batch_size)

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("")

            gen = DataLoader(
                train_dataset,
                shuffle=train_shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=seg_dataset_collate,
                sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
            )
            gen_val = DataLoader(
                val_dataset,
                shuffle=val_shuffle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=seg_dataset_collate,
                sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
            )

            UnFreeze_flag = True

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train, model, loss_history, eval_callback, optimizer, epoch,
            epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
            dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
            save_period, save_dir, local_rank
        )

        if distributed:
            dist.barrier()

        if local_rank == 0 and (epoch + 1) % eval_period == 0:
            epoch_end_time = time.time()
            train_time_min = (epoch_end_time - epoch_start_time) / 60.0
                
            current_miou   = eval_callback.mious[-1] if hasattr(eval_callback, 'mious') and len(eval_callback.mious) > 0 else 0
            current_mpa    = eval_callback.mPAs[-1] if hasattr(eval_callback, 'mPAs') and len(eval_callback.mPAs) > 0 else 0
            current_acc    = eval_callback.Accs[-1] if hasattr(eval_callback, 'Accs') and len(eval_callback.Accs) > 0 else 0
            current_f1     = eval_callback.f1s[-1] if hasattr(eval_callback, 'f1s') and len(eval_callback.f1s) > 0 else 0
            current_cldice = eval_callback.cldices[-1] if hasattr(eval_callback, 'cldices') and len(eval_callback.cldices) > 0 else 0
            current_ccf1   = eval_callback.cc_f1s[-1] if hasattr(eval_callback, 'cc_f1s') and len(eval_callback.cc_f1s) > 0 else 0

            with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row_data = [
                    epoch + 1,
                    f"{train_time_min:.2f}",
                    f"{params_mb:.2f}",
                    f"{params_m:.3f}",
                    f"{flops_g:.3f}",
                    f"{current_miou * 100:.2f}",
                    f"{current_mpa * 100:.2f}",
                    f"{current_acc * 100:.2f}",
                    f"{current_f1:.4f}",
                    f"{current_cldice * 100:.2f}",
                    f"{current_ccf1 * 100:.2f}"
                ]
                writer.writerow(row_data)

            table_data = {
                "Metric": [
                    "Epoch",
                    "Train Time (min)",
                    "Params Size (MB)",
                    "Total Params (10^6)",
                    "FLOPs (G)",
                    "mIoU (%)",
                    "mPA (%)",
                    "Accuracy (%)",
                    "F1 Score",
                    "clDice (%)",
                    "CC-F1 (%)"
                ],
                "Value": row_data
            }
            df = pd.DataFrame(table_data)
            print("\n" + "="*50)
            print(f"Performance Summary at Epoch {epoch + 1}")
            print("="*50)
            print(df.to_string(index=False))
            print("="*50 + "\n")

    if local_rank == 0 and loss_history is not None:
        loss_history.writer.close()


