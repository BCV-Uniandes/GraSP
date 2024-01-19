#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import random
import numpy as np
import shutil
import os
import pprint
import torch

import tapir.models.losses as losses
import tapir.models.optimizer as optim
import tapir.utils.checkpoint as cu
import tapir.utils.distributed as du
import tapir.utils.logging as logging
import tapir.utils.misc as misc

from tapir.datasets import loader
from tapir.models import build_model
from tapir.utils.meters import EpochTimer, SurgeryMeter

logger = logging.get_logger(__name__)

def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py

    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    tasks = cfg.TASKS.TASKS
    loss_funs = cfg.TASKS.LOSS_FUNC

    loss_dict = {task:losses.get_loss_func(loss_funs[t_id])(reduction=cfg.SOLVER.REDUCTION) for t_id,task in enumerate(tasks)}
    type_dict = {task:losses.get_loss_type(loss_funs[t_id]) for t_id,task in enumerate(tasks)}
    loss_weights = cfg.TASKS.LOSS_WEIGHTS
    if cfg.REGIONS.ENABLE and cfg.TASKS.PRESENCE_RECOGNITION:
        pres_loss_dict = {f'{task}_presence':losses.get_loss_func('bce')(reduction=cfg.SOLVER.REDUCTION) for task in cfg.TASKS.PRESENCE_TASKS}
        pres_type_dict = {f'{task}_presence':losses.get_loss_type('bce') for task in cfg.TASKS.PRESENCE_TASKS}
        loss_dict.update(pres_loss_dict)
        type_dict.update(pres_type_dict)
        loss_weights += cfg.TASKS.PRESENCE_WEIGHTS
    
    for cur_iter, (inputs, labels, data, image_names) in enumerate(train_loader):

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            inputs[0] = inputs[0].cuda(non_blocking=True)

            for key, val in data.items():
                data[key] = val.cuda(non_blocking=True)

            for key, val in labels.items():
                labels[key] = val.cuda(non_blocking=True)
            
            if cfg.NUM_GPUS>1:
                image_names = image_names.cuda(non_blocking=True)
                    
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            rpn_ftrs = data["rpn_features"] if cfg.FEATURES.ENABLE else None
            boxes_mask = data["boxes_mask"] if cfg.REGIONS.ENABLE else None
            preds = model(inputs, rpn_ftrs, boxes_mask)

            # Explicitly declare reduction to mean and compute the loss for each task.
            loss = []
            for task in loss_dict:
                loss_fun = loss_dict[task]
                target_type = type_dict[task]
                loss.append(loss_fun(preds[task], labels[task].to(target_type))) 

        if len(loss_dict) >1:
            final_loss = losses.compute_weighted_loss(loss, loss_weights)
        else:
            final_loss = loss[0]
            
        # check Nan Loss.
        misc.check_nan_losses(final_loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(final_loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        if cfg.NUM_GPUS > 1:
            final_loss = du.all_reduce([final_loss])[0]
        final_loss = final_loss.item()

        # Update and log stats.
        train_meter.update_stats(None, None, None, final_loss, loss, lr)
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    complete_tasks = cfg.TASKS.TASKS
    region_tasks = {task for task in cfg.TASKS.TASKS if task in cfg.ENDOVIS_DATASET.REGION_TASKS}
    if cfg.REGIONS.ENABLE:
        if cfg.TASKS.PRESENCE_RECOGNITION and cfg.TASKS.EVAL_PRESENCE:
            pres_tasks = [f'{task}_presence' for task in cfg.TASKS.PRESENCE_TASKS]
            complete_tasks += pres_tasks

    for cur_iter, (inputs, labels, data, image_names) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            inputs[0] = inputs[0].cuda(non_blocking=True)

            for key, val in data.items():
                data[key] = val.cuda(non_blocking=True)
            
            for key, val in labels.items():
                labels[key] = val.cuda(non_blocking=True)
            
            if cfg.NUM_GPUS>1:
                image_names = image_names.cuda(non_blocking=True)
                    
        val_meter.data_toc()

        rpn_ftrs = data["rpn_features"] if cfg.FEATURES.ENABLE else None
        boxes_mask = data["boxes_mask"] if cfg.REGIONS.ENABLE else None
        ori_boxes = data["ori_boxes"] if cfg.REGIONS.ENABLE else None
        boxes_idxs = data["ori_boxes_idxs"] if cfg.REGIONS.ENABLE else None
        boxes = data["boxes"] if cfg.REGIONS.ENABLE else None

        assert (not (cfg.REGIONS.ENABLE and cfg.FEATURES.ENABLE)) or len(rpn_ftrs)==len(image_names)==len(boxes), f'Inconsistent lenghts {len(rpn_ftrs)} & {len(image_names)} & {len(boxes)}'

        preds = model(inputs, rpn_ftrs, boxes_mask)

        if cfg.NUM_GPUS:
            preds = {task: preds[task].cpu() for task in complete_tasks}
            ori_boxes = ori_boxes.cpu() if cfg.REGIONS.ENABLE else None
            boxes_idxs = boxes_idxs.cpu() if cfg.REGIONS.ENABLE else None
            boxes_mask = boxes_mask.cpu() if cfg.REGIONS.ENABLE else None

            if cfg.NUM_GPUS>1:
                image_names = image_names.cpu()
                image_names = torch.cat(du.all_gather_unaligned(image_names),dim=0).tolist()

                for task in preds:
                    if task not in region_tasks:
                        preds[task] = torch.cat(du.all_gather_unaligned(preds[task]), dim=0).tolist()
                    else:
                        preds[task] = torch.cat(du.all_gather_unaligned(preds[task]), dim=0)

                if cfg.REGIONS.ENABLE:
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                    boxes_mask = torch.cat(du.all_gather_unaligned(boxes_mask), dim=0)
                    idxs_gather = du.all_gather_unaligned(boxes_idxs)
                    for i in range(len(idxs_gather)):
                        idxs_gather[i]+= torch.tensor((cfg.TEST.BATCH_SIZE/cfg.NUM_GPUS)*i).long()

                    boxes_idxs = torch.cat(idxs_gather, dim=0)

        val_meter.iter_toc()

        if cfg.REGIONS.ENABLE:
            ori_boxes = [ori_boxes[boxes_idxs==idx].tolist() for idx in range(len(boxes_mask))]
            for task in region_tasks:
                preds[task] = [preds[task][boxes_idxs==idx].tolist() for idx in range(len(boxes_mask))]
        
        # Update and log stats.
        val_meter.update_stats(preds, image_names, ori_boxes)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    if cfg.NUM_GPUS > 1:
        if du.is_master_proc():
            task_map, mean_map, out_files = val_meter.log_epoch_stats(cur_epoch)
        else:
            task_map, mean_map, out_files =  [0, 0, 0]
        torch.distributed.barrier()
    else:
        task_map, mean_map, out_files = val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

    return task_map, mean_map, out_files


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)

    # Calculating model info (param & flops). 
    # Remove if it is not working
    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
            
    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    
    # Create meters.
    train_meter = SurgeryMeter(len(train_loader), cfg, mode="train")
    val_meter = SurgeryMeter(len(val_loader), cfg, mode="val")

    # Perform final test
    if cfg.TEST.ENABLE:
        logger.info("Evaluating epoch: {}".format(start_epoch + 1))
        map_task, mean_map, out_files = eval_epoch(val_loader, model, val_meter, start_epoch, cfg)
        if not cfg.TRAIN.ENABLE:
            return
    elif cfg.TRAIN.ENABLE:
        # Perform the training loop.
        logger.info("Start epoch: {}".format(start_epoch + 1))
        
    # Stats for saving checkpoint:
    complete_tasks = cfg.TASKS.TASKS
    best_task_map = {task: 0 for task in complete_tasks}
    best_mean_map = 0
    epoch_timer = EpochTimer()
    
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None
        )

        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        
        if not cfg.MODEL.KEEP_ALL_CHECKPOINTS:
            del_fil = os.path.join(cfg.OUTPUT_DIR,'checkpoints', 'checkpoint_epoch_{0:05d}.pyth'.format(cur_epoch-1))
            if os.path.exists(del_fil):
                os.remove(del_fil)
            
        # Evaluate the model on validation set.
        if is_eval_epoch:
            map_task, mean_map, out_files = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            if (cfg.NUM_GPUS > 1 and du.is_master_proc()) or cfg.NUM_GPUS == 1:
                main_path = os.path.split(list(out_files.values())[0])[0]
                fold = main_path.split('/')[-1]
                best_preds_path = main_path.replace(fold, fold+'/best_predictions')
                if not os.path.exists(best_preds_path):
                    os.makedirs(best_preds_path)
                # Save best results
                if mean_map > best_mean_map:
                    best_mean_map = mean_map
                    logger.info("Best mean map at epoch {}".format(cur_epoch))
                    cu.save_best_checkpoint(
                        cfg.OUTPUT_DIR,
                        model,
                        optimizer,
                        'mean',
                        cfg,
                        scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
                    for task in complete_tasks:
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best_all') )
                        shutil.copyfile(out_files[task], copy_path)
                
                for task in complete_tasks:
                    if list(map_task[task].values())[0] > best_task_map[task]:
                        best_task_map[task] = list(map_task[task].values())[0]
                        logger.info("Best {} map at epoch {}".format(task, cur_epoch))
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best') )
                        shutil.copyfile(out_files[task], copy_path)
                        cu.save_best_checkpoint(
                            cfg.OUTPUT_DIR,
                            model,
                            optimizer,
                            task,
                            cfg,
                            scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
    cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )

