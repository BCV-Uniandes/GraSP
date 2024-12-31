#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import logging
import math
import numpy as np
import os
from datetime import datetime
import psutil
import torch
from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
from matplotlib import pyplot as plt
from torch import nn

import tapis.utils.logging as logging
import tapis.utils.multiprocessing as mpu
from tapis.models.batchnorm_helper import SubBatchNorm3d
from tapis.datasets.utils import pack_pathway_output

logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        breakpoint()
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def _get_model_analysis_input(cfg, use_input_frames):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model (counting flops and activations etc.).
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, return the input for training. Otherwise,
            return the input for testing.

    Returns:
        inputs: the input for model analysis.
    """
    rgb_dimension = 3
    if use_input_frames:
        input_tensors = torch.rand(
                        rgb_dimension,
                        cfg.DATA.NUM_FRAMES,
                        cfg.DATA.TRAIN_CROP_SIZE,
                        cfg.DATA.TRAIN_CROP_SIZE_LARGE,
        )
        input_tensors = pack_pathway_output(cfg, input_tensors)
        input_tensors = [input.unsqueeze(0) for input in input_tensors]
        if cfg.NUM_GPUS:
            input_tensors = [input.cuda(non_blocking=True) for input in input_tensors]
    else:
        final_dim = int(cfg.MVIT.EMBED_DIM * math.prod([dim_mul[-1] for dim_mul in cfg.MVIT.DIM_MUL]))
        time_resolution = cfg.DATA.NUM_FRAMES // cfg.MVIT.PATCH_STRIDE[0]
        h_resolution = (cfg.DATA.TEST_CROP_SIZE // cfg.MVIT.PATCH_STRIDE[1]) // math.prod([q_pool[-1] for q_pool in cfg.MVIT.POOL_Q_STRIDE])
        w_resolution = (cfg.DATA.TEST_CROP_SIZE_LARGE // cfg.MVIT.PATCH_STRIDE[2]) // math.prod([q_pool[-1] for q_pool in cfg.MVIT.POOL_Q_STRIDE])
        
        input_tensors = torch.rand(1, time_resolution*h_resolution*w_resolution, final_dim)
        
        if cfg.NUM_GPUS:
            input_tensors = input_tensors.cuda(non_blocking=True)

    # If detection is enabled, count flops for max region proposal.
    if cfg.REGIONS.ENABLE:
        max_boxes = cfg.DATA.MAX_BBOXES * 2 if cfg.ENDOVIS_DATASET.INCLUDE_GT else cfg.DATA.MAX_BBOXES
        features = np.random.rand(1, max_boxes, cfg.FEATURES.DIM_FEATURES)
        features = torch.tensor(features).float()
        bbox_mask = torch.ones(features.shape[:-1]).bool()
        if cfg.NUM_GPUS:
            features = features.cuda()
            bbox_mask = bbox_mask.cuda()
        if cfg.FEATURES.USE_RPN and use_input_frames:
            ratio = cfg.DATA.TEST_CROP_SIZE_LARGE/cfg.DATA.TEST_CROP_SIZE
            images = torch.tensor(np.random.rand(1, 3, cfg.FEATURES.RPN_CFG.INPUT.IMAGE_SIZE, round(ratio*cfg.FEATURES.RPN_CFG.INPUT.IMAGE_SIZE))).float()
            bboxes = torch.tensor(np.zeros((1, max_boxes, 4)))
            bboxes[:,:,1] = cfg.DATA.TEST_CROP_SIZE
            bboxes[:,:,3] = cfg.DATA.TEST_CROP_SIZE_LARGE
            if cfg.NUM_GPUS:
                images = images.cuda()
                bboxes = bboxes.cuda()
            inputs = (input_tensors, None, bbox_mask, images, bboxes)
        else:
            inputs = (input_tensors, features, bbox_mask)
    else:
        inputs = (input_tensors,)
    return inputs


def get_model_stats(model, cfg, mode, use_train_input):
    """
    Compute statistics for the current model given the config.
    Args:
        model (model): model to perform analysis.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        mode (str): Options include `flop` or `activation`. Compute either flop
            (gflops) or activation count (mega).
        use_train_input (bool): if True, compute statistics for training. Otherwise,
            compute statistics for testing.

    Returns:
        float: the total number of count of the given model.
    """
    assert mode in [
        "flop",
        "activation",
    ], "'{}' not supported for model analysis".format(mode)
    if mode == "flop":
        model_stats_fun = flop_count
    elif mode == "activation":
        model_stats_fun = activation_count

    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    model_mode = model.training
    model.eval()
    inputs = _get_model_analysis_input(cfg, use_train_input)
    count_dict, *_ = model_stats_fun(model, inputs)
        
    count = sum(count_dict.values())
    model.train(model_mode)
    return count


def log_model_info(model, cfg, use_train_input=True):
    """
    Log info, includes number of parameters, gpu usage, gflops and activation count.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        use_train_input (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    logger.info(
        "Flops: {:,} G".format(
            get_model_stats(model, cfg, "flop", use_train_input)
        )
    )
    logger.info(
        "Activations: {:,} M".format(
            get_model_stats(model, cfg, "activation", use_train_input)
        )
    )
    logger.info("nvidia-smi")
    os.system("nvidia-smi")


def is_eval_epoch(cfg, cur_epoch, multigrid_schedule):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (int): current epoch.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True
    if multigrid_schedule is not None:
        prev_epoch = 0
        for s in multigrid_schedule:
            if cur_epoch < s[-1]:
                period = max(
                    (s[-1] - prev_epoch) // cfg.MULTIGRID.EVAL_FREQ + 1, 1
                )
                return (s[-1] - 1 - cur_epoch) % period == 0
            prev_epoch = s[-1]

    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0


def plot_input(tensor, bboxes=(), texts=(), path="./tmp_vis.png"):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor.float()
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    f, ax = plt.subplots(nrows=1, ncols=tensor.shape[0], figsize=(50, 20))
    for i in range(tensor.shape[0]):
        ax[i].axis("off")
        ax[i].imshow(tensor[i].permute(1, 2, 0))
        # ax[1][0].axis('off')
        if bboxes is not None and len(bboxes) > i:
            for box in bboxes[i]:
                x1, y1, x2, y2 = box
                ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

        if texts is not None and len(texts) > i:
            ax[i].text(0, 0, texts[i])
    f.savefig(path)


def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def aggregate_sub_bn_stats(module):
    """
    Recursively find all SubBN modules and aggregate sub-BN stats.
    Args:
        module (nn.Module)
    Returns:
        count (int): number of SubBN module found.
    """
    count = 0
    for child in module.children():
        if isinstance(child, SubBatchNorm3d):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_sub_bn_stats(child)
    return count


def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)