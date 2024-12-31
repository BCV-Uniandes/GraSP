#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
from functools import partial
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from . import utils as utils
from .build import build_dataset

def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    images, all_labels, extra_data, image_names = zip(*batch)
    image_names = image_names if type(image_names[0])==str else torch.tensor(image_names)

    images = default_collate(images)

    collated_extra_data = {}
    for key in extra_data[0]:
        data = [d[key] for d in extra_data]

        if key == "boxes_mask":
            collated_extra_data[key] = default_collate(data).bool()
        elif key == "ori_boxes":
            idxs = list(itertools.chain(*[[d_id]*len(d) for d_id,d in enumerate(data)]))
            data = list(itertools.chain(*data))
            collated_extra_data[key] = torch.tensor(data).float()
            collated_extra_data["ori_boxes_idxs"] = torch.tensor(idxs)
        elif key == "images":
            collated_extra_data[key] = torch.nested.nested_tensor(data).float()
        else:
            collated_extra_data[key] = default_collate(data).float()
    
    collated_labels = {}
    for key in all_labels[0]:
        data = [d[key] for d in all_labels]

        #TODO: These are just security checks, REMOVE when all done
        assert all(type(data[0]) == type(d) for d in data), f'Inconsistent data type {data}'
        
        if isinstance(data[0],list):
            data = list(itertools.chain(*data))
        else:
            #TODO: These are just security checks, REMOVE when all done
            assert isinstance(data[0],int) or isinstance(data[0],np.ndarray), f'Type {type(data[0])} not supported'
            
        collated_labels[key] = torch.tensor(data).float()

    return images, collated_labels, collated_extra_data, image_names


def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    if isinstance(dataset, torch.utils.data.IterableDataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    else:
        # Create a sampler for multi-process training
        sampler = utils.create_sampler(dataset, shuffle, cfg)
        # Create a loader
        collate_func = detection_collate

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=collate_func,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    if (
        loader._dataset_kind
        == torch.utils.data.dataloader._DatasetKind.Iterable
    ):
        if hasattr(loader.dataset, "sampler"):
            sampler = loader.dataset.sampler
        else:
            raise RuntimeError(
                "Unknown sampler for IterableDataset when shuffling dataset"
            )
    else:
        sampler = (
            loader.sampler
        )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
