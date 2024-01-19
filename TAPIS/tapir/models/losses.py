#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}

_TYPES = {
    "cross_entropy": torch.long,
    "bce": torch.float,
    "bce_logit": torch.float,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

def get_loss_type(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _TYPES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _TYPES[loss_name]

def compute_weighted_loss(losses, weight_vector):
    """
    Weighted loss function
    """
    final_loss = 0
    for ind, loss in enumerate(losses):
        final_loss+= loss * weight_vector[ind]
    return final_loss
