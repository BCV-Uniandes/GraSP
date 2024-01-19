#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
from detectron2.layers import ROIAlign


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.projection_faster = nn.Linear(256, 1024, bias = True)
        
        self.projection_pathways = nn.Linear(sum(dim_in), 1024, bias = True)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(1024, num_classes, bias=True)
        self.act_func = act_func
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes, features=None):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection_pathways(x)
        
        if features is not None:
            features = features[:,1:]
            features = self.projection_faster(features)
        x = torch.cat((x, features), axis=1)

        x = self.projection(x)
        
        if self.training and self.act_func == "sigmoid" or not self.training:
            x = self.act(x)
        
        return x
        


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.extra_pool = nn.AvgPool3d([pool_size[0][0], pool_size[0][0], 1], stride=1)
        
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # handle extra heads need of extra pooling
        x = self.extra_pool(x)
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False,
        recognition=False
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        self.cls_embed = cls_embed
        self.recognition = recognition
        self.act_func = act_func

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x, features, boxes_mask):
        if self.cls_embed and not self.recognition:
            x = x[:, 0]
        elif self.cls_embed:
            x = x[:,1:].mean(1)
        else:
            x = x.mean(1)

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if self.act_func == "sigmoid" or not self.training:
            x = self.act(x)
        return x


class TransformerRoIHead(nn.Module):
    """
    Box classification head in TAPIR. 
    """

    def __init__(
        self,
        cfg,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False
    ):
        
        super(TransformerRoIHead, self).__init__()
        self.cfg = cfg
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.cls_embed = cls_embed
        
        # Feature Vector dimension from faster is 1024
        dim_faster = 1024 
        # Feature Vector dimension from deformable detr is 256
        dim_detr = 256
        if cfg.FEATURES.MODEL=='m2f':
            dim_detr = 512
             
        if self.cfg.FEATURES.ENABLE:
            self.dim_add = 1024
            dim_out = self.dim_add + 768 #1152
            
        else: 
            self.dim_add = 0
            dim_out = self.dim_add + 1024
        
        
        if cfg.FEATURES.MODEL in ['detr','m2f']:
            # We redimension deformable detr features to the same as in faster r-cnn

            if cfg.MODEL.DECODER:
                self.trans_project = nn.Sequential(nn.Linear(dim_detr,768,bias=True))
                decoder_layer = nn.TransformerDecoderLayer(768, 8, batch_first=True)
                self.decoder = nn.TransformerDecoder(decoder_layer, 1)
                dim_final = 768
            elif cfg.MODEL.TRANS:
                self.trans_project = nn.Sequential(nn.Linear(768,dim_faster,bias=True),
                                               nn.ReLU(),
                                               nn.Linear(dim_faster,dim_faster,bias=True))
            
                self.mlp = nn.Sequential(nn.Linear(dim_detr, dim_faster, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(dim_faster, dim_faster, bias=True),
                                        )
                dim_final = dim_faster*2
            else:
                self.mlp = nn.Sequential(nn.Linear(dim_detr, dim_faster, bias=False),
                                        nn.BatchNorm1d(dim_faster))
                dim_final = dim_out
                
        elif not self.cfg.FEATURES.ENABLE:
            self.mlp = nn.Sequential(nn.Linear(768, dim_out, bias=False),
                                    nn.BatchNorm1d(dim_out))
            dim_final = dim_out
        
        else:
            if cfg.MODEL.DECODER:
                self.trans_project = nn.Sequential(nn.Linear(dim_faster,768,bias=True))
                decoder_layer = nn.TransformerDecoderLayer(768, 8, batch_first=True)
                self.decoder = nn.TransformerDecoder(decoder_layer, 1)
                dim_final = 768
        
        # Fanal classification layer 
        if cfg.MODEL.TRANS or cfg.MODEL.DECODER:
            self.projection = nn.Sequential(nn.Linear(dim_final, num_classes, bias=True),)
        else:
            self.projection = nn.Linear(dim_out, num_classes, bias=True)
        self.act_func = act_func
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )
    
    def forward(self, inputs, features=None, boxes_mask=None):
        if self.cls_embed:
            inputs = inputs[:, 1:, :]
        else:
            inputs = inputs[:, :, :]
        
        # Mean pool through time
        if self.cfg.MODEL.DECODER:
            x = inputs
            features = self.trans_project(features)
            x = self.decoder(features, x, tgt_key_padding_mask=~boxes_mask)
            x = x[boxes_mask]
        
        else:
            x = inputs.mean(1)
            if self.cfg.MODEL.TRANS:
                x = self.trans_project(x)

            max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self.training else self.cfg.DATA.MAX_BBOXES
            # Repeat pooled time features to match the batch dimensions of box proposals
            x_boxes = x.unsqueeze(1).repeat(1,max_boxes,1)[boxes_mask] # Use box mask to remove padding
            
            if features is not None:
                features = features[boxes_mask] # Use box mask to remove padding
                if self.cfg.FEATURES.MODEL in ['detr','m2f']:
                    features = self.mlp(features)
                x = torch.cat([x_boxes, features], dim=1)
            elif features is None and self.cfg.FEATURES.ENABLE: # Hack for calculating model info
                x_boxes = torch.zeros(len(features), x.shape[1], device = inputs.device, requires_grad=True)
                features = torch.zeros(x_boxes.shape[0], self.dim_add, device = inputs.device, requires_grad=True)
                x = torch.cat([x_boxes, features], dim=1)
            else:
                raise NotImplementedError('There are no features')

        x = self.projection(x)

        # Only apply final activation for validation or for bce loss
        if self.training and self.act_func == "sigmoid" or not self.training:
            x = self.act(x)

        return x
        
