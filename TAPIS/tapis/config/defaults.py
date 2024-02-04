#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 72

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "kinetics"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = False

# Evaluate training performance
_C.TRAIN.EVAL_TRAIN = False

# Evaluate training performance
_C.TRAIN.FILTER_EMPTY = True

# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()

# Whether to enable randaug.
_C.AUG.ENABLE = False

# Number of repeated augmentations to used during training.
# If this is greater than 1, then the actual batch size is
# TRAIN.BATCH_SIZE * AUG.NUM_SAMPLE.
_C.AUG.NUM_SAMPLE = 1

# Not used if using randaug.
_C.AUG.COLOR_JITTER = 0.4

# RandAug parameters.
_C.AUG.AA_TYPE = "rand-m9-mstd0.5-inc1"

# Interpolation method.
_C.AUG.INTERPOLATION = "bicubic"

# Probability of random erasing.
_C.AUG.RE_PROB = 0.25

# Random erasing mode.
_C.AUG.RE_MODE = "pixel"

# Random erase count.
_C.AUG.RE_COUNT = 1

# Do not random erase first (clean) augmentation split.
_C.AUG.RE_SPLIT = False


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "psi_ava"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"

# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""


# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"

# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["2d", "c2d", "i3d", "slow", "x3d", "mvit", "mvitv2", "videoswintransformer"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.KEEP_ALL_CHECKPOINTS = False

# Use time mlp.
_C.MODEL.TIME_MLP = False

# Add linear layers before temporal pooling.
_C.MODEL.PREV_MLP = True

# Number of linear layers before temporal pooling.
_C.MODEL.PREV_MLP_LAYERS = 1

# Hidden dimension of the linear layers before temporal pooling.
_C.MODEL.PREV_MLP_HID_DIM = 1024

# Output dimension of the linear layers before temporal pooling.
_C.MODEL.PREV_MLP_OUT_DIM = 1024

# Number of linear layers after temporal pooling.
_C.MODEL.POST_MLP_LAYERS = 1

# Hidden dimension of the linear layers after temporal pooling.
_C.MODEL.POST_MLP_HID_DIM = 1024

# Output dimension of the linear layers after temporal pooling.
_C.MODEL.POST_MLP_OUT_DIM = 1024

# Number of linear layers to transform features.
_C.MODEL.FEAT_MLP_LAYERS = 1

# Hidden dimension of the linear layers for feature transformation.
_C.MODEL.FEAT_MLP_HID_DIM = 1024

# Output dimension of the linear layers for feature transformation
_C.MODEL.FEAT_MLP_OUT_DIM = 1024

# Use cross-attention layer.
_C.MODEL.DECODER = False

# Use cross-attention layer.
_C.MODEL.DECODER_HID_DIM = 2048

# Use cross-attention layer.
_C.MODEL.DECODER_NUM_HEADS = 8

# Use cross-attention layer.
_C.MODEL.DECODER_NUM_LAYERS = 1

# Model float precision
_C.MODEL.PRECISION = 32


# -----------------------------------------------------------------------------
# MViT options
# -----------------------------------------------------------------------------
_C.MVIT = CfgNode()

# Options include `conv`, `max`.
_C.MVIT.MODE = "conv"

# If True, perform pool before projection in attention.
_C.MVIT.POOL_FIRST = False

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = True

# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [3, 7, 7]

# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [2, 4, 4]

# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [2, 4, 4]

# If True, use 2d patch, otherwise use 3d patch.
_C.MVIT.PATCH_2D = False

# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1

# Depth of the transformer.
_C.MVIT.DEPTH = 16

# Normalization layer for the transformer. Only layernorm is supported now.
_C.MVIT.NORM = "layernorm"

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = None

# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = []

# If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
# Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
_C.MVIT.POOL_KVQ_KERNEL = None

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = True

# If True, use norm after stem.
_C.MVIT.NORM_STEM = False

# If True, perform separate positional embedding.
_C.MVIT.SEP_POS_EMBED = False

# If True, use absolute positional embedding.
_C.MVIT.USE_ABS_POS = True

# If True, use relative positional embedding for spatial dimentions
_C.MVIT.REL_POS_SPATIAL = False

# If True, use relative positional embedding for temporal dimentions
_C.MVIT.REL_POS_TEMPORAL = False

# If True, init rel with zero
_C.MVIT.REL_POS_ZERO_INIT = False

# Dropout rate for the MViT backbone.
_C.MVIT.DROPOUT_RATE = 0.0

# Freeze embedding patch. https://arxiv.org/abs/2104.02057
_C.MVIT.FREEZE_PATCH = False

# If True, use frozen sin cos positional embedding.
_C.MVIT.USE_FIXED_SINCOS_POS = False

# Dim mul in qkv linear layers of attention block instead of MLP
_C.MVIT.DIM_MUL_IN_ATT = False

# If True, using Residual Pooling connection
_C.MVIT.RESIDUAL_POOLING = False

# If True, using separate linear layers for Q, K, V in attention blocks.
_C.MVIT.SEPARATE_QKV = False

# -----------------------------------------------------------------------------
# Video Swin Transformers options
# -----------------------------------------------------------------------------
# TODO DOCUMENT VST.
_C.VST = CfgNode()

# Load checkpoint from Video Swin Transformers
_C.VST.PRETRAINED = False
_C.VST.PRETRAINED2D = False

#
_C.VST.DEPTHS = [2, 2, 18, 2]

#
_C.VST.EMBED_DIM = 128

#
_C.VST.PATCH_NORM = True

#
_C.VST.FROZEN_STAGES = -1

#
_C.VST.WINDOW_SIZE = (8,7,7)

#
_C.VST.PATCH_SIZE = (4,4,4)

#
_C.VST.NUM_HEADS = [4, 8, 16, 32]

#
_C.VST.DROP_RATE = 0.

#
_C.VST.USE_CHECKPOINT = False

#
_C.VST.MLP_RATIO = 4.0

#
_C.VST.QKV_BIAS = True

#
_C.VST.QK_SCALE = None

#
_C.VST.ATTN_DROP_RATE = 0.

# Number of channels in the input video frames
_C.VST.IN_CHANS = 3

# 
_C.VST.DROP_PATH_RATE = 0.2


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.DATA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.DATA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# If a imdb have been dumpped to a local file with the following format:
# `{"im_path": im_path, "class": cont_id}`
# then we can skip the construction of imdb and load it from the local file.
_C.DATA.PATH_TO_PRELOAD_IMDB = ""

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The relative scale range of Inception-style area based random resizing augmentation.
# If this is provided, DATA.TRAIN_JITTER_SCALES above is ignored.
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = []

# The relative aspect ratio range of Inception-style area based random resizing
# augmentation.
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = []

# If True, perform stride length uniform temporal sampling.
_C.DATA.USE_OFFSET_SAMPLING = False

# Whether to apply motion shift for augmentation.
_C.DATA.TRAIN_JITTER_MOTION_SHIFT = False

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# Max possible boxes
_C.DATA.MAX_BBOXES = 5

# Just use a center crop
_C.DATA.JUST_CENTER = False

# Verify consistency in data loading
_C.DATA.VERIFICATIONS = True


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# Loss reduction
_C.SOLVER.REDUCTION = "mean"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = ""

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.REGIONS = CfgNode()

# Whether enable video detection.
_C.REGIONS.ENABLE = False

# Whether enable video detection.
_C.REGIONS.LEVEL = "detection"

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.REGIONS.ALIGNED = True

# Spatial scale factor.
_C.REGIONS.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.REGIONS.ROI_XFORM_RESOLUTION = 7

# Filter images with incomplete annotations.
_C.REGIONS.FILTER_INCOMPLETE = False

# -----------------------------------------------------------------------------
# Endoscopic Surgical Dataset options
# -----------------------------------------------------------------------------
_C.ENDOVIS_DATASET = CfgNode()

# Directory path of frames.
_C.ENDOVIS_DATASET.FRAME_DIR = ""

# Directory path for files of frame lists.
_C.ENDOVIS_DATASET.FRAME_LIST_DIR = ""

# Directory path for annotation files.
_C.ENDOVIS_DATASET.ANNOTATION_DIR = ""

# Filenames of training samples list files.
_C.ENDOVIS_DATASET.TRAIN_LISTS = "train.csv"

# Filenames of test samples list files.
_C.ENDOVIS_DATASET.TEST_LISTS = "val.csv"

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.ENDOVIS_DATASET.TRAIN_GT_BOX_JSON = "train_coco_anns.json"

_C.ENDOVIS_DATASET.TEST_GT_BOX_JSON = ""

# Filenames of box list files for train.
_C.ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON = "train_coco_preds.json"

# Filenames of box list files for test.
_C.ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON = "val_coco_preds.json"

# This option controls the score threshold for the predicted boxes to use.
_C.ENDOVIS_DATASET.DETECTION_SCORE_THRESH = 0.0

# If use BGR as the format of input frames.
_C.ENDOVIS_DATASET.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.ENDOVIS_DATASET.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.ENDOVIS_DATASET.TRAIN_PCA_JITTER_ONLY = True

# Whether to do horizontal flipping during test.
_C.ENDOVIS_DATASET.TEST_FORCE_FLIP = False

# The name of the file to the ava groundtruth.
_C.ENDOVIS_DATASET.GROUNDTRUTH_FILE = ""

# Backend to process image, includes `pytorch` and `cv2`.
_C.ENDOVIS_DATASET.IMG_PROC_BACKEND = "cv2"

# Test annotation file of groundtruth in coco 
_C.ENDOVIS_DATASET.TEST_COCO_ANNS = ""

# Supported Tasks
_C.ENDOVIS_DATASET.TASKS = ["phases", "steps", "instruments", "actions"]

# Region Tasks
_C.ENDOVIS_DATASET.REGION_TASKS = ["instruments", "actions"]

# Use GT Boxes
_C.ENDOVIS_DATASET.INCLUDE_GT = True

# Use Predicted Boxes
_C.ENDOVIS_DATASET.USE_PREDS = True

# -----------------------------------------------------------------------------
# Classification heads options
# -----------------------------------------------------------------------------
_C.TASKS = CfgNode()

# Extra hierarchical heads
_C.TASKS.ENABLE = True

# Task names for each extra head
_C.TASKS.TASKS = ["actions", "phases", "steps", "instruments"]

# Task metrics
_C.TASKS.METRICS = ["mAP@50_det", "mAP@50_seg", "mAP", "mIoU"]

# Number of classes per extra head
_C.TASKS.NUM_CLASSES = [14, 11, 21, 7]

# Activation function for each extra head
_C.TASKS.HEAD_ACT = ["sigmoid", "softmax", "softmax", "softmax"]

# Loss function for each extra head
_C.TASKS.LOSS_FUNC = ["bce", "cross_entropy", "cross_entropy", "cross_entropy"]

# Overall loss function weights for each extra head and original head
_C.TASKS.LOSS_WEIGHTS = [0.3, 0.2, 0.3, 0.2]

# Include presence recognition.
_C.TASKS.PRESENCE_RECOGNITION = False

# Tasks to supervise presence.
_C.TASKS.PRESENCE_TASKS = ["instruments"]

# Tasks to supervise presence.
_C.TASKS.PRESENCE_WEIGHTS = [1]

# Tasks to supervise presence.
_C.TASKS.EVAL_PRESENCE = False

# ---------------------------------------------------------------------------- #
# FEATURES
# ---------------------------------------------------------------------------- #
_C.FEATURES = CfgNode()

# When true, the model load the features of the detector
_C.FEATURES.ENABLE = False

# When true, the model adapts to load the features of the deformable detr detector.
# If false, it adapts to load the faster rcnn features.
_C.FEATURES.DIM_FEATURES = 256

_C.FEATURES.MODEL = 'detr'

# Path to .pth file with the detected train boxes features
_C.FEATURES.TRAIN_FEATURES_PATH = ''

# Path to .pth file with the detected validation boxes features
_C.FEATURES.TEST_FEATURES_PATH = ''

# Add custom config with default values.
custom_config.add_custom_config(_C)


def assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.NUM_GPUS == 0 or cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.NUM_GPUS == 0 or cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
