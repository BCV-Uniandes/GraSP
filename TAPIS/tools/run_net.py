#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from tapis.config.defaults import assert_and_infer_cfg
from tapis.utils.misc import launch_job
from tapis.utils.parser import load_config, parse_args

from train_net import train


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE or cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)



if __name__ == "__main__":
    main()
