#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import abstractmethod
import torch
import logging
import numpy as np

from . import surgical_dataset_helper as data_helper
from . import cv2_transform as cv2_transform
from . import utils as utils

logger = logging.getLogger(__name__)

class SurgicalDataset(torch.utils.data.Dataset):
    """
    We adapt the AVA Dataset management in Slowfast to manage Endoscopic Vision databases.
    """

    def __init__(self, cfg, split, load=True):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = {key: n_class for key, n_class in zip(cfg.TASKS.TASKS, cfg.TASKS.NUM_CLASSES)}
        self._region_tasks = {task for task in cfg.TASKS.TASKS if task in cfg.ENDOVIS_DATASET.REGION_TASKS}
        self._frame_tasks = {task for task in cfg.TASKS.TASKS if task not in cfg.ENDOVIS_DATASET.REGION_TASKS}

        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.ENDOVIS_DATASET.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = (cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE_LARGE)
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.ENDOVIS_DATASET.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.ENDOVIS_DATASET.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.DATA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.DATA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = (cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE_LARGE)
            self._test_force_flip = cfg.ENDOVIS_DATASET.TEST_FORCE_FLIP
            self.aspect_ratio_th = cfg.ENDOVIS_DATASET.ASPECT_RATION_TH
        
        if load:
            self._load_data(cfg)
    
    @abstractmethod
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        pass
    
    @abstractmethod
    def frame_spliting(self, video_name, sec):
        pass
    
    @abstractmethod
    def frame_num_joining(self, video_num, sec):
        pass
    
    @abstractmethod
    def frame_name_joining(self, video_name, sec):
        pass

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = data_helper.load_image_lists(cfg, is_train=(self._split == "train"))

        # Loading annotations for boxes and labels.
        boxes_and_labels = data_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        assert len(boxes_and_labels) == len(self._image_paths)
        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = data_helper.get_keyframe_data(boxes_and_labels, self.keyframe_mapping)
        
        if self.cfg.REGIONS.ENABLE:
            # Calculate the number of used boxes.
            self._num_boxes_used = data_helper.get_num_boxes_used(
                self._keyframe_indices, self._keyframe_boxes_and_labels
            )

        # Read Region features
        if cfg.FEATURES.ENABLE:
            self.feature_boxes = data_helper.load_features_boxes(cfg, self._split)

        self.print_summary()

    def print_summary(self):
        logger.info(f"=== {self.dataset_name} dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        if self.cfg.REGIONS.ENABLE:
            logger.info("Number of instances: {}.".format(self._num_boxes_used))

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes, image=None):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        height, width, _ = imgs[0].shape
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, we only have
        # one np.array.
        boxes = [boxes.astype('float')]
        
        # The image now is in HWC, BGR format.
        if self._split == "train" and not self.cfg.DATA.JUST_CENTER:  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes, image = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes, image=image
            )

            if self.random_horizontal_flip:

                if image is not None:
                    imgs.append(image)

                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )

                if image is not None:
                    image = imgs.pop()

        elif self._split == "val" or self.cfg.DATA.JUST_CENTER:
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size[0], img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size[0], boxes[0], height, width
                )
            ]
            imgs, boxes, _ = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes, image=None
            )
            
            ori_aspect_ratio = (width/height)
            crop_aspect_ratio = (self.cfg.DATA.TEST_CROP_SIZE_LARGE/self.cfg.DATA.TEST_CROP_SIZE)
            assert image is None or ori_aspect_ratio-crop_aspect_ratio<self.aspect_ratio_th , f'Test aspect ratio difference is too large for inference with RPN'

            if not self.cfg.DATA.JUST_CENTER and self._test_force_flip:
                if image is not None:
                    imgs.append(image)

                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )

                if image is not None:
                    image = imgs.pop()
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        if image is not None:
            image = cv2_transform.BGR2RGB(image)
            image = cv2_transform.HWC2CHW(image)
            image = torch.tensor(image)
        return imgs, boxes, image