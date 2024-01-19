#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from bdb import Breakpoint
import os
import torch
import logging
import numpy as np

from copy import deepcopy
from .surgical_dataset import SurgicalDataset
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class HeiChole(SurgicalDataset):
    """
    We adapt the AVA Dataset management in Slowfast to manage PSI-AVA database.
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = {key: n_class for key, n_class in \
                            zip(cfg.TASKS.TASKS, cfg.TASKS.NUM_CLASSES)}
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.AVA.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.DATA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.DATA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP

        # Read Faster features
        if cfg.FASTER.ENABLE:
            self.feature_boxes = ava_helper.load_features_boxes(cfg)
        else: 
            self.features_boxes = None
        
        self._load_data(cfg)

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
        ) = ava_helper.load_image_lists(cfg, is_train=(self._split == "train"))

        # Loading annotations for boxes and labels.
        boxes_and_labels = ava_helper.load_boxes_and_labels(
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
        ) = ava_helper.get_keyframe_data(boxes_and_labels)
        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        logger.info("=== PSI-AVA dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

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

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
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

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train" and not self.cfg.DATA.JUST_CENTER:  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val" or self.cfg.DATA.JUST_CENTER:
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if not self.cfg.DATA.JUST_CENTER and self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

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
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train" and not self.cfg.DATA.JUST_CENTER:
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val" or self.cfg.DATA.JUST_CENTER:
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if not self.cfg.DATA.JUST_CENTER and self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(
            boxes, self._crop_size, self._crop_size
        )

        return imgs, boxes

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        # Get the path of the middle frame 
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])
        complete_name = video_name+'/'+str(sec).zfill(5)+'.jpg'
        path_complete_name = os.path.join(folder_to_images,complete_name)
        center_idx = self._image_paths[video_idx].index(path_complete_name)

        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        labels = []

        # Tasks to solve in training.
        all_tasks = self.cfg.TASKS.TASKS
        
        # Add labels depending on the task
        all_labels = {k:np.zeros(len(clip_label_list)) for k in all_tasks}
        keep_box = [True]*len(clip_label_list)
        
        if self.cfg.FASTER.ENABLE:
            faster_features = []
            # Get all the possible boxes that correspond to the current video frame. 
            try:
                box_features = [x for x in self.feature_boxes if x['file_name'] == complete_name][0]['bboxes']
            except:
                print(complete_name, 'prediction not found in feature boxes file')
        else:
            faster_features = None
        
        for b_idx, box_labels in enumerate(clip_label_list):
            if len(box_labels[0]) == 0:
                # Consider this for the recognition tasks.
                # Create box of the image size
                box_labels[0] = [0.0, 0.0, 1.0, 1.0]
                # No atomic action
                box_labels[1] = [-1]
                keep_box[b_idx] = False

            boxes.append(box_labels[0])
            labels.append(box_labels[1])
            faster_box_key = " ".join(map(str,box_labels[0]))
            if self.cfg.FASTER.ENABLE:
                if faster_box_key not in box_features[0].keys():
                    # Get features of the box with the highest IoU
                    faster_box_key = utils.get_best_features(box_labels[0],box_features)
                try:
                    if isinstance(box_features[0][faster_box_key], list):
                        box_features[0][faster_box_key] = torch.tensor(box_features[0][faster_box_key])
                    faster_features.append([box_features[0][faster_box_key].cpu().detach().numpy()])
                except:
                    # If there are no predictions for that frame, we add a vector of zeros.
                    # This data is further ignored with the "keep box"
                    if self.cfg.FASTER.DETR:
                        # Deformable DETR extracted features size is 256
                        faster_features.append([np.zeros(256)])
                    else:
                        # Faster-RCNN extracted features size is 1024
                        faster_features.append([np.zeros(1024)])
                        
                    logger.info(f"=== No box features found for frame {path_complete_name} ===")

            for task in self.cfg.TASKS.TASKS:
                if task == 'phases':
                    all_labels[task][b_idx] = box_labels[3][1]
                elif task == 'steps':
                    all_labels[task][b_idx] = box_labels[3][0]
                elif task == 'tools':
                    all_labels[task][b_idx] = box_labels[2][0]
                    
        # Construct label arrays. Modifications for including the 3 different actions
        if 'actions' in self.cfg.TASKS.TASKS:
            final_labels = np.zeros((len(labels), self._num_classes['actions']), dtype=np.int32)
            for i, box_labels in enumerate(labels):
                
                # AVA label index starts from 1.
                for label in box_labels:
                    if label == -1:
                        continue
                    
                    assert label >= 1 and label <= self._num_classes['actions'], print(label)
                    final_labels[i][label-1] = 1

            all_labels['actions'] = final_labels
                                     
        boxes = np.array(boxes)
        if self.cfg.FASTER.ENABLE:
            faster_features = np.array(faster_features).squeeze()
            if len(faster_features.shape) == 1:
                faster_features = np.expand_dims(faster_features, axis=0)
            
        # Score is not used.
        boxes = boxes[:, :4].copy()
        ori_boxes = deepcopy(boxes)
        metadata = [[video_idx, sec ]] * len(boxes)
        all_names = [[complete_name]] * len(boxes)
        
        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.AVA.IMG_PROC_BACKEND
        )
        
        if self.cfg.AVA.IMG_PROC_BACKEND == "pytorch":
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            imgs, boxes = self._images_and_boxes_preprocessing(
                imgs, boxes=boxes
            )
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(
                imgs, boxes=boxes
            )
        
        # Padding and masking for a consistent dimensions in batch
        max_boxes = self.cfg.DATA.MAX_BBOXES
        bbox_mask = np.zeros(max_boxes,dtype=bool)
        if self.cfg.FASTER.ENABLE:
            bbox_mask[:len(boxes)] = True
            if len(boxes)<max_boxes:
                c_boxes = np.concatenate((boxes,np.zeros((max_boxes-len(boxes),4))),axis=0)
                c_faster_features = np.concatenate((faster_features,np.zeros((max_boxes-len(faster_features),256 if self.cfg.FASTER.DETR else 1024))),axis=0)
                boxes = c_boxes
                faster_features = c_faster_features
        
        imgs = utils.pack_pathway_output(self.cfg, imgs)

        extra_data = {
            "boxes_mask": bbox_mask,
            "boxes": boxes,
            "ori_boxes": ori_boxes,
            "metadata": metadata,
            "img_names": all_names,
            "keep_box": keep_box
        }
            
        if self.cfg.FASTER.ENABLE:
            extra_data["faster_features"] = faster_features
        
        return imgs, all_labels, idx, extra_data
        
