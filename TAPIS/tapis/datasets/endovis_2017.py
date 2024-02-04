#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import os
import torch
import logging
import numpy as np
import traceback

from copy import deepcopy
from .surgical_dataset import SurgicalDataset
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Endovis_2017(SurgicalDataset):
    """
    EndoVis 2017 dataloader.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "EndoVis 2017"
        self.zero_fill = 3
        self.image_type = "png"
        self.fps = 1
        super().__init__(cfg,split)

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
        complete_name = 'seq{}_frame{}.{}'.format(video_name[3:], str(sec).zfill(self.zero_fill), self.image_type)

        #TODO: REMOVE when all done
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-1])
        path_complete_name = os.path.join(folder_to_images,complete_name)
        found_idx = self._image_paths[video_idx].index(path_complete_name)

        assert path_complete_name == self._image_paths[video_idx][center_idx], f'Different paths {path_complete_name} & {self._image_paths[video_idx][center_idx]}'
        assert found_idx == center_idx, f'Different indexes {found_idx} & {center_idx}'
        assert int(self._image_paths[video_idx][center_idx].split('/')[-1].split('_')[-1].replace('.'+self.image_type,'').replace('frame',''))==sec, f'Different {self._image_paths[video_idx][center_idx].split("/")[-1].replace("."+self.image_type,"")} {sec}'

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
        
        # Add labels depending on the task
        all_labels = {task:[] for task in self._region_tasks}
        keep_box = [True]*len(clip_label_list)
        
        if self.cfg.FEATURES.ENABLE:
            rpn_features = []
            box_features = self.feature_boxes[complete_name] 

        if self.cfg.REGIONS.ENABLE:
            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                all_labels_presence = {f'{task}_presence':np.zeros(self._num_classes[task]) for task in self._region_tasks}
                all_labels.update(all_labels_presence)
            for box_labels in clip_label_list:
                if box_labels['bbox'] != [0,0,0,0]:
                    boxes.append(box_labels['bbox'])
                    if self.cfg.FEATURES.ENABLE:
                        rpn_box_key = " ".join(map(str,box_labels['bbox']))
                        if rpn_box_key not in box_features.keys():
                            rpn_box_key = utils.get_best_features(box_labels["bbox"],box_features)
                        try:
                            features = np.array(box_features[rpn_box_key])
                            rpn_features.append(features)
                        except:

                            if self.cfg.FEATURES.MODEL=='detr':
                                # Deformable DETR extracted features size is 256
                                rpn_features.append(np.zeros(256))
                                
                            elif self.cfg.FEATURES.MODEL=='faster':
                                # Faster-RCNN extracted features size is 1024
                                rpn_features.append(np.zeros(1024))
                                
                            logger.info(f"=== No box features found for frame {path_complete_name} ===")

                    for task in self._region_tasks:
                        if isinstance(box_labels[task],list):
                            binary_task_label = np.zeros(self._num_classes[task]+1,dtype='uint8')
                            box_task_labels = np.array(box_labels[task])-1
                            binary_task_label[box_task_labels] = 1
                            all_labels[task].append(binary_task_label)
                            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                                all_labels[f'{task}_presence'][box_task_labels] = 1
                        elif isinstance(box_labels[task],int):
                            all_labels[task].append(box_labels[task]-1)
                            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                                all_labels[f'{task}_presence'][box_labels[task]-1] = 1
                        else:
                            raise ValueError(f'Do not support annotation {box_labels[task]} of type {type(box_labels[task])} in frame {complete_name}')
            
        else:
            for task in self._region_tasks:
                binary_task_label = np.zeros(self._num_classes[task]+1, dtype='uint8')
                label_list = [label[task] for label in clip_label_list]
                assert all(type(label_list[0])==type(lab_item) for lab_item in label_list), f'Inconsistent label type {label_list} in frame {complete_name}'
                if isinstance(label_list[0], list):
                    label_list = set(list(itertools.chain(*label_list)))
                    binary_task_label[label_list] = 1
                elif isinstance(label_list[0], int):
                    label_list = set(label_list)
                    binary_task_label[label_list] = 1
                else:
                    raise ValueError(f'Do not support annotation {label_list[0]} of type {type(label_list[0])} in frame {complete_name}')
                all_labels[task] = binary_task_label[1:]

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]

        extra_data = {}
        if self.cfg.REGIONS.ENABLE:
            max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self._split == 'train' else self.cfg.DATA.MAX_BBOXES
            if  len(boxes):
                ori_boxes = deepcopy(boxes)
                boxes = np.array(boxes)
                if self.cfg.FEATURES.ENABLE:
                    rpn_features = np.array(rpn_features)
            else:
                ori_boxes = []
                boxes = np.zeros((max_boxes, 4))
                
        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND
        )
        
        # Preprocess images and boxes
        imgs, boxes = self._images_and_boxes_preprocessing_cv2(
            imgs, boxes=boxes
        )
        
        # Padding and masking for a consistent dimensions in batch
        if self.cfg.REGIONS.ENABLE and len(ori_boxes):

            # TODO: REMOVE when all done
            assert len(boxes)==len(ori_boxes)==len(rpn_features), f'Inconsistent lengths {len(boxes)} {len(ori_boxes)} {len(rpn_features)}'
            assert len(boxes)<= max_boxes and len(ori_boxes)<=max_boxes and len(rpn_features)<=max_boxes, f'Incorrect lengths respect max box num{len(boxes)} {len(ori_boxes)} {len(rpn_features)}'

            bbox_mask = np.zeros(max_boxes,dtype=bool)
            bbox_mask[:len(boxes)] = True
            extra_data["boxes_mask"] = bbox_mask

            if len(boxes)<max_boxes:
                c_boxes = np.concatenate((boxes,np.zeros((max_boxes-len(boxes),4))),axis=0)
                boxes = c_boxes
            extra_data["ori_boxes"] = ori_boxes
            extra_data["boxes"] = boxes

            if self.cfg.FEATURES.ENABLE:
                if len(rpn_features)<max_boxes:
                    c_rpn_features = np.concatenate((rpn_features,np.zeros((max_boxes-len(rpn_features), 256 if self.cfg.FEATURES.MODEL=='detr' else 1024))),axis=0)
                    rpn_features = c_rpn_features
                extra_data["rpn_features"] = rpn_features
        
        else:
            bbox_mask = np.zeros(max_boxes,dtype=bool)
            extra_data["boxes_mask"] = bbox_mask
            extra_data["ori_boxes"] = ori_boxes
            extra_data["boxes"] = boxes

            if self.cfg.FEATURES.ENABLE:
                extra_data["rpn_features"] = np.zeros((max_boxes, 256 if self.cfg.FEATURES.MODEL=='detr' else 1024))

        
        imgs = utils.pack_pathway_output(self.cfg, imgs)
        
        return imgs, all_labels, extra_data, complete_name