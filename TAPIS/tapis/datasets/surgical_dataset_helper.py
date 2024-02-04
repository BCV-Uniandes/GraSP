#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import gc
import json
import os
from re import split
import torch
import logging
from collections import defaultdict
from tapis.utils.env import pathmgr

logger = logging.getLogger(__name__)

def load_features_boxes(cfg,split):
    """
    Load boxes features from region proposal model.

    Args:
        cfg (CfgNode): config.

    Returns:
        features (tensor): a tensor of faster weights.
    """
    
    if split=='train':
        features = torch.load(cfg.FEATURES.TRAIN_FEATURES_PATH) 
    else:
        features = torch.load(cfg.FEATURES.TEST_FEATURES_PATH) 
    features = {feat['file_name'] : feat['features'] for feat in features}
    
    for file,boxes in features.items():
        assert all(len(feats)==cfg.FEATURES.DIM_FEATURES for feats in boxes.values()), f'Incorrect feature length in image{file}. Excpected size {cfg.FEATURES.DIM_FEATURES}'
    return features


def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    list_filenames = [
        os.path.join(cfg.ENDOVIS_DATASET.FRAME_LIST_DIR, cfg.ENDOVIS_DATASET.TRAIN_LISTS if is_train else cfg.ENDOVIS_DATASET.TEST_LISTS)
    ]
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for list_filename in list_filenames:
        with pathmgr.open(list_filename, "r") as f:
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path
                assert len(row) == 4
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]
                image_paths[data_key].append(os.path.join(cfg.ENDOVIS_DATASET.FRAME_DIR,row[3]))

    image_paths = [image_paths[i] for i in range(len(image_paths))]
    logger.info("Finished loading image paths from: %s" % ", ".join(list_filenames))

    return image_paths, video_idx_to_name


def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            {`bbox`:<box_coord>, *`box_labels`:<label> where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """

    if mode=='train':
        gt_lists =[cfg.ENDOVIS_DATASET.TRAIN_GT_BOX_JSON] if cfg.ENDOVIS_DATASET.INCLUDE_GT or \
                                                            not cfg.ENDOVIS_DATASET.USE_PREDS else []
        pred_lists = [cfg.ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON] if cfg.ENDOVIS_DATASET.USE_PREDS else []
    elif cfg.ENDOVIS_DATASET.USE_PREDS:
        gt_lists =[]
        pred_lists = [cfg.ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON]
    else:
        gt_lists = [cfg.ENDOVIS_DATASET.TEST_GT_BOX_JSON]
        pred_lists = []

    ann_filenames = [
        os.path.join(cfg.ENDOVIS_DATASET.ANNOTATION_DIR, filename)
        for filename in gt_lists + pred_lists
    ]
    ann_is_gt_box = [True] * len(gt_lists) + [False] * len(pred_lists)
    detect_thresh = cfg.ENDOVIS_DATASET.DETECTION_SCORE_THRESH

    all_boxes, count, count_unqiue = parse_bboxes_file(
        ann_filenames=ann_filenames,
        ann_is_gt_box=ann_is_gt_box,
        detect_thresh=detect_thresh,
        cfg=cfg,
        split=mode
    )
    
    logger.info("Finished loading annotations from: %s" % ", ".join(ann_filenames))
    logger.info("Detection threshold: {}".format(detect_thresh))
    logger.info("Number of annotations: %d" % count)
    logger.info("Number of unique annotations: %d" % count_unqiue)

    return all_boxes

def parse_bboxes_file(ann_filenames, ann_is_gt_box, detect_thresh, cfg, split):
    """
    Parse bounding boxes files.
    Args:
        ann_filenames (list of str(s)): a list of bounding boxes coco annotation files.
        ann_is_gt_box (list of bools): a list of boolean to indicate whether the corresponding
            ann_file is ground-truth. `ann_is_gt_box[i]` correspond to `ann_filenames[i]`.
        detect_thresh (float): threshold for accepting predicted boxes, range [0, 1].
        boxes_sample_rate (int): sample rate for test bounding boxes. Get 1 every `boxes_sample_rate`.
    """
    count = 0
    filter = split=='train' and cfg.TRAIN.FILTER_EMPTY
    annotated_frames_dict = {}
    complete_frames = {} # 
    id2frame = {}
    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        with pathmgr.open(filename, "r") as f:
            data = json.load(f)
        for image in data['images']:
            id2frame[image['id']] = (image['video_name'], image['frame_num'], image['width'], image['height'])
            if image['video_name'] not in annotated_frames_dict:
                annotated_frames_dict[image['video_name']] = {image['frame_num']:[]}
                complete_frames[image['video_name']] = {image['frame_num']: True}
            elif image['frame_num'] not in annotated_frames_dict[image['video_name']]:
                annotated_frames_dict[image['video_name']][image['frame_num']] = []
                complete_frames[image['video_name']][image['frame_num']] = True
            elif not is_gt_box:
                logger.warning('Thres seem to be a repeated video name or frame number, better check if this is valid and comment this line if so.')
                breakpoint()
        
        for annotation in data['annotations']:
            if verify_annots(annotation,cfg,filter):
                video_name, frame_num, width, height = id2frame[annotation['image_id']]
                labels = {task:annotation[task] for task in cfg.TASKS.TASKS}
                labels['is_gt'] = is_gt_box

                if cfg.REGIONS.ENABLE:
                    if not is_gt_box:
                        assert 'score' in annotation, f'No score in prediction with id {annotation["id"]}'
                        if annotation['score']<detect_thresh:
                            continue

                    x1,y1,w,h = annotation['bbox']
                    bbox = [x1,y1,x1+w,y1+h]

                    labels['bbox'] = bbox
                    #TODO: REMOVE when all done
                    try:
                        assert bbox not in [item['bbox'] for item in annotated_frames_dict[video_name][frame_num]], f'bbox {bbox} is reapeted in frame {frame_num} of video {video_name}'
                    except AssertionError:
                        logger.warning(f'Repeated bounding box in dataset, check if this is ok and comment this line if so.')
                        if cfg.ENDOVIS_DATASET.INCLUDE_GT:
                            pass
                        else:
                            breakpoint()

                annotated_frames_dict[video_name][frame_num].append(labels)
                count += 1
            else:
                video_name, frame_num, width, height = id2frame[annotation['image_id']]
                complete_frames[video_name][frame_num] = False
        
    count_unique = 0
    all_labels = {}
    # Filter keyframes without complete annotations for all tasks
    for video_name in annotated_frames_dict:
        all_labels[video_name]={}
        for frame in annotated_frames_dict[video_name]:
            if len(annotated_frames_dict[video_name][frame]) and \
                (complete_frames[video_name][frame] or not cfg.REGIONS.FILTER_INCOMPLETE):
                all_labels[video_name][frame] = annotated_frames_dict[video_name][frame]
                count_unique += len(annotated_frames_dict[video_name][frame])
    
    del complete_frames
    assert count and count_unique, f"There are no annotations for this list of tasks: {cfg.TASKS.TASKS}"
    
    data = None
    annotated_frames_dict = None
    gc.collect()

    return all_labels, count, count_unique

def verify_annots(annotation,cfg,filter):
    """
        Verify that all annotations vage correct integer values and have labels for all desired tasks
        and have bounding boxes or segmentations for localized tasks
    """
    if filter:
        tasks = all(task in annotation and (annotation[task]>-1 if type(annotation[task]) is int else min(annotation[task])>-1) for task in cfg.TASKS.TASKS)
    else:
        tasks = all(task in annotation for task in cfg.TASKS.TASKS)
    if not cfg.REGIONS.ENABLE:
        return tasks
    elif cfg.REGIONS.LEVEL == 'detection':
        return tasks and ('bbox' in annotation)
    elif cfg.REGIONS.LEVEL == 'segmentation':
        return tasks and ('segmentation' in annotation) and ('bbox' in annotation)
    else:
        raise NotImplementedError(f'{cfg.REGIONS.LEVEL} level not supported')

def get_keyframe_data(boxes_and_labels,keyframe_mapping):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        keyframe_boxes_and_labels.append([])
        for sec_idx,sec in enumerate(boxes_and_labels[video_idx]):
            keyframe_indices.append((video_idx, sec_idx, sec, keyframe_mapping(video_idx, sec_idx, sec)))
            keyframe_boxes_and_labels[video_idx].append(
                boxes_and_labels[video_idx][sec]
            )
            count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels

def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count