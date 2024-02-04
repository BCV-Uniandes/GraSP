import itertools
import os
import os.path as osp
import json
import numpy as np
import sklearn
import argparse
import pandas as pd

from .classification_eval import eval_classification
from .detection_eval import eval_detection
from .semantic_segmentation_eval import eval_segmentation as eval_sem_segmentation
from .instance_segmentation_eval import eval_segmentation as eval_inst_segmentation
from .utils import load_json, save_json

def eval_segmentation(task, coco_anns, preds, img_ann_dict, mask_path):
    inst_seg_results, aux_inst_seg = eval_inst_segmentation(task, coco_anns, preds, img_ann_dict, mask_path)
    print('{} task mAP@0.5IoU_segm: {}'.format(task, round(inst_seg_results,8)))
    
    sem_seg_results, aux_sem_seg = eval_sem_segmentation(task, coco_anns, preds, img_ann_dict, mask_path)
    print('{} task mIoU: {}'.format(task, round(sem_seg_results,8)))
    
    inst_seg_results = {'mAP@0.5IoU_segm': inst_seg_results}
    inst_seg_results.update(aux_sem_seg)
    inst_seg_results.update(aux_inst_seg)
    return sem_seg_results, inst_seg_results
    

METRIC_DICT = {'mAP': eval_classification,
               'mAP@0.5IoU_box': eval_detection,
               'mAP@0.5IoU_segm': eval_inst_segmentation,
               'mIoU': eval_sem_segmentation,
               'mIoU_mAP@0.5': eval_segmentation,
               'classification': eval_classification,
               'detection': eval_detection,
               'inst_segmentation': eval_inst_segmentation,
               'sem_segmentation': eval_sem_segmentation,
               'segmentation': eval_segmentation}

def get_img_ann_dict(coco_anns,task):
    img_ann_dict = {}
    for img in coco_anns["images"]:
        img_ann_dict[img["file_name"]] = []
    
    for idx, ann in enumerate(coco_anns["annotations"]):
        if ((task=='instruments' and 'category_id' in ann) and ann['category_id']>=0) or \
            (task in ann and (ann[task]>=0 if type(ann[task]) is int else min(ann[task])>=0)):
            img_ann_dict[ann["image_name"]].append(idx)
    
    return img_ann_dict

def eval_task(task, metric, coco_anns, preds, masks_path):
    img_ann_dict = get_img_ann_dict(coco_anns,task)
    try:
        metric_funct = METRIC_DICT[metric]
    except KeyError:
        raise NotImplementedError(f'Metric {metric} is not supported')
    
    main_metric, aux_metrics = metric_funct(task, coco_anns, preds, img_ann_dict, masks_path)
    return main_metric, aux_metrics


def main_per_task(coco_ann_path, pred_path, task, metric, masks_path=None):
    # Load coco anns and preds
    coco_anns = load_json(coco_ann_path)
    preds = load_json(pred_path) if type(pred_path)==str else pred_path

    task_eval, aux_metrics = eval_task(task, metric, coco_anns, preds, masks_path)
    aux_metrics = dict(zip(aux_metrics.keys(),map(lambda x: round(x,8), aux_metrics.values())))
    print('{} task {}: {} {}'.format(task, metric, round(task_eval,8), aux_metrics))
    
    final_metrics = {metric: round(task_eval,8)}
    final_metrics.update(aux_metrics)    
    return final_metrics