"""
Evaluation taken from AVA and ActivityNet repository
"""
import sys
# sys.path.append('../evaluation/ava_evaluation')
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from .ava_evaluation import ( 
    object_detection_evaluation,
    standard_fields
)
from .utils import xywhbbox_to_dxdydxdybbox as normalize_bbox

def eval_detection(task, coco_anns, preds, img_ann_dict, **kwargs):
    # Transform data to pascal format
    categories = coco_anns[f'{task}_categories'] if f'{task}_categories' in coco_anns else coco_anns['categories']
    num_classes = len(categories)
    print("Formating annotations and preds...")
    groundtruth = organize_data_pascal(coco_anns,img_ann_dict,task)
    detections= organize_pred_pascal(groundtruth[-1],preds,task,num_classes)
    groundtruth[-1] = [] # Delete sizes info
    excluded_keys = []
    print("Evaluating Detection...")
    results, PR_results = run_evaluation(categories, groundtruth, detections, excluded_keys)
    results = list(results.values())
    cat_names = [f'{cat["name"]}-AP_box' for cat in categories]
    return results[0], dict(zip(cat_names,results[1:]))


def organize_data_pascal(coco_anns, img_ann_dict, task):
    '''
        bboxes for groundtruth are in [x1,y1,w,h]
    '''
    bboxes = defaultdict(list)
    labels = defaultdict(list)
    sizes = defaultdict(list) # For datasets with multiple frame sizes

    for img_name, img_idx in tqdm(img_ann_dict.items()): 
        new_key = img_name
        img = [img for img in coco_anns["images"] if img["file_name"] == img_name][0]
        im_w = img["width"]
        im_h = img["height"]
        sizes[new_key] = [im_w,im_h]

        # Each new action, bbox, img has a new key
        if len(img_idx) == 0:
            continue
        for idx in img_idx:
            if task in coco_anns['annotations'][idx]:
                if isinstance(coco_anns['annotations'][idx][task],list): 
                    lbl = coco_anns['annotations'][idx][task]
                elif isinstance(coco_anns['annotations'][idx][task],int):
                    lbl = [coco_anns['annotations'][idx][task]]
                else:
                    raise ValueError(f'Annotation {coco_anns["annotations"][idx][task]} of type {type(coco_anns["annotations"][idx][task])} is not supported')
            elif task=='instruments' and 'category_id' in coco_anns['annotations'][idx]:
                lbl = [coco_anns['annotations'][idx]['category_id']]
                
            bbox = coco_anns['annotations'][idx]['bbox']
            x1, y1, x2, y2 = normalize_bbox(bbox, im_w, im_h)
            new_bbox = [y1,x1,y2,x2]
            
            for a_idx, act in enumerate(lbl):
                bboxes[new_key].extend([new_bbox])
                labels[new_key].extend([act])
    groundtruth = [bboxes, labels, sizes]
    return groundtruth

def organize_pred_pascal(gt_keys, preds, task, num_classes):
    '''
        bboxes for preds are in format [x1,y1,x2,y2]
    '''

    pred_bboxes = defaultdict(list)
    pred_scores = defaultdict(list)
    pred_labels = defaultdict(list)
    pred_keys = list(preds.keys())
    for new_key in tqdm(gt_keys): 
        im_w, im_h = gt_keys[new_key]
        img_name = new_key
        if img_name not in pred_keys:
            continue
        pred_image = preds[img_name]["instances"]
        pred_image.sort(key=lambda x: max(x[f'{task}_score_dist']), reverse=True)
        
        if not len(pred_image):
            continue

        for this_box in pred_image:
            box, prob_task = this_box['bbox'], this_box[f'{task}_score_dist']
            x1, y1, x2, y2 = box
            new_box = [y1/im_h, x1/im_w, y2/im_h, x2/im_w]
            pred_bboxes[new_key].extend([new_box for _ in range(num_classes)])
            pred_scores[new_key].extend(prob_task)
            pred_labels[new_key].extend(list(range(1, num_classes + 1)))
   
    detection = [pred_bboxes, pred_labels, pred_scores]

    return detection

def run_evaluation(
    categories, groundtruth, detections, excluded_keys, verbose=True
):
    """AVA evaluation main logic."""

    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories
    )

    boxes, labels, _ = groundtruth
    gt_keys = []
    pred_keys = []

    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(
                (
                    "Found excluded timestamp in ground truth: %s. "
                    "It will be ignored."
                ),
                image_key,
            )
            continue
        
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key,
            {
                standard_fields.InputDataFields.groundtruth_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.InputDataFields.groundtruth_classes: np.array(
                    labels[image_key], dtype=int #Tiene que haber mismo número de cajas que de labels
                ),
                standard_fields.InputDataFields.groundtruth_difficult: np.zeros(
                    len(boxes[image_key]), dtype=bool
                ),
            },
        )

        gt_keys.append(image_key)

    boxes, labels, scores = detections

    for image_key in tqdm(boxes):
        if image_key in excluded_keys:
            logging.info(
                (
                    "Found excluded timestamp in detections: %s. "
                    "It will be ignored."
                ),
                image_key,
            )
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key,
            {
                standard_fields.DetectionResultFields.detection_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.DetectionResultFields.detection_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.DetectionResultFields.detection_scores: np.array(
                    scores[image_key], dtype=float
                ),
            },
        )

        pred_keys.append(image_key)
    print("Calculating metric...")
    metrics = pascal_evaluator.evaluate()

    return metrics