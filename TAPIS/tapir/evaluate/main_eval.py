import itertools
import os
import os.path as osp
import json
import numpy as np
import sklearn
import argparse

from .classification_eval import eval_classification
from .detection_eval import eval_detection
from .semantic_segmentation_eval import eval_segmentation as eval_sem_segmentation
from .instance_segmentation_eval import eval_segmentation as eval_inst_segmentation
from .utils import read_detectron2_output

def eval_segmentation():
    pass

METRIC_DICT = {'mAP': eval_classification,
               'mAP@50_det': eval_detection,
               'mAP@50_seg': eval_inst_segmentation,
               'mIoU': eval_sem_segmentation,
               'mIoU_mAP_seg': eval_segmentation,
               'classification': eval_classification,
               'detection': eval_detection,
               'inst_segmentation': eval_inst_segmentation,
               'sem_segmentation': eval_sem_segmentation,
               'segmentation': eval_segmentation}

def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def save_json(data, json_file, indent=4):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent)

def get_img_ann_dict(coco_anns,task):
    img_ann_dict = {}
    for img in coco_anns["images"]:
        img_ann_dict[img["file_name"]] = []
    
    for idx, ann in enumerate(coco_anns["annotations"]):
        if ((task=='tools' and 'category_id' in ann) and ann['category_id']>=0)or \
            (task in ann and (ann[task]>=0 if type(ann[task]) is int else min(ann[task])>=0)):
            img_ann_dict[ann["image_name"]].append(idx)
    
    return img_ann_dict

def eval_task(task, metric, coco_anns, preds, masks_path):
    img_ann_dict = get_img_ann_dict(coco_anns,task)
    try:
        metric_funct = METRIC_DICT[metric]
    except KeyError:
        raise NotImplementedError(f'Metric {metric} is not supported')
    
    main_metric, metric_1, metric_2, class_metric = metric_funct(task, coco_anns, preds, img_ann_dict, masks_path)
    return main_metric, metric_1, metric_2, class_metric


def main_per_task(coco_ann_path, pred_path, task, metric, masks_path=None):
    # Load coco anns and preds
    coco_anns = load_json(coco_ann_path)
    preds = load_json(pred_path) if type(pred_path)==str else pred_path
    # breakpoint()
    task_eval, task_met1, task_met2, task_csl_met = eval_task(task, metric, coco_anns, preds, masks_path)
    all_metrics = {metric:round(task_eval,6), 
                   f'{metric}_1':round(task_met1,6), 
                   f'{metric}_2':round(task_met2,6), 
                   f'{metric}_per_class':[round(cm,6) for cm in task_csl_met]}
    print('{} task {}: {} {} {} {}'.format(task, metric, *list(all_metrics.values())))
    return all_metrics

def main(coco_ann_path, pred_path, tasks, metrics, masks_path):
    # Load coco anns and preds
    coco_anns = load_json(coco_ann_path)
    preds = load_json(pred_path) if type(pred_path)==str else pred_path
    # breakpoint()
    all_metrics = {}
    for task, metric in zip(tasks,metrics):
        task_eval, task_met1, task_met2, task_csl_met = eval_task(task, metric, coco_anns, preds, masks_path)
        all_metrics[task] = [task_eval, task_met1, task_met2, task_csl_met]
        print('{} task {}: {} {} {} {}'.format(task, metric, *list(all_metrics.values())))
    overall_metric = np.mean([val[0] for val in list(all_metrics.values())])
    print('Overall Metric: {}'.format(overall_metric))
    return overall_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation parser')
    parser.add_argument('--coco-ann-path', default=None, type=str, help='path to coco style anotations')
    parser.add_argument('--pred-path', default=None, type=str, help='path to predictions')
    parser.add_argument('--filter', action='store_true', help='path to predictions')
    parser.add_argument('--tasks', nargs='+', help='tasks to be evaluated', required=True, default=None)
    parser.add_argument('--metrics', nargs='+', help='metrics to be evaluated',
                        choices=['mAP', 'mAP@50_det', 'mAP@50_seg', 'mIoU', 'mIoU_mAP_seg',
                                 'classification', 'detection','inst_segmentation', 
                                 'sem_segmentation', 'segmentation'],
                        required=True, default=None)
    parser.add_argument('--masks-path', default=None, type=str, help='path to predictions')
    parser.add_argument('--selection', type=str, default='thresh', 
                        choices=['thresh', 'topk', 'topk_thresh', 'cls_thresh', 'cls_topk', 
                                'cls_topk_thresh', 'all'], 
                        help='Prediction selection method')
    parser.add_argument('--selection_info', help='Hypermarameters to perform selection', default=0.75)

    args = parser.parse_args()
    print(args)
    
    assert len(args.tasks) == len(args.metrics), f'{args.tasks} {args.metrics}'
    preds = args.pred_path
    if args.filter:
        assert len(args.metrics)==1, args.metrics
        assert args.tasks == ['tools'], args.tasks
        segmentation = 'segmentation' in args.metrics[0] or 'seg' in args.metrics[0] or 'mIoU' in args.metrics[0]
        if args.selection =='thresh':
            selection_params = [None, float(args.selection_info)]
        elif args.selection == 'topk':
            selection_params = [int(args.selection_info), None]
        elif args.selection == 'topk_thresh':
            assert type(args.selection_info) == str and ',' in args.selection_info and len(args.selection_info.split(','))==2
            selection_params = args.selection_info.split(',')
            selection_params[0] = int(selection_params[0])
            selection_params[1] = float(selection_params[1])
        elif 'cls' in args.selection:
            assert type(args.selection_info) == str
            assert os.path.isfile(args.selectrion_info)
            with open(args.selection_info, 'r') as f:
                selection_params = json.load(f)
        else:
            raise ValueError(f'Incorrect selection type {args.selection}')
        preds = read_detectron2_output(args.coco_ann_path, preds, args.selection, selection_params, segmentation)
    main(args.coco_ann_path, preds, args.tasks, args.metrics, args.masks_path)