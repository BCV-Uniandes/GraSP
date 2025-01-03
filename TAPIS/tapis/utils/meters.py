#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

from copy import deepcopy
import datetime
from math import sqrt
import traceback
import numpy as np
import os
import json
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import tapis.evaluate.main_eval as grasp_eval
import tapis.utils.logging as logging
import tapis.utils.misc as misc
from tapis.evaluate.utils import mask_to_rle

logger = logging.get_logger(__name__)

# IDENT_FUNCT_DICT = {'psi_ava': lambda x,y: 'CASE{:03d}/{:05d}.jpg'.format(x,y),
#                     'grasp': lambda x,y: 'CASE{:03d}/{:09d}.jpg'.format(x,y),
#                     'endovis_2018': lambda x,y: 'seq_{}_frame{:03d}.jpg'.format(x,y),
#                     'endovis_2017': lambda x,y: 'seq{}_frame{:03d}.jpg'.format(x,y),}

from tapis.datasets import DATASET_REGISTRY

class SurgeryMeter(object):
    """
    Measure the PSI-AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.dataset_name = cfg.TEST.DATASET
        self.parallel = cfg.NUM_GPUS > 1
        # if self.parallel:
        self.join_funct = DATASET_REGISTRY.get(self.dataset_name.capitalize())(cfg,'test',False).frame_num_joining
        self.eval_train = cfg.TRAIN.EVAL_TRAIN
        self.regions = cfg.REGIONS.ENABLE

        self.tasks = deepcopy(cfg.TASKS.TASKS)
        self.log_tasks = deepcopy(cfg.TASKS.TASKS)
        self.metrics = deepcopy(cfg.TASKS.METRICS)
        
        if cfg.REGIONS.ENABLE:
            self._region_tasks = {task for task in cfg.TASKS.TASKS if task in cfg.ENDOVIS_DATASET.REGION_TASKS}
            if cfg.TASKS.PRESENCE_RECOGNITION:
                pres_tasks = [task for task in cfg.TASKS.PRESENCE_TASKS]
                self.log_tasks +=  pres_tasks
                if cfg.TASKS.EVAL_PRESENCE:
                    self.tasks += pres_tasks
                    self.metrics += ["mAP"]*len(pres_tasks)
        
        self.generate_masks = not cfg.FEATURES.PRECALCULATE_TEST

        self.all_classes = cfg.TASKS.NUM_CLASSES
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.task_loss = TaskMeter(cfg.LOG_PERIOD, len(self.log_tasks)) 
        self.mode = mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.all_preds = {k: [] for k in self.tasks}
        self.full_map = {}
        self.all_boxes,  self.all_detect_names, self.all_names = [], [], []
        self.overall_iters = overall_iters
        self.groundtruth = cfg.ENDOVIS_DATASET.TEST_COCO_ANNS
        self.segmentation = cfg.REGIONS.ENABLE and cfg.REGIONS.LEVEL=='segmentation'
        
        if self.segmentation and os.path.isdir(cfg.ENDOVIS_DATASET.MASKS_PATH):
            self.mask_path = cfg.ENDOVIS_DATASET.MASKS_PATH
        else:
            self.mask_path = None
            
        if self.segmentation and not self.generate_masks:
            with open(os.path.join(cfg.ENDOVIS_DATASET.ANNOTATION_DIR,cfg.ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON)) as f:
                self.region_proposals = json.load(f)['annotations']
            final_proposals = {}
            for rp in self.region_proposals:
                frame = rp['image_name']
                if frame not in final_proposals:
                    final_proposals[frame] = {}
                bbox = rp['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

                final_proposals[frame][tuple(bbox)] = rp['segmentation']
            self.region_proposals = final_proposals

        elif self.generate_masks:
            self.all_masks = []

        self.output_dir = cfg.OUTPUT_DIR

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                "lr": self.lr,
                "overall_loss": self.task_loss.get_tasks_median_sum(),
            }
            all_loss_medians = self.task_loss.get_win_median()
            for idx, task in enumerate(self.log_tasks):
                stats["loss_{}".format(task)] = all_loss_medians[idx]
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.task_loss.reset()
        self.full_map = {}
        self.all_preds = {task:[] for task in self.tasks}
        self.all_names = []
        self.all_boxes = []

        if self.generate_masks:
            self.all_masks = []

    def update_stats(self, preds, names, boxes, final_loss= None, losses=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            keep_box(tensor): tensor of boolean to keep the original bounding boxes. 
            boxes (tensor): predicted boxes (x1, y1, x2, y2).
            d_names (list): names of the keyframes with detection anns.
            names (list): names of all the keyframes.
            final_loss (float): final loss value.
            lr (float): learning rate.
        """ 
        if self.eval_train or self.mode in ["val", "test"]:
            
            for task in self.tasks:
                self.all_preds[task].extend(preds[task])
            
            if self.parallel:
                try:
                    names = [self.join_funct(name[0],name[1]) for name in names] #[IDENT_FUNCT_DICT[self.dataset_name](*name) for name in names]
                except:
                    print(names)
                    raise
            self.all_names.extend(names)
            if self.regions:
                self.all_boxes.extend(boxes)
                assert all(len(names)==len(boxes)==len(preds[t]) for t in self.tasks)
                if 'masks' in preds:
                    for masks in preds['masks']:
                        self.all_masks.append([mask_to_rle(mask.astype('uint8')) for mask in masks])
                    
                    assert all(len(names)==len(boxes)==len(preds['masks'])==len(preds[t]) for t in self.tasks)
            else:
                assert all(len(names)==len(preds[t]) for t in self.tasks)

        if losses is not None:
            self.task_loss.add_value(losses)
        if final_loss is not None:
            self.loss.add_value(final_loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, epoch, log=True):
        """
        Calculate and log the final PSI-AVA metrics.
        """
        out_name = {}
        for task,metric in zip(self.tasks, self.metrics):
            out_name[task] = self.save_json(task, epoch)
            self.full_map[task] = grasp_eval.main_per_task(self.groundtruth, out_name[task], task, metric, masks_path=self.mask_path)
            if log:
                stats = {"mode": self.mode, "task": task, "metric": self.full_map[task]}
                logging.log_json_stats(stats)
        if log:
            stats = {"mode": self.mode, "mean metric": np.mean([v[m] for v,m in zip(list(self.full_map.values()), self.metrics)])}
            logging.log_json_stats(stats)
        
        return self.full_map, np.mean([v[m] for v,m in zip(list(self.full_map.values()), self.metrics)]), out_name
                    
    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            metrics_val, mean_map, out_files = self.finalize_metrics(cur_epoch +1)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            for idx, task in enumerate(self.tasks):
                stats["{}_map".format(task)] = self.full_map[task]

            logging.log_json_stats(stats)
            
            return metrics_val, mean_map, out_files

    def save_json(self, task, epoch):
        """
        Save json for the specific task.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        save_json_dict = {}

        preds, boxes, names = self.all_preds, self.all_boxes, self.all_names

        if self.regions:
            assert len(preds[task])==len(names)==len(boxes), f'Inconsistent lengths {len(preds[task])} {len(names)} {len(boxes)}'
            if self.generate_masks:
                masks = self.all_masks
                assert len(preds[task])==len(names)==len(boxes)==len(masks), f'Inconsistent lengths {len(preds[task])} {len(names)} {len(boxes)} {len(masks)}'
        else:
            assert len(preds[task])==len(names), f'Inconsistent lengths {len(preds[task])} {len(names)}'
        
        for idx, (pred, name) in enumerate(zip(preds[task], names)):
            task_key_name = f'{task}_score_dist'
            if self.regions and task in self._region_tasks:
                if self.segmentation:
                    save_json_dict[name] = {'instances': [{"bbox":box, task_key_name:pred[b_id], "segment": self.region_proposals[name][tuple(box)] if not self.generate_masks else masks[idx][b_id]} for b_id,box in enumerate(boxes[idx]) if box != [0,0,0,0]]}
                else:
                    save_json_dict[name] = {'instances': [{"bbox":box, task_key_name:pred[b_id]} for b_id,box in enumerate(boxes[idx]) if box != [0,0,0,0]]}
            else:
                save_json_dict[name] = {task_key_name:pred}

        path_prediction = os.path.join(self.output_dir, f'epoch_{epoch}_preds_{task}.json')
        with open(path_prediction, "w") as outfile:  
            json.dump(save_json_dict, outfile) 
            
        return path_prediction
    
    
class TaskMeter(object):
    """
    A task meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size, num_tasks):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.num_tasks = num_tasks
        self.task_meters = [ScalarMeter(window_size) for _ in range(num_tasks)]

    def reset(self):
        """
        Reset the individual meters.
        """
        [meter.reset() for meter in self.task_meters]

    def add_value(self, values):
        """
        Add a new scalar value to each of the task's deques.
        """
        [self.task_meters[idx].add_value(val.item()) for idx, val in enumerate(values)]

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
    
        return [np.median(meter.deque) for meter in self.task_meters]
    
    def get_tasks_median_avg(self):
        """
        """
        return np.mean(np.array(self.get_win_median()))
    
    def get_tasks_median_sum(self):
        """
        """
        return np.sum(np.array(self.get_win_median()))

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return [np.mean(meter.deque) for meter in self.task_meters]

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return [meter.total/meter.count for meter in self.task_meters]


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)