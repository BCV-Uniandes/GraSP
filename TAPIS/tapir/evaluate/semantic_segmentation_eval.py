import numpy as np
import skimage.io as io
from tqdm import tqdm
import os

from .utils import decode_rle_to_mask

def eval_segmentation(task, coco_anns, preds, img_ann_dict, mask_path=None):
    if 'tools_categories' in coco_anns:
        cats = coco_anns['tools_categories']
    else:
        cats = coco_anns[f'{task[:-1]}_categories'] if f'{task[:-1]}_categories' in coco_anns else coco_anns['categories']
    cats = set(sorted([cat['id'] for cat in cats]))
    annotations = coco_anns['annotations']
    coco_anns = coco_anns['images']

    ious = []
    gt_ious = []
    pcls_ious = {cat:[] for cat in cats}
    for image in tqdm(coco_anns):
        width = image['width']
        height = image['height']

        file_name = image['file_name']
        if mask_path is not None:
            gt_img = io.imread(os.path.join(mask_path, file_name.split('.')[0]+'.png'))
            gt_classes = set(np.unique(gt_img))
            gt_classes.remove(0)
        else:
            image_anns = img_ann_dict[file_name]
            image_anns = [annotations[ann_idx] for ann_idx in image_anns]
            gt_classes = set()
            gt_img = np.zeros((height, width))
            for im_ann in image_anns:
                ann_cat = im_ann['tools'] if 'tools' in im_ann else im_ann['category_id']
                gt_classes.add(ann_cat)
                ann_mask = decode_rle_to_mask(im_ann['segmentation'], 'bool')
                gt_img[ann_mask] = ann_cat
        
        image_preds = preds[file_name]["instances"]
        instances = []
        for pred in image_preds:
            logits = pred['tools_logits']
            category = np.argmax(logits)
            score = logits[category]
            segmentation = pred['segment']
            instances.append({'segmentation': segmentation, 'category_id': category+1, 'score': score})
        instances.sort(key = lambda x: x['score'])

        sem_im = np.zeros((height,width))
        for ins in instances:
            p_mask = decode_rle_to_mask(ins['segmentation'], 'bool')
            sem_im[p_mask]=ins['category_id']
        pred_classes = set(np.unique(sem_im))
        pred_classes.remove(0)

        iou = []
        gt_iou = []
        for label in cats:
            if label in gt_classes or label in pred_classes:
                pred_mask = (sem_im==label).astype('uint8')
                gt_mask = (gt_img==label).astype('uint8')

                intersection = np.sum(pred_mask * gt_mask)
                union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
                im_IoU = intersection/union
                assert im_IoU>=0 and im_IoU<=1, im_IoU
                iou.append(im_IoU)
                pcls_ious[label].append(im_IoU)

                if label in gt_classes and label not in pred_classes:
                    assert im_IoU == 0
                
                if label not in gt_classes and label in pred_classes:
                    assert im_IoU == 0
                
                if label in gt_classes:
                    gt_iou.append(im_IoU)

        assert len(iou)==len(gt_classes.union(pred_classes))
        assert len(gt_iou)==len(gt_classes)

        if len(iou)>0:
            ious.append(float(np.mean(iou)))
        
        if len(gt_iou)>0:
            gt_ious.append(float(np.mean(gt_iou)))

    total_iou = float(np.mean(ious))
    total_gt_iou = float(np.mean(gt_ious))

    for cls in pcls_ious:
        pcls_ious[cls] = float(np.mean(pcls_ious[cls])) if len(pcls_ious[cls])>0 else np.nan
    cious = list(pcls_ious.values())
    total_ciou = float(np.nanmean(cious))

    return total_gt_iou, total_iou, total_ciou, cious