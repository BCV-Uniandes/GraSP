import numpy as np
import skimage.io as io
from tqdm import tqdm
import os

from .utils import decode_rle_to_mask  # Custom function to decode Run-Length Encoding (RLE) into masks

# Function to evaluate segmentation performance
def eval_segmentation(task, coco_anns, preds, img_ann_dict, **kwargs):
    """
    Evaluates segmentation predictions against ground truth annotations.

    Args:
        task (str): The task type (e.g., 'actions' or 'instrument').
        coco_anns (dict): COCO-style annotations including images, categories, and annotations.
        preds (dict): Predictions for each image, structured as a dictionary.
        img_ann_dict (dict): Dictionary mapping image file names to their annotation indices.
        **kwargs: Optional arguments, e.g., 'mask_path' for ground truth mask directory.

    Returns:
        tuple: Total ground truth IoU and a dictionary with evaluation metrics (IoU and per-class IoUs).
    """

    # Determine the categories for evaluation
    if 'instruments_categories' in coco_anns:
        cats = coco_anns['instruments_categories']
    else:
        cats = coco_anns[f'{task}_categories'] if f'{task}_categories' in coco_anns else coco_anns['categories']
    
    # Prepare category names and IDs
    cat_names = [f"{cat['name']}-IoU" for cat in cats]
    cats = set([cat['id'] for cat in cats])

    # Extract annotations and images
    annotations = coco_anns['annotations']
    coco_anns = coco_anns['images']

    # Initialize metrics
    ious = []
    gt_ious = []
    pcls_ious = {cat: [] for cat in cats}

    # Iterate over each image for evaluation
    for image in tqdm(coco_anns, desc='Evaluating Semantic Segmentation'):
        width, height = image['width'], image['height']
        file_name = image['file_name']

        # Load ground truth mask
        if 'mask_path' in kwargs and kwargs['mask_path'] is not None:
            mask_path = kwargs['mask_path']
            gt_img = io.imread(os.path.join(mask_path, file_name.split('.')[0] + '.png'))
            gt_classes = set(np.unique(gt_img))
            gt_classes.discard(0)  # Remove background class
        else:
            # Generate ground truth mask from annotations
            image_anns = img_ann_dict[file_name]
            image_anns = [annotations[ann_idx] for ann_idx in image_anns]
            gt_classes = set()
            all_masks = []
            for im_ann in image_anns:
                ann_cat = im_ann['instruments'] if 'instruments' in im_ann else im_ann['category_id']
                gt_classes.add(ann_cat)
                p_mask = decode_rle_to_mask(im_ann['segmentation'], 'uint8')
                all_masks.append(p_mask * (im_ann['category_id']))
            gt_img = np.max(np.array(all_masks), axis=0)

        # Load predicted masks and classes
        image_preds = preds[file_name]["instances"]
        if image_preds:
            instances = []
            for pred in image_preds:
                logits = pred['instruments_score_dist']
                category = np.argmax(logits)
                score = logits[category]
                segmentation = pred['segment']
                instances.append({'segmentation': segmentation, 'category_id': category + 1, 'score': score})

            # Combine predicted masks into a single semantic image
            score_masks = []
            for ins in instances:
                p_mask = decode_rle_to_mask(ins['segmentation'], 'float32')
                this_score_mask = np.zeros((height, width, len(cats) + 1), dtype=np.float32)
                this_score_mask[:, :, ins['category_id']] = p_mask * ins['score']
                score_masks.append(this_score_mask)
                
            # This is necessary due to possible overlapping masks 
            max_scores = np.max(np.array(score_masks), axis=0)
            sem_im = np.argmax(max_scores, axis=2)
        else:
            sem_im = np.zeros((height, width))

        pred_classes = set(np.unique(sem_im))
        pred_classes.discard(0)  # Remove background class

        # Calculate IoU for each category
        iou = []
        gt_iou = []
        for label in cats:
            if label in gt_classes or label in pred_classes:
                pred_mask = (sem_im == label).astype('uint8')
                gt_mask = (gt_img == label).astype('uint8')

                intersection = np.sum(pred_mask * gt_mask)
                union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
                im_IoU = intersection / union if union != 0 else 0
                assert 0 <= im_IoU <= 1, im_IoU
                iou.append(im_IoU)
                pcls_ious[label].append(im_IoU)

                # Check consistency between ground truth and predictions
                if label in gt_classes and label not in pred_classes:
                    assert im_IoU == 0
                if label not in gt_classes and label in pred_classes:
                    assert im_IoU == 0
                if label in gt_classes:
                    gt_iou.append(im_IoU)

        assert len(iou) == len(gt_classes.union(pred_classes))
        assert len(gt_iou) == len(gt_classes)

        # Append IoU results
        if iou:
            ious.append(float(np.mean(iou)))
        if gt_iou:
            gt_ious.append(float(np.mean(gt_iou)))

    # Calculate overall metrics
    total_iou = float(np.mean(ious))
    total_gt_iou = float(np.mean(gt_ious))

    # Per-class IoU
    for cls in pcls_ious:
        pcls_ious[cls] = float(np.mean(pcls_ious[cls])) if pcls_ious[cls] else np.nan
    cious = list(pcls_ious.values())
    total_ciou = float(np.nanmean(cious))

    # Prepare final results
    metric_results = {'IoU': total_iou, 'mcIoU': total_ciou}
    class_ious = dict(zip(cat_names, cious))
    metric_results.update(class_ious)

    return total_gt_iou, metric_results
