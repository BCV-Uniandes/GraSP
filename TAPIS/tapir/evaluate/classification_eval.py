import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score


def eval_classification(task, coco_anns, preds, img_ann_dict, mask_path):
    
    classes = coco_anns[f'{task[:-1]}_categories']
    num_classes = len(classes)
    bin_labels = np.zeros((len(coco_anns["annotations"]), num_classes))
    bin_preds = np.zeros((len(coco_anns["annotations"]), num_classes))
    evaluated_frames = []
    for idx, ann in enumerate(coco_anns["annotations"]):
        ann_class = int(ann[task])
        bin_labels[idx, :] = label_binarize([ann_class], classes=list(range(0, num_classes)))
        # TODO Quitar cuando se hable de los datos.
        if  ann["image_name"] in preds.keys():
            these_probs = preds[ann["image_name"]]['{}_logits'.format(task)]
            if len(these_probs) == 0:
                print("Prediction not found for image {}".format(ann["image_name"]))
                these_probs = np.zeros((1, num_classes))
            else:
                evaluated_frames.append(idx)
            bin_preds[idx, :] = these_probs
        else:
            print("Image {} not found in predictions lists".format(ann["image_name"]))
            these_probs = np.zeros((1, num_classes))
            bin_preds[idx, :] = these_probs
            
    bin_labels = bin_labels[evaluated_frames]
    bin_preds = bin_preds[evaluated_frames]
    
    precision = {}
    recall = {}
    threshs = {}
    ap = {}
    for c in range(0, num_classes):
        precision[c], recall[c], threshs[c] = precision_recall_curve(bin_labels[:, c], bin_preds[:, c])
        ap[c] = average_precision_score(bin_labels[:, c], bin_preds[:, c])

    mAP = np.nanmean(list(ap.values()))
    mP = np.nanmean(list(precision.values()))
    mR = np.nanmean(list(recall.values()))
    return mAP, mP, mR, list(ap.values())

