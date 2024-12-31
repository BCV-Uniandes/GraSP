import itertools
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from tqdm import tqdm

def eval_classification(task, coco_anns, preds, **kwargs):
    
    classes = coco_anns[f'{task}_categories']
    num_classes = len(classes)
    bin_labels = np.zeros((len(coco_anns["annotations"]), num_classes))
    bin_preds = np.zeros((len(coco_anns["annotations"]), num_classes))
    ann_preds_dict = {}
    evaluated_frames = []
    for idx, ann in tqdm(enumerate(coco_anns["annotations"]), total=len(coco_anns["annotations"])):
        ann_class = int(ann[task])
        bin_labels[idx, :] = label_binarize([ann_class], classes=list(range(0, num_classes)))
        

        if  ann["image_name"] in preds.keys():
            # TODO: This might be different on other datasets, change in the future
            video = ann["image_name"].split('/')[0]
            these_probs = preds[ann["image_name"]]['{}_score_dist'.format(task)]
            
            if len(these_probs) == 0:
                print("Prediction not found for image {}".format(ann["image_name"]))
                these_probs = np.zeros((1, num_classes))
            else:
                evaluated_frames.append(idx)
            bin_preds[idx, :] = these_probs
            
            if video in ann_preds_dict:
                ann_preds_dict[video].append((ann_class,np.argmax(these_probs)))
            else:
                ann_preds_dict[video] = [(ann_class,np.argmax(these_probs))]
        else:
            print("Image {} not found in predictions lists".format(ann["image_name"]))
            
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
    
    cat_names = [f"{cat['name']}-AP" for cat in classes]
    cat_res_dict = dict(zip(cat_names,list(ap.values())))
    
    f1_dict = {}
    for video, anns_preds in ann_preds_dict.items():
        np_anns_preds = np.array(anns_preds)
        anns = np_anns_preds[:,0]
        preds = np_anns_preds[:,1]
        f1_dict[video] = f1_score(anns, preds, average='macro')
    
    f1 = np.nanmean(list(f1_dict.values()))
    
    cat_res_dict.update({'f1_score':f1})
    cat_res_dict.update(f1_dict)
            
    return mAP, cat_res_dict

def eval_presence(task, coco_anns, preds, img_ann_dict, **kwargs):
    classes = coco_anns[f'{task}_categories']
    num_classes = len(classes)
    bin_labels = np.zeros((len(coco_anns["images"]), num_classes))
    bin_preds = np.zeros((len(coco_anns["images"]), num_classes))
    evaluated_frames = []
    for idx, img in tqdm(enumerate(coco_anns["images"]), total=len(coco_anns["images"])):
        binary_task_label = np.zeros(num_classes+1, dtype='uint8')
        label_list = [coco_anns["annotations"][idx][task] for idx in img_ann_dict[img['file_name']]]
        assert all(type(label_list[0])==type(lab_item) for lab_item in label_list), f'Inconsistent label type {label_list} in frame {img["file_name"]}'
        
        if isinstance(label_list[0], list):
            label_list = list(set(itertools.chain(*label_list)))
        elif isinstance(label_list[0], int):
            label_list = list(set(label_list))
        else:
            raise ValueError(f'Do not support annotation {label_list[0]} of type {type(label_list[0])} in frame {img["file_name"]}')
        binary_task_label[label_list] = 1
        ann_classes = binary_task_label[1:].tolist()
        bin_labels[idx, :] = ann_classes
        

        if  img["file_name"] in preds.keys():            
            these_probs = preds[img["file_name"]]['{}_score_dist'.format(task)]
            
            if len(these_probs) == 0:
                print("Prediction not found for image {}".format(img["file_name"]))
                these_probs = np.zeros((1, num_classes))
            else:
                evaluated_frames.append(idx)
            bin_preds[idx, :] = these_probs
            
        else:
            print("Image {} not found in predictions lists".format(img["file_name"]))
            
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
    
    cat_names = [f"{cat['name']}-AP" for cat in classes]
    cat_res_dict = dict(zip(cat_names,list(ap.values())))
            
    return mAP, cat_res_dict