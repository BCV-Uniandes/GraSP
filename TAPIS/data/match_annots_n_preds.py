import argparse
from copy import deepcopy
import itertools
import json
import traceback
import numpy as np
import torch
import os
import pycocotools.mask as m
import cv2
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import warnings

def gather_info(coco_json):
    return_dict = {}
    name2id = {}
    for image in coco_json['images']:
        name2id[image["file_name"]] = image["id"]
        if 'video_name' not in image or 'frame_num' not in image:
            case,frame = image["file_name"].split('/')
            image['video_name'] = case
            image['frame_num'] = int(frame.split('.')[0])
        image['annotations'] = []
        image['predictions'] = []
        return_dict[image['id']] = image

    for annot in coco_json['annotations']:
        annot['instruments'] = annot['category_id']
        if 'phases' in annot:
            del annot['phases']
        if 'steps' in annot:
            del annot['steps']
        
        return_dict[annot['image_id']]['annotations'].append(annot)
    
    return return_dict, name2id

def compute_bbox_iou(bb1,bb2):
    x1 = max(bb1[0],bb2[0])
    y1 = max(bb1[1],bb2[1])
    x2 = min(bb1[2],bb2[2])
    y2 = min(bb1[3],bb2[3])
    if x2<x1 or y2<y1:
        return 0.0
    elif y2==y1 and bb1[1]==bb1[3]==bb2[1]==bb2[3]:
        return 1
    elif x2==x1 and bb1[0]==bb1[2]==bb2[0]==bb2[2]:
        return 1
    inter = (x2-x1)*(y2-y1)
    area1 = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    if (area1+area2-inter)==0:
        breakpoint()
    box_iou = inter/(area1+area2-inter)

    assert box_iou>=0 and box_iou<=1
    return box_iou

def compute_mask_iou(m1,m2):
    intersection = np.sum(m1*m2)
    if intersection==0:
        return 0.0
    union = np.sum(m1) + np.sum(m2) - intersection
    mask_iou = intersection/union
    assert mask_iou>=0 and mask_iou<=1
    return mask_iou

def polygon_to_rle(polygons, width, height) -> dict:
    polys = [np.array(p).flatten().tolist() for p in polygons]
    rles = m.frPyObjects(polys, height, width)
    return m.merge(rles)

def rle_to_polygon(rle) -> dict:
    mask = decode_rle_to_mask(rle)
    return mask_to_polygon(mask)

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.squeeze().ravel().tolist()
        if len(contour)%2 == 0 and len(contour)>=6:
            polygons.append(contour)
    return polygons

def mask_to_rle(mask):
    return m.encode(np.asfortranarray(mask))
    
def decode_polygon_to_mask(polygons, width, height):
    rle = polygon_to_rle(polygons, width, height)
    return decode_rle_to_mask(rle)

def decode_rle_to_mask(rle):
    mask = m.decode(rle).astype('uint8')
    return mask

def mask_to_bbox(mask, full_coordinates=False):
    ys,xs = np.where(mask>0)
    x1 = np.min(xs)
    x2 = np.max(xs)
    y1 = np.min(ys)
    y2 = np.max(ys)
    
    if full_coordinates:
        return [x1, y1, x2, y2]

    return [x1, y1, x2-x1, y2-y1]

def xywh_to_x1y1x2y2(bbox):
    bbox = list(map(round,bbox))
    xy_bbox = [bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]]
    return xy_bbox

def filter_preds(preds_list, method, parameters):
    preds_list = remove_duplicates_n_features(preds_list)
    if method == 'all':
        return preds_list
    
    if 'topk' in method:
        preds_list.sort(key=lambda x: x['score'], reverse=True)
    
    if 'cls' in method:
        cls_preds = {i:[] for i in range(1,len(parameters)+1)}
        use_thresh = False
        use_topk = False
        
        if 'thresh' in method:
            thresh = [parameters[i]['threshold'] for i in range(1,len(parameters)+1)]
            use_thresh = True
        
        if 'topk' in method:
            topk = [parameters[i]['top_k'] for i in range(1,len(parameters)+1)]
            use_topk = True
            
        for pred in preds_list:
            if ((not use_thresh) or (use_thresh and pred['score']>=thresh[pred['category_id']])) and \
               ((not use_topk)   or (use_topk   and len(cls_preds[pred['category_id']+1]) < topk[pred['category_id']])):
                cls_preds[pred['category_id']+1].append(pred)
        
        return list(itertools.chain(*list(cls_preds.values())))
    else:
        if 'topk' in method:
            preds_list = preds_list[:parameters[0]]
        if 'thresh' in method:
            preds_list = [p for p in preds_list if p['score']>=parameters[1]]
        return preds_list

def remove_duplicates_n_features(instances):
    no_dups = {}
    for inst in instances:
        if inst['score']>0:
            bbox_key = tuple(inst['bbox'])
            if bbox_key not in no_dups:
                no_dups[bbox_key] = inst
            else:
                if inst['score']>no_dups[bbox_key]['score']:
                    no_dups[bbox_key] = inst
    return list(no_dups.values())

parser = argparse.ArgumentParser(description='Arguments for predictions matching.')

parser.add_argument('--coco_anns_path', type=str, help='Path to COCO JSON with annotations')
parser.add_argument('--preds_path', type=str, help='Path to file with predictions')
parser.add_argument('--out_coco_anns_path', type=str, help='Path to save JSON file with annotations')
parser.add_argument('--out_coco_preds_path', type=str, help='Path to save JSON file with predictions')
parser.add_argument('--out_features_path', type=str, help='Path to save tprch file with prediction features')
parser.add_argument('--features_key', nargs='+', type=str, default='features', help='Dict key that contains the features')

parser.add_argument('--selection', type=str, default='thresh', 
                    choices=['thresh', 'topk', 'topk_thresh', 'cls_thresh', 'cls_topk', 'cls_topk_thresh', 'all'], 
                    help='Prediction selection method')
parser.add_argument('--selection_info', help='Hypermarameters to perform selection')

parser.add_argument('--segmentation', default=False, action='store_true', help='Use segmentation annotations and predictions')
parser.add_argument('--validation', default=False, action='store_true', help='Use validation annotations and predictions')

args = parser.parse_args()

# breakpoint()
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
elif args.selection=='all':
    selection_params = [None, None]
else:
    raise ValueError(f'Incorrect selection type {args.selection}')

with open(args.coco_anns_path, 'r') as f:
    coco_anns = json.load(f)
    data_dict, id2name = gather_info(coco_anns)
    coco_anns = {k:v if k not in ['images', 'annotations'] else []  for k,v in coco_anns.items()}

if 'pth' in args.preds_path:
    preds = torch.load(args.preds_path)
    if len(data_dict)!=len(preds):
        warnings.warn(f"Annotations have a different length than predictions {len(data_dict)}!={len(preds)}. This is probably an error.")
    for pred in tqdm(preds, desc='Filtering preds'):
        instances = filter_preds(pred['instances'], args.selection, selection_params)
        data_dict[pred['image_id']]['predictions'].extend(instances)
    
elif 'json' in args.preds_path:
    with open(args.preds_path, 'r') as f:
        preds = json.load(f)
    
    ids = set()
    for pred in tqdm(preds, desc='Separating preds'):
        ids.add(pred['image_id'])
        if pred['category_id']>0 and pred['score']>0.0:
            pred['category_id'] -= 1
            data_dict[pred['image_id']]['predictions'].append(pred)
    if len(data_dict)!=len(ids):
        warnings.warn(f"Annotations have a different length than predictions {len(data_dict)}!={len(ids)}. This is probably an error.")
    
    for idx in tqdm(data_dict, desc='Filtering preds'):
        instances = filter_preds(data_dict[idx]['predictions'], args.selection, selection_params)
        data_dict[idx]['predictions'] = instances

save_ann_dict = deepcopy(coco_anns)
save_ann_dict['instruments_categories'] = save_ann_dict['categories']
del save_ann_dict['categories']

save_pred_dict = deepcopy(coco_anns)
save_pred_dict['instruments_categories'] = save_pred_dict['categories']
del save_pred_dict['categories']

save_feats = []

pred_anns = 0
for idx in tqdm(data_dict, desc='Matching annotations'):

    annotations = data_dict[idx]['annotations']
    num_annots = len(annotations)
    predictions = data_dict[idx]['predictions']
    num_preds = len(predictions)
    width = data_dict[idx]['width']
    height = data_dict[idx]['height']

    feat_save = {'image_id': idx, 'file_name': data_dict[idx]['file_name'], 'features': {}, 'width': width, 'height': height}

    this_im = {k:v for k,v in data_dict[idx].items() if k not in ['annotations','predictions']}
    save_ann_dict['images'].append(this_im)
    save_ann_dict['annotations'].extend(annotations)
    save_pred_dict['images'].append(this_im)
    
    if num_annots>0 and num_preds>0:
        ious = np.zeros((num_annots, num_preds))

        for ann_id, ann in enumerate(annotations):
            for pred_id, pred in enumerate(predictions):
                
                ann_box = ann['bbox']
                ann_box = [ann_box[0], ann_box[1], ann_box[2]+ann_box[0], ann_box[3]+ann_box[1]]
                if args.segmentation and pred['bbox']==[0,0,0,0]:
                    breakpoint()

                pred_box = pred['bbox']
                pred_box = [pred_box[0], pred_box[1], pred_box[2]+pred_box[0], pred_box[3]+pred_box[1]]

                bbox_iou = compute_bbox_iou(ann_box, pred_box)

                if args.segmentation and bbox_iou>0:
                    if type(ann['segmentation']) == list:
                        ann_mask = decode_polygon_to_mask(ann['segmentation'], width, height)
                    elif type(ann['segmentation']) == dict:
                        ann_mask = decode_rle_to_mask(ann['segmentation'])
                    
                    pred_mask = decode_rle_to_mask(pred['segmentation'])
                    mask_iou = compute_mask_iou(ann_mask, pred_mask)
                    ious[ann_id, pred_id] = mask_iou
                else:
                    ious[ann_id, pred_id] = bbox_iou

        if args.validation:
            indices = np.argmax(ious, axis=0)

            for pred_id, ann_id in enumerate(indices):
                pred_anns+=1
                this_ann = deepcopy(annotations[ann_id])
                this_pred = predictions[pred_id]  
                this_ann['id'] = pred_anns
                this_ann['bbox'] = list(map(round,this_pred['bbox']))
                this_ann['score'] = this_pred['score']
                box_key = tuple(xywh_to_x1y1x2y2(this_pred['bbox']))
                feat_save['features'][box_key] = list(itertools.chain(*[this_pred[f_key] for f_key in args.features_key])) #Concatenate features
                if args.segmentation:
                    this_ann['segmentation'] = this_pred['segmentation']
                save_pred_dict['annotations'].append(this_ann)

        else:
            row_indices, col_indices = linear_sum_assignment(-ious)
            matches = list(zip(row_indices, col_indices)) 

            for ann_id, pred_id in matches:
                pred_anns += 1 
                this_ann = deepcopy(annotations[ann_id])
                this_pred = predictions[pred_id]  
                this_ann['id'] = pred_anns
                this_ann['bbox'] = list(map(round,this_pred['bbox']))
                this_ann['score'] = this_pred['score']
                box_key = tuple(xywh_to_x1y1x2y2(this_pred['bbox']))
                feat_save['features'][box_key] = list(itertools.chain(*[this_pred[f_key] for f_key in args.features_key]))
                if args.segmentation:
                    this_ann['segmentation'] = this_pred['segmentation']
                save_pred_dict['annotations'].append(this_ann)
    
    elif num_preds>0 and args.validation:
        for pred_id,pred in enumerate(predictions):
            pred_anns += 1
            this_pred = {'id': pred_anns,
                         'image_id': idx,
                         'image_name': data_dict[idx]['file_name'],
                         'category_id': pred['category_id']+1,
                         'instruments': pred['category_id']+1,
                         'actions': [1],
                         'bbox': list(map(round,pred['bbox'])),
                         'iscrowd': 0,
                         'area': pred['bbox'][2]*pred['bbox'][3],
                         'score': pred['score']
                         }
            box_key = tuple(xywh_to_x1y1x2y2(pred['bbox']))
            feat_save['features'][box_key] = list(itertools.chain(*[pred[f_key] for f_key in args.features_key]))

            if args.segmentation:
                pred['segmentation'] = pred['segmentation']
            save_pred_dict['annotations'].append(this_pred)  

    elif num_preds==0:
        pred_anns += 1
        this_pred = {'id': pred_anns,
                         'image_id': idx,
                         'image_name': data_dict[idx]['file_name'],
                         'category_id': -1,
                         'instruments': -1,
                         'actions': [-1],
                         'bbox': [0,0,0,0],
                         'iscrowd': 0,
                         'area': 0,
                         'score': 1
                         }  
        if args.segmentation:
            this_pred['segmentation'] = {}
        save_pred_dict['annotations'].append(this_pred)   

    save_feats.append(feat_save)

os.makedirs(os.path.join('/',*args.out_coco_anns_path.split('/')[1:-1]),exist_ok=True)
with open(args.out_coco_anns_path, 'w') as f:
    json.dump(save_ann_dict, f, indent=4)   

os.makedirs(os.path.join('/',*args.out_coco_preds_path.split('/')[1:-1]),exist_ok=True)
with open(args.out_coco_preds_path, 'w') as b:
    json.dump(save_pred_dict, b, indent=4)  

os.makedirs(os.path.join('/',*args.out_features_path.split('/')[1:-1]),exist_ok=True)
torch.save(save_feats, args.out_features_path)  