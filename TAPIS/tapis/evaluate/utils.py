import base64
import itertools
import json
import zlib
import cv2
import numpy as np
import torch
from tqdm import tqdm
import pycocotools.mask as m

def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def save_json(data, json_file, indent=4):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent)

def gather_info(coco_json):
    return_dict = {}
    id2name = {}
    for image in coco_json['images']:
        id2name[image['id']]={'file_name': image['file_name'], 'width': image['width'], 'height': image['height']}
        return_dict[image['file_name']]={'instances': []}
    
    return return_dict, id2name

def box_iou(bb1,bb2):
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

def decode_compressed_rle_to_mask(rle, dtype='uint8'):
    compressed_rle_dict = rle.copy()
    uncodedStr = base64.b64decode(rle['counts'])
    uncompressedStr = zlib.decompress(uncodedStr,wbits = zlib.MAX_WBITS)  
    compressed_rle_dict['counts'] = uncompressedStr 
    mask = m.decode(compressed_rle_dict).astype(dtype)
    return mask

def decode_rle_to_mask(rle, dtype='uint8'):
    mask = m.decode(rle).astype(dtype)
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

def xywhbbox_to_dxdydxdybbox(bbox, width, height):
    d_bbox = [bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]]
    d_bbox[0] /= width
    d_bbox[1] /= height
    d_bbox[2] /= width
    d_bbox[3] /= height
    return d_bbox

def roundbox(box):
    keys = box.split(' ')
    keys = [str(round(float(k),4)) for k in keys]
    return ' '.join(keys)

def floatbox(box):
    keys = box.split(' ')
    return list(map(float,keys))

def getrealbox(boxes,box):
    maxiou = 0
    realbox = ''
    box = floatbox(box)
    for box2 in boxes:
        iou = box_iou(floatbox(box2),box)
        if iou>maxiou:
            maxiou=iou
            realbox=box2
    
    assert maxiou>0.9 and maxiou<=1
    return realbox

def filter_preds(preds_list, method, parameters):
    if method == 'all':
        return [p for p in preds_list if p['score']>0]
    
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
    
def format_instances(instances, width, height, segmentation=False):
    formated_instances = []
    for instance in instances:  
        bbox = instance['bbox']
        x1, y1, w, h = bbox
        x2, y2 = x1+w, y1+h
        x1 /= width
        x2 /= width
        y1 /= height
        y2 /= height

        pred_tools = instance['score_dist']
        this_instance = {'bbox': [x1,y1,x2,y2], 'instruments_score_dist': pred_tools}
        if segmentation:
            segment = instance['segmentation']
            this_instance['segment'] = segment
        
        formated_instances.append(this_instance)
            
    return formated_instances
    
def read_detectron2_output(coco_anns_path, preds_path, selection, selection_params, segmentation=False):
    data_dict = load_json(coco_anns_path)
    data_dict, id2name = gather_info(data_dict)

    if 'pth' in preds_path:
        preds = torch.load(preds_path)
        for pred in tqdm(preds, desc='Filtering preds'):
            instances = filter_preds(pred['instances'], selection, selection_params)
            instances = format_instances(instances=instances, 
                                         width=id2name[pred['image_id']]['width'], 
                                         height=id2name[pred['image_id']]['height'], 
                                         segmentation=segmentation)
            data_dict[id2name[pred['image_id']]['file_name']]['instances'] = instances
        
    elif 'json' in preds_path:
        preds = load_json(preds_path)
        
        for pred in tqdm(preds, desc='Separating preds'):
            if pred['category_id']>0 and pred['score']>0.0:
                pred['category_id'] -= 1
                data_dict[id2name[pred['image_id']]['file_name']]['instances'].append(pred)
        
        for name in tqdm(data_dict, desc='Filtering preds'):
            if len(data_dict[name]['instances']):
                image_id = data_dict[name]['instances'][0]['image_id']
                instances = filter_preds(data_dict[name]['instances'], selection, selection_params)
                instances = format_instances(instances=instances, 
                                            width=id2name[image_id]['width'], 
                                            height=id2name[image_id]['height'], 
                                            segmentation=segmentation)
                data_dict[name]['instances'] = instances
            else:
                data_dict[name]['instances'] = []

    return data_dict