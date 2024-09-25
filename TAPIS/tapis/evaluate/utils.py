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
    '''
    This function computes the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    bb1, bb2 : list or tuple of 4 elements
        Each contains the coordinates of a bounding box in the format [x_min, y_min, x_max, y_max].
    
    Returns:
    float
        The IoU of the two bounding boxes, a value between 0 and 1.
    '''
    
    # Determine the coordinates of the intersection rectangle
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    
    # If there is no intersection, return IoU as 0
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Special cases: vertical or horizontal line overlap
    elif y2 == y1 and bb1[1] == bb1[3] == bb2[1] == bb2[3]:
        return 1
    elif x2 == x1 and bb1[0] == bb1[2] == bb2[0] == bb2[2]:
        return 1
    
    # Compute the area of the intersection rectangle
    inter = (x2 - x1) * (y2 - y1)
    
    # Compute the area of both bounding boxes
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    
    # Check for potential division by zero
    if (area1 + area2 - inter) == 0:
        breakpoint()
    
    # Compute the IoU
    box_iou = inter / (area1 + area2 - inter)
    
    # Ensure the IoU is within the valid range
    assert box_iou >= 0 and box_iou <= 1
    
    return box_iou

def compute_mask_iou(m1,m2):
    '''
    This function computes the Intersection over Union (IoU) of two binary masks.
    
    Parameters:
    m1, m2 : numpy arrays
        Each contains a binary mask where pixels within the mask are 1 and outside are 0.
    
    Returns:
    float
        The IoU of the two masks, a value between 0 and 1.
    '''
    
    # Compute the intersection of the two masks
    intersection = np.sum(m1 * m2)
    
    # If there is no intersection, return IoU as 0
    if intersection == 0:
        return 0.0
    
    # Compute the union of the two masks
    union = np.sum(m1) + np.sum(m2) - intersection
    
    # Compute the IoU
    mask_iou = intersection / union
    
    # Ensure the IoU is within the valid range
    assert mask_iou >= 0 and mask_iou <= 1
    
    return mask_iou

def polygon_to_rle(polygons, width, height) -> dict:
    '''
    This function converts a list of polygons to a run-length encoding (RLE) format.
    
    Parameters:
    polygons : list of lists
        A list containing polygon coordinates, where each polygon is represented as a list of points.
    width : int
        The width of the image.
    height : int
        The height of the image.
    
    Returns:
    dict
        The RLE representation of the input polygons.
    '''
    
    # Flatten the polygon coordinates and convert them to a list
    polys = [np.array(p).flatten().tolist() for p in polygons]
    
    # Convert the polygon coordinates to RLE format using the frPyObjects function
    rles = m.frPyObjects(polys, height, width)
    
    # Merge the RLEs into a single RLE
    return m.merge(rles)

def rle_to_polygon(rle) -> dict:
    '''
    This function converts a run-length encoding (RLE) to polygon format.
    
    Parameters:
    rle : dict
        The RLE representation of a mask.
    
    Returns:
    dict
        The polygon representation of the mask.
    '''
    
    # Decode the RLE to a binary mask
    mask = decode_rle_to_mask(rle)
    
    # Convert the binary mask to polygon format
    return mask_to_polygon(mask)

def mask_to_polygon(mask):
    '''
    This function converts a binary mask to a list of polygons.
    
    Parameters:
    mask : numpy array
        A binary mask where the mask region is 1 and the background is 0.
    
    Returns:
    list of lists
        A list containing the polygon coordinates, where each polygon is represented as a list of points.
    '''
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize the list to store polygons
    polygons = []
    
    # Iterate through each contour found
    for contour in contours:
        # Flatten and convert the contour points to a list
        contour = contour.squeeze().ravel().tolist()
        
        # Ensure the contour has an even number of points and has at least 3 points (6 values)
        if len(contour) % 2 == 0 and len(contour) >= 6:
            # Append the valid contour to the polygons list
            polygons.append(contour)
    
    # Return the list of polygons
    return polygons

def mask_to_rle(mask):
    '''
    This function converts a binary mask to run-length encoding (RLE) format.
    
    Parameters:
    mask : numpy array
        A binary mask where the mask region is 1 and the background is 0.
    
    Returns:
    dict
        The RLE representation of the mask.
    '''
    
    # Convert the binary mask to a Fortran-contiguous array and encode it to RLE format
    return m.encode(np.asfortranarray(mask))
    
def decode_polygon_to_mask(polygons, width, height):
    '''
    This function converts polygons to a binary mask.
    
    Parameters:
    polygons : list of lists
        A list containing polygon coordinates, where each polygon is represented as a list of points.
    width : int
        The width of the image.
    height : int
        The height of the image.
    
    Returns:
    numpy array
        The binary mask where the mask region is 1 and the background is 0.
    '''
    
    # Convert the polygons to run-length encoding (RLE) format
    rle = polygon_to_rle(polygons, width, height)
    
    # Decode the RLE to a binary mask
    return decode_rle_to_mask(rle)

def decode_compressed_rle_to_mask(rle, dtype='uint8'):
    '''
    This function decodes a compressed run-length encoding (RLE) to a binary mask.
    
    Parameters:
    rle : dict
        The compressed RLE representation of a mask.
    dtype : str, optional
        The desired data type of the output mask. Default is 'uint8'.
    
    Returns:
    numpy array
        The binary mask where the mask region is 1 and the background is 0, with the specified data type.
    '''
    
    # Create a copy of the RLE dictionary to avoid modifying the original
    compressed_rle_dict = rle.copy()
    
    # Decode the base64-encoded RLE counts string
    uncodedStr = base64.b64decode(rle['counts'])
    
    # Decompress the RLE counts string
    uncompressedStr = zlib.decompress(uncodedStr, wbits=zlib.MAX_WBITS)
    
    # Update the counts in the copied RLE dictionary with the decompressed string
    compressed_rle_dict['counts'] = uncompressedStr
    
    # Decode the RLE to a binary mask and convert it to the specified data type
    mask = m.decode(compressed_rle_dict).astype(dtype)
    
    # Return the binary mask
    return mask


def decode_rle_to_mask(rle, dtype='uint8'):
    '''
    This function decodes a run-length encoding (RLE) to a binary mask.
    
    Parameters:
    rle : dict
        The RLE representation of a mask.
    
    Returns:
    numpy array
        The binary mask where the mask region is 1 and the background is 0.
    '''
    
    # Decode the RLE to a binary mask and convert it to unsigned 8-bit integer type
    mask = m.decode(rle).astype('uint8')
    
    # Return the binary mask
    return mask

def mask_to_bbox(mask, full_coordinates=False):
    '''
    This function converts a binary mask to a bounding box.
    
    Parameters:
    mask : numpy array
        A binary mask where the mask region is 1 and the background is 0.
    full_coordinates : bool, optional
        If True, returns the bounding box as [x1, y1, x2, y2].
        If False, returns the bounding box as [x1, y1, width, height].
    
    Returns:
    list
        The bounding box coordinates. The format depends on the value of full_coordinates.
    '''
    
    # Find the y and x coordinates where the mask is greater than 0
    ys, xs = np.where(mask > 0)
    
    # Determine the minimum and maximum x and y coordinates of the mask
    x1 = np.min(xs)
    x2 = np.max(xs)
    y1 = np.min(ys)
    y2 = np.max(ys)
    
    # Return the bounding box coordinates
    if full_coordinates:
        # Return as [x1, y1, x2, y2]
        return [x1, y1, x2, y2]
    
    # Return as [x1, y1, width, height]
    return [x1, y1, x2 - x1, y2 - y1]

def xywhbbox_to_dxdydxdybbox(bbox, width, height):
    '''
    Convert a bounding box from (x, y, width, height) format (xywh) 
    to (x_min, y_min, x_max, y_max) format, then normalize it by the given image dimensions.

    `bbox`: The bounding box in (x, y, width, height) format.
    `width` and `height`: The width and height of the image to normalize the coordinates.
    '''
    
    # Create a new bounding box in (x_min, y_min, x_max, y_max) format.
    # bbox[0] is x (left), bbox[1] is y (top), bbox[2] is width, bbox[3] is height.
    d_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
    
    # Normalize the bounding box coordinates by dividing by image width and height.
    d_bbox[0] /= width  # Normalize x_min (left) by image width.
    d_bbox[1] /= height # Normalize y_min (top) by image height.
    d_bbox[2] /= width  # Normalize x_max (right) by image width.
    d_bbox[3] /= height # Normalize y_max (bottom) by image height.
    
    # Return the normalized (x_min, y_min, x_max, y_max) bounding box.
    return d_bbox

def xywh_to_x1y1x2y2(bbox):
    '''
    This function converts a bounding box from [x, y, width, height] format to [x1, y1, x2, y2] format.
    
    Parameters:
    bbox : list or tuple of 4 elements
        The bounding box in [x, y, width, height] format.
    
    Returns:
    list
        The bounding box in [x1, y1, x2, y2] format, with coordinates rounded to the nearest integer.
    '''
    
    # Round the coordinates and dimensions of the bounding box
    bbox = list(map(round, bbox))
    
    # Convert to [x1, y1, x2, y2] format
    xy_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
    
    # Return the converted bounding box
    return xy_bbox

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