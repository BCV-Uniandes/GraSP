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
    '''
    This function processes a COCO-formatted JSON object containing image and annotation data.
    It returns a dictionary mapping image IDs to image data and a dictionary mapping image file names to image IDs.
    '''
    
    # Initialize the return dictionaries
    return_dict = {}
    name2id = {}
    
    # Iterate through each image in the COCO JSON
    for image in coco_json['images']:
        # Map the image file name to its ID
        name2id[image["file_name"]] = image["id"]
        
        # Add 'video_name' and 'frame_num' fields if they are not present
        # This is specific for GraSP dataset, modify for any other dataset
        if 'video_name' not in image or 'frame_num' not in image:
            case, frame = image["file_name"].split('/')
            image['video_name'] = case
            image['frame_num'] = int(frame.split('.')[0])
        
        # Initialize 'annotations' and 'predictions' lists for the image
        image['annotations'] = []
        image['predictions'] = []
        
        # Add the processed image data to the return dictionary
        return_dict[image['id']] = image

    # Iterate through each annotation in the COCO JSON
    for annot in coco_json['annotations']:
        # Map 'category_id' to 'instruments' in the annotation
        annot['instruments'] = annot['category_id']
        
        # Remove 'phases' and 'steps' fields if they exist
        if 'phases' in annot:
            del annot['phases']
        if 'steps' in annot:
            del annot['steps']
        
        # Append the annotation to the corresponding image's 'annotations' list
        return_dict[annot['image_id']]['annotations'].append(annot)
    
    # Return the processed dictionaries
    return return_dict, name2id


def compute_bbox_iou(bb1, bb2) -> float:
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


def compute_mask_iou(m1, m2) -> float:
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


def decode_rle_to_mask(rle):
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

def remove_duplicates_n_features(instances):
    '''
    This function removes duplicate instances based on their bounding boxes, keeping the instance with the highest score.
    
    Parameters:
    instances : list of dicts
        Each dictionary represents an instance with keys including 'bbox' (bounding box) and 'score'.
    
    Returns:
    list of dicts
        A list of unique instances, where each bounding box appears only once, keeping the instance with the highest score.
    '''
    
    # Initialize a dictionary to store unique instances keyed by their bounding boxes
    no_dups = {}
    
    # Iterate through each instance
    for inst in instances:
        # Only consider instances with a positive score
        if inst['score'] > 0:
            # Convert the bounding box to a tuple to use as a dictionary key
            bbox_key = tuple(inst['bbox'])
            
            # If the bounding box is not already in the dictionary, add the instance
            if bbox_key not in no_dups:
                no_dups[bbox_key] = inst
            else:
                # If the bounding box is already in the dictionary, keep the instance with the higher score
                if inst['score'] > no_dups[bbox_key]['score']:
                    no_dups[bbox_key] = inst
    
    # Return the list of unique instances
    return list(no_dups.values())

def filter_preds(preds_list, method, parameters):
    '''
    This function filters a list of predictions based on the specified method and parameters.
    
    Parameters:
    preds_list : list of dicts
        A list of predictions, where each prediction is represented as a dictionary with keys including 'score' and 'category_id'.
    method : str
        The method to use for filtering. Possible values include 'all', 'topk', 'cls', 'thresh'.
    parameters : list or dict
        Parameters used for filtering. The exact content depends on the method.
    
    Returns:
    list of dicts
        The filtered list of predictions.
    '''
    
    # Remove duplicate predictions based on their bounding boxes and keep the ones with the highest score
    preds_list = remove_duplicates_n_features(preds_list)
    
    # If the method is 'all', return the predictions list as is
    if method == 'all':
        return preds_list
    
    # If 'topk' is in the method, sort the predictions by score in descending order
    if 'topk' in method:
        preds_list.sort(key=lambda x: x['score'], reverse=True)
    
    # If 'cls' is in the method, apply class-specific filtering
    if 'cls' in method:
        cls_preds = {i: [] for i in range(1, len(parameters) + 1)}
        use_thresh = False
        use_topk = False
        
        # Check if 'thresh' is in the method and set thresholds for each class
        if 'thresh' in method:
            thresh = [parameters[i]['threshold'] for i in range(1, len(parameters) + 1)]
            use_thresh = True
        
        # Check if 'topk' is in the method and set top k for each class
        if 'topk' in method:
            topk = [parameters[i]['top_k'] for i in range(1, len(parameters) + 1)]
            use_topk = True
            
        # Iterate through the predictions and filter based on per-class thresholds and per-class top k
        for pred in preds_list:
            if ((not use_thresh) or (use_thresh and pred['score'] >= thresh[pred['category_id']])) and \
               ((not use_topk)   or (use_topk and len(cls_preds[pred['category_id'] + 1]) < topk[pred['category_id']])):
                cls_preds[pred['category_id'] + 1].append(pred)
        
        # Return the concatenated list of class-specific predictions
        return list(itertools.chain(*list(cls_preds.values())))
    else:
        # Apply general top k filtering if specified
        if 'topk' in method:
            preds_list = preds_list[:parameters[0]]
        
        # Apply general threshold filtering ifspecified
        if 'thresh' in method:
            preds_list = [p for p in preds_list if p['score'] >= parameters[1]]
        
        # Return the filtered predictions list
        return preds_list


parser = argparse.ArgumentParser(description='Arguments for predictions matching.')

parser.add_argument('--coco_anns_path', type=str, help='Path to COCO JSON with annotations')
parser.add_argument('--preds_path', type=str, help='Path to file (JSON or pytorch) with predictions')
parser.add_argument('--out_coco_anns_path', type=str, help='Path to save JSON file with annotations')
parser.add_argument('--out_coco_preds_path', type=str, help='Path to save JSON file with predictions')
parser.add_argument('--out_features_path', type=str, help='Path to save pytorch file with prediction features')
parser.add_argument('--features_key', nargs='+', type=str, default='features', help='Dict key that contains the features')

parser.add_argument('--selection', type=str, default='thresh', 
                    choices=['thresh', # General threshold filtering
                             'topk', # General top k filtering
                             'topk_thresh', # Threshold and top k filtering
                             'cls_thresh', # Per-class threshold filtering
                             'cls_topk', # Per-class top k filtering
                             'cls_topk_thresh', # Per-class top k and and threshold filtering
                             'all' # No filtering
                             ], 
                    help='Prediction selection method')
parser.add_argument('--selection_info', help='Hypermarameters to perform filtering')

parser.add_argument('--segmentation', default=False, action='store_true', help='Use segmentation annotations and predictions')

# Use these parameter for validation predictions 
# This argument keeps all predicted masks (after filtering) and only uses ground truth to assign the expected class of each instance for metric calculation
parser.add_argument('--validation', default=False, action='store_true', help='Don use the ground truth annotations as reference for matching')

args = parser.parse_args()

# Determine selection parameters based on the selection method specified in the arguments
if args.selection == 'thresh':
    # For threshold-based selection, set the parameters to [None, threshold_value]
    selection_params = [None, float(args.selection_info)]
elif args.selection == 'topk':
    # For top-k selection, set the parameters to [k_value, None]
    selection_params = [int(args.selection_info), None]
elif args.selection == 'topk_thresh':
    # For combined top-k and threshold selection, ensure the selection_info is a string containing two values separated by a comma
    assert type(args.selection_info) == str and ',' in args.selection_info and len(args.selection_info.split(',')) == 2
    
    # Split the selection_info string into the two parameters, converting them to int and float respectively
    selection_params = args.selection_info.split(',')
    selection_params[0] = int(selection_params[0])
    selection_params[1] = float(selection_params[1])
elif 'cls' in args.selection:
    # For class-specific selection, ensure selection_info is a valid file path
    assert type(args.selection_info) == str
    assert os.path.isfile(args.selection_info)
    
    # Load the class-specific selection parameters from the JSON file
    with open(args.selection_info, 'r') as f:
        selection_params = json.load(f)
elif args.selection == 'all':
    # For selecting all predictions, set the parameters to [None, None]
    selection_params = [None, None]
else:
    # Raise an error if the selection type is not recognized
    raise ValueError(f'Incorrect selection type {args.selection}')

# Load the COCO annotations from the specified path
with open(args.coco_anns_path, 'r') as f:
    coco_anns = json.load(f)
    # Process the annotations and gather information into data_dict and id2name
    data_dict, id2name = gather_info(coco_anns)
    # Clear the 'images' and 'annotations' fields in the COCO annotations dictionary
    coco_anns = {k: v if k not in ['images', 'annotations'] else [] for k, v in coco_anns.items()}

# Process predictions based on the file type (pth or json)
if 'pth' in args.preds_path:
    # Load the predictions from a .pth file
    preds = torch.load(args.preds_path)
    # Check for length mismatch between annotations and predictions
    if len(data_dict) != len(preds):
        warnings.warn(f"Annotations have a different length than predictions {len(data_dict)}!={len(preds)}. This is probably an error.")
    
    # Filter predictions for each image and update the data_dict
    for pred in tqdm(preds, desc='Filtering preds'):
        instances = filter_preds(pred['instances'], args.selection, selection_params)
        data_dict[pred['image_id']]['predictions'].extend(instances)
    
elif 'json' in args.preds_path:
    # Load the predictions from a .json file
    with open(args.preds_path, 'r') as f:
        preds = json.load(f)
    
    ids = set()
    # Process each prediction, adjusting the category_id and updating the data_dict
    for pred in tqdm(preds, desc='Separating preds'):
        ids.add(pred['image_id'])
        if pred['category_id'] > 0 and pred['score'] > 0.0:
            pred['category_id'] -= 1
            data_dict[pred['image_id']]['predictions'].append(pred)
    
    # Check for length mismatch between annotations and predictions
    if len(data_dict) != len(ids):
        warnings.warn(f"Annotations have a different length than predictions {len(data_dict)}!={len(ids)}. This is probably an error.")
    
    # Filter predictions for each image and update the data_dict
    for idx in tqdm(data_dict, desc='Filtering preds'):
        instances = filter_preds(data_dict[idx]['predictions'], args.selection, selection_params)
        data_dict[idx]['predictions'] = instances

# Prepare the final annotation and prediction dictionaries for saving
save_ann_dict = deepcopy(coco_anns)
save_ann_dict['instruments_categories'] = save_ann_dict['categories']
del save_ann_dict['categories']

save_pred_dict = deepcopy(coco_anns)
save_pred_dict['instruments_categories'] = save_pred_dict['categories']
del save_pred_dict['categories']

# Initialize an empty list to save features
save_feats = []

# Initialize a counter for predicted annotations
pred_anns = 0

# Iterate through each image in the data dictionary
for idx in tqdm(data_dict, desc='Matching annotations'):

    # Extract annotations, predictions, and image dimensions for the current image
    annotations = data_dict[idx]['annotations']
    num_annots = len(annotations)
    predictions = data_dict[idx]['predictions']
    num_preds = len(predictions)
    width = data_dict[idx]['width']
    height = data_dict[idx]['height']

    # Prepare a dictionary to save features for the current image
    feat_save = {
        'image_id': idx, 
        'file_name': data_dict[idx]['file_name'], 
        'features': {}, 
        'width': width, 
        'height': height
    }

    # Copy basic image information to the annotations and predictions dictionaries
    this_im = {k: v for k, v in data_dict[idx].items() if k not in ['annotations', 'predictions']}
    save_ann_dict['images'].append(this_im)
    save_ann_dict['annotations'].extend(annotations)
    save_pred_dict['images'].append(this_im)
    
    if num_annots > 0 and num_preds > 0:
        # Initialize an IoU matrix for annotations and predictions
        ious = np.zeros((num_annots, num_preds))

        for ann_id, ann in enumerate(annotations):
            for pred_id, pred in enumerate(predictions):
                # Convert bounding boxes to [x1, y1, x2, y2] format
                ann_box = ann['bbox']
                ann_box = [ann_box[0], ann_box[1], ann_box[2] + ann_box[0], ann_box[3] + ann_box[1]]
                
                if args.segmentation and pred['bbox'] == [0, 0, 0, 0]:
                    breakpoint()

                pred_box = pred['bbox']
                pred_box = [pred_box[0], pred_box[1], pred_box[2] + pred_box[0], pred_box[3] + pred_box[1]]

                # Compute the IoU for this annotation and prediction bounding boxes
                bbox_iou = compute_bbox_iou(ann_box, pred_box)

                # If segment masks present and and bounding boxes IoU is not 0
                if args.segmentation and bbox_iou > 0:
                    # Decode the masks and compute the mask IoU
                    if type(ann['segmentation']) == list: # Is polygon mask
                        ann_mask = decode_polygon_to_mask(ann['segmentation'], width, height)
                    elif type(ann['segmentation']) == dict:# Is RLE mask
                        ann_mask = decode_rle_to_mask(ann['segmentation'])
                    
                    # Pred masks are always RLEs
                    pred_mask = decode_rle_to_mask(pred['segmentation'])
                    
                    # Compute mask IoU
                    mask_iou = compute_mask_iou(ann_mask, pred_mask)
                    ious[ann_id, pred_id] = mask_iou
                else:
                    ious[ann_id, pred_id] = bbox_iou

        # Keep all predicted masks (after filtering) and only use ground truths to assign espected validation class
        if args.validation:
            
            # Assign a highest annotation IoU for each prediction (to assign the expected class of each prediction)
            indices = np.argmax(ious, axis=0)

            # Iterate through ALL prediction
            for pred_id, ann_id in enumerate(indices):
                pred_anns += 1
                
                # Get the annotation instance with highest IoU
                this_ann = deepcopy(annotations[ann_id])
                # Current prediction
                this_pred = predictions[pred_id]
                
                # Assign prediction ID according to number of counted predictions
                this_ann['id'] = pred_anns
                
                # Keep the predicted bounding box instead of the annotated bounding box
                this_ann['bbox'] = list(map(round, this_pred['bbox']))
                
                # Include the predicted score 
                this_ann['score'] = this_pred['score']
                
                # Save prediction
                box_key = tuple(xywh_to_x1y1x2y2(this_pred['bbox']))
                feat_save['features'][box_key] = list(itertools.chain(*[this_pred[f_key] for f_key in args.features_key]))  # Concatenate features
                
                # Keep predicted segment instead of ground truth segment
                if args.segmentation:
                    this_ann['segmentation'] = this_pred['segmentation']
                save_pred_dict['annotations'].append(this_ann)

        
        else:
            # Use the Hungarian algorithm to match annotations and predictions and assign an independent prediction to each annotation (and discard unassigned predictions)
            # This is necessarry to obtain a region feature for each annotated instance
            row_indices, col_indices = linear_sum_assignment(-ious)
            matches = list(zip(row_indices, col_indices)) 

            for ann_id, pred_id in matches:
                pred_anns += 1
                this_ann = deepcopy(annotations[ann_id])
                this_pred = predictions[pred_id]
                this_ann['id'] = pred_anns
                this_ann['bbox'] = list(map(round, this_pred['bbox']))
                this_ann['score'] = this_pred['score']
                box_key = tuple(xywh_to_x1y1x2y2(this_pred['bbox']))
                feat_save['features'][box_key] = list(itertools.chain(*[this_pred[f_key] for f_key in args.features_key]))
                if args.segmentation:
                    this_ann['segmentation'] = this_pred['segmentation']
                save_pred_dict['annotations'].append(this_ann)
    
    elif num_preds > 0 and args.validation:
        # Handle cases with no annotations and only predictions during validation
        for pred_id, pred in enumerate(predictions):
            pred_anns += 1
            this_pred = {
                'id': pred_anns,
                'image_id': idx,
                'image_name': data_dict[idx]['file_name'],
                'category_id': pred['category_id'] + 1,
                'instruments': pred['category_id'] + 1,
                'actions': [1],
                'bbox': list(map(round, pred['bbox'])),
                'iscrowd': 0,
                'area': pred['bbox'][2] * pred['bbox'][3],
                'score': pred['score']
            }
            box_key = tuple(xywh_to_x1y1x2y2(pred['bbox']))
            feat_save['features'][box_key] = list(itertools.chain(*[pred[f_key] for f_key in args.features_key]))

            if args.segmentation:
                pred['segmentation'] = pred['segmentation']
            save_pred_dict['annotations'].append(this_pred)

    elif num_preds == 0:
        # Handle cases with no predictions
        pred_anns += 1
        this_pred = {
            'id': pred_anns,
            'image_id': idx,
            'image_name': data_dict[idx]['file_name'],
            'category_id': -1,
            'instruments': -1,
            'actions': [-1],
            'bbox': [0, 0, 0, 0],
            'iscrowd': 0,
            'area': 0,
            'score': 1
        }
        if args.segmentation:
            this_pred['segmentation'] = {}
        save_pred_dict['annotations'].append(this_pred)

    # Append the feature save dictionary to the save_feats list
    save_feats.append(feat_save)

# Save output files
os.makedirs(os.path.join('/',*args.out_coco_anns_path.split('/')[1:-1]), exist_ok=True)
with open(args.out_coco_anns_path, 'w') as f:
    json.dump(save_ann_dict, f, indent=4)   

os.makedirs(os.path.join('/',*args.out_coco_preds_path.split('/')[1:-1]), exist_ok=True)
with open(args.out_coco_preds_path, 'w') as b:
    json.dump(save_pred_dict, b, indent=4)  

os.makedirs(os.path.join('/',*args.out_features_path.split('/')[1:-1]), exist_ok=True)
torch.save(save_feats, args.out_features_path)  