import json
import argparse
import os
from pycocotools import mask
import numpy as np
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

def rle_to_polygon(coco_json_path, output_json_path):
    """
    Convert RLE annotations to polygon format in a COCO JSON.

    Parameters:
    - coco_json_path: path to the input COCO JSON with RLE annotations.
    - output_json_path: path to save the modified COCO JSON with polygon annotations.
    """
    # Load the COCO JSON
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Iterate through the annotations
    for ann in tqdm(data['annotations']):
        if 'segmentation' in ann and type(ann['segmentation']) is not list:
            
            # Decode RLE to binary mask
            binary_mask = mask.decode(ann['segmentation'])
            
            # Convert binary mask to contours using OpenCV
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation = []
            for contour in contours:
                contour = contour.squeeze().ravel().tolist()

                # Exclude invalid polygons
                if len(contour)%2 == 0 and len(contour) >= 6:
                    segmentation.append(contour)
            
            # Replace RLE with polygon
            ann['segmentation'] = segmentation

    # Save the modified COCO JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    # Parse GraSP annotation files
    for split in ['fold1', 'fold2', 'train', 'test']:
        input_coco_json = os.path.join(args.data_path, f"grasp_short-term_{split}.json")
        output_coco_json = os.path.join(args.data_path, f"grasp_short-term_{split}_polygon.json")
        rle_to_polygon(input_coco_json, output_coco_json)
