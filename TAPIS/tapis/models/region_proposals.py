import traceback
from region_proposals import MaskFormer
import torch.nn as nn
import torch
from scipy.optimize import linear_sum_assignment

class RegionProposal(nn.Module):
    def __init__(self, cfg, crop_size, use_masks):
        super(RegionProposal, self).__init__()
        self.rpn = MaskFormer(cfg)
        self.y_crop = crop_size[0]
        self.x_crop = crop_size[1]
        self.use_masks = use_masks
    
    def compute_iou(self, pred_boxes, gt_boxes):
        """
        Compute the IoU (Intersection over Union) between two sets of bounding boxes.
        :param pred_boxes: Tensor of size (P, 4), predicted boxes.
        :param gt_boxes: Tensor of size (G, 4), ground truth boxes.
        :return: Tensor of size (P, G), pairwise IoU scores.
        """
        # Expand dimensions to compute pairwise IoU
        pred_boxes = pred_boxes.unsqueeze(1)  # (P, 1, 4)
        gt_boxes = gt_boxes.unsqueeze(0)      # (1, G, 4)
        
        # Compute intersection coordinates
        inter_min = torch.max(pred_boxes[..., :2], gt_boxes[..., :2])  # Top-left corner
        inter_max = torch.min(pred_boxes[..., 2:], gt_boxes[..., 2:])  # Bottom-right corner
        inter = torch.clamp(inter_max - inter_min, min=0)  # Intersection width and height
        inter_area = inter[..., 0] * inter[..., 1]  # Intersection area
        
        # Compute areas of each box
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        gt_area = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
        
        # Compute union area
        union_area = pred_area + gt_area - inter_area
        
        # IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        return iou
    
    def forward(self, inputs, bboxes, boxes_mask, training):
        regions = self.rpn(inputs)

        out_features = []
        out_boxes = []
        out_masks = []
        for image, bbox, box_mask, (pred_boxes, box_features, pred_masks) in zip(inputs, bboxes, boxes_mask, regions):

            _, h, w = image.shape

            gt_boxes = bbox[box_mask]  # Shape: (G, 4), G <= 5
            gt_boxes[:,[0,2]] *= (h/self.y_crop)
            gt_boxes[:,[1,3]] *= (w/self.x_crop)

            pred_boxes = pred_boxes.tensor.to(gt_boxes.device)
            
            # Compute IoU between predictions and valid ground truth boxes
            iou_matrix = self.compute_iou(pred_boxes, gt_boxes)  # Shape: (10, G)
            
            # Hungarian algorithm for bipartite matching (maximize IoU)
            cost_matrix = -iou_matrix.cpu().numpy()  # Negative IoU as cost
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

            best_features = torch.zeros((boxes_mask.shape[-1],box_features.shape[-1])).to(boxes_mask.device)
            best_features[gt_indices] = box_features[pred_indices]
            
            out_features.append(best_features)

            if self.use_masks and not training:
                best_boxes = torch.zeros((boxes_mask.shape[-1], 4)).to(boxes_mask.device)
                best_boxes[gt_indices] = pred_boxes[pred_indices]

                out_boxes.append(best_boxes)

                best_masks = torch.zeros((boxes_mask.shape[-1], pred_masks[0].shape[0], pred_masks[0].shape[1])).to(boxes_mask.device)
                best_masks[gt_indices] = pred_masks[pred_indices]

                out_masks.append(best_masks)

        if self.use_masks and not training:
            assert len(out_features)==len(out_boxes)==len(out_masks), f'Inconsistent output regions batch {len(out_features)} {len(out_boxes)} {len(out_masks)}'
            return torch.stack(out_features, dim=0), torch.stack(out_boxes, dim=0), torch.stack(out_masks, dim=0)      
        else:
            return torch.stack(out_features, dim=0), None, None