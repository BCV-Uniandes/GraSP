# ############################################################### Mask2Former SwinL

# python match_annots_n_preds.py --coco_anns_path /media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/GraSP/GraSP_30fps/grasp_short-term_fold1.json \
# --preds_path /media/SSD0/nayobi/Endovis/Mask2Former/output/PSI-AVA/SwinL/Fold1/train_features/inference/instances_predictions.pth \
# --out_coco_anns_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold1/train_anns.json \
# --out_coco_preds_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold1/train_preds.json \
# --out_features_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/features/fold1/train_region_features.pth \
# --selection all --segmentation --features_key decoder_output

# python match_annots_n_preds.py --coco_anns_path /media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/GraSP/GraSP_30fps/grasp_short-term_fold2.json \
# --preds_path /media/SSD0/nayobi/Endovis/Mask2Former/output/PSI-AVA/SwinL/Fold2/train_features/inference/instances_predictions.pth \
# --out_coco_anns_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold2/train_anns.json \
# --out_coco_preds_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold2/train_preds.json \
# --out_features_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/features/fold2/train_region_features.pth \
# --selection all --segmentation --features_key decoder_output

# python match_annots_n_preds.py --coco_anns_path /media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/GraSP/GraSP_30fps/grasp_short-term_fold1.json \
# --preds_path /media/SSD0/nayobi/Endovis/Mask2Former/output/PSI-AVA/SwinL/Fold2/inference/instances_predictions.pth \
# --out_coco_anns_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold2/fold1_anns.json \
# --out_coco_preds_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold2/fold1_preds.json \
# --out_features_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/features/fold2/fold1_preds_region_features.pth \
# --selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_output

# python match_annots_n_preds.py --coco_anns_path /media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/GraSP/GraSP_30fps/grasp_short-term_fold2.json \
# --preds_path /media/SSD0/nayobi/Endovis/Mask2Former/output/PSI-AVA/SwinL/Fold1/inference/instances_predictions.pth \
# --out_coco_anns_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold1/fold2_anns.json \
# --out_coco_preds_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold1/fold2_preds.json \
# --out_features_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/features/fold1/fold2_preds_region_features.pth \
# --selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_output

# ############################################################### Mask2Former SwinL Test

# python match_annots_n_preds.py --coco_anns_path /media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/GraSP/GraSP_30fps/grasp_short-term_train.json \
# --preds_path /media/SSD0/nayobi/Endovis/Mask2Former/output/PSI-AVA/SwinL/Test/train_features/inference/instances_predictions.pth \
# --out_coco_anns_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/train/train_anns.json \
# --out_coco_preds_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/train/train_preds.json \
# --out_features_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/features/train/train_region_features.pth \
# --selection all --segmentation --features_key decoder_output

# python match_annots_n_preds.py --coco_anns_path /media/SSD1/nayobi/All_datasets/PSI-AVA/annotations/PSI-AVA_v2_final/GraSP/GraSP_30fps/grasp_short-term_test.json \
# --preds_path /media/SSD0/nayobi/Endovis/Mask2Former/output/PSI-AVA/SwinL/Test/inference/instances_predictions.pth \
# --out_coco_anns_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/train/test_anns.json \
# --out_coco_preds_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/train/test_preds.json \
# --out_features_path /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/features/train/test_preds_region_features.pth \
# --selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_output

############################################################### Mask2Former SwinL

python match_annots_n_preds.py --coco_anns_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/grasp_short-term_fold1.json \
--preds_path /data/nayobi/Endovis/GraSP/TAPIS/region_proposals/output/Fold1_generic/Train_inference/instances_predictions.pth \
--out_coco_anns_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold1_anns.json \
--out_coco_preds_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/Generic/fold1_train_preds.json \
--out_features_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/features/Generic/fold1_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/grasp_short-term_fold2.json \
--preds_path /data/nayobi/Endovis/GraSP/TAPIS/region_proposals/output/Fold2_generic/Train_inference/instances_predictions.pth \
--out_coco_anns_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold2_anns.json \
--out_coco_preds_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/Generic/fold2_train_preds.json \
--out_features_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/features/Generic/fold2_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/grasp_short-term_fold1.json \
--preds_path /data/nayobi/Endovis/GraSP/TAPIS/region_proposals/output/Fold2_generic/inference/instances_predictions.pth \
--out_coco_anns_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold1_anns.json \
--out_coco_preds_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/Generic/fold1_val_preds.json \
--out_features_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/features/Generic/fold1_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/grasp_short-term_fold2.json \
--preds_path /data/nayobi/Endovis/GraSP/TAPIS/region_proposals/output/Fold1_generic/inference/instances_predictions.pth \
--out_coco_anns_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/fold2_anns.json \
--out_coco_preds_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/Generic/fold2_val_preds.json \
--out_features_path /data/nayobi/Endovis/GraSP/TAPIS/data/GraSP/annotations/features/Generic/fold2_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out