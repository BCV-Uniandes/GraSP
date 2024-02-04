CUDA_VISIBLE_DEVICES=3 python train_net.py --num-gpus 1 --eval-only \
--config-file configs/grasp/GraSP_R50_fold1.yaml \
OUTPUT_DIR output/R50/fold1 \
MODEL.WEIGHTS /media/SSD0/nayobi/Endovis/GraSP/TAPIS/data/GraSP/pretrained_models/SEGMENTATION_BASELINE/fold1.pth