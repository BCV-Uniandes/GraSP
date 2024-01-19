# Experiment setup
FOLD="1" 
EXP_NAME="Dense_MVITv1_baseline_phases_pretrain"
TASK="ACTIONS" 
ARCH="MVIT"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MICCAI2023/MaskFeat-TAPIR/outputs/k400_MVIT_L_MaskFeat_PT_epoch_00800.pyth"
CHECKPOINT="/media/SSD0/nayobi/Endovis/Final-TAPIR/outputs/PSI_AVA/PHASES/Dense_MVITv1_baseline/Fold1/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava_dense"
EXPERIMENT_NAME=$EXP_NAME"/Fold"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVA/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA/data_annotations/"$DATA_VER"/fold"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA/data_annotations/"$DATA_VER"/fold"$FOLD"/train/bbox_features.pth" 
FF_VAL="./data/PSIAVA/data_annotations/"$DATA_VER"/fold"$FOLD"/val/bbox_features.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TEST.DATASET psi_ava_dense \
TRAIN.ENABLE True \
TRAIN.DATASET psi_ava_dense \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

#####################################################################################