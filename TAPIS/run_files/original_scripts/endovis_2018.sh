# Experiment setup
EXP_NAME="MATIS_test"
TASK="ACTIONS" 
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MATIS/data/endovis_2018/models/matis_pretrained_model.pyth"
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/MVIT_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/endovis_2018/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018/annotations"
COCO_ANN_PATH="./data/endovis_2018/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018/features/region_features_decoder_train.pth" 
FF_VAL="./data/endovis_2018/features/region_features_decoder_val.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -W ignore tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
SOLVER.MAX_EPOCH 100 \
SOLVER.WEIGHT_DECAY 1e-4 \
SOLVER.BASE_LR 0.125 \
DATA.JUST_CENTER False \
TRAIN.BATCH_SIZE 6 \
TEST.BATCH_SIZE 6 \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 