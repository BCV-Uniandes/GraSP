# Experiment setup
TRAIN_FOLD="fold2" # or fold1, train
TEST_FOLD="fold1" # or fold2, test
EXP_PREFIX="test_run" # costumize
TASK="PHASES"
ARCH="TAPIS"

#-------------------------
DATASET="GraSP"
EXPERIMENT_NAME=$EXP_PREFIX"/"$TRAIN_FOLD
CONFIG_PATH="configs/"$DATASET"/"$ARCH"/"$ARCH"_"$TASK".yaml"
OUTPUT_DIR="./outputs/"$DATASET"/"$TASK"/"$EXPERIMENT_NAME

# Change this variables if data is not located in ./data
FRAME_DIR="./data/"$DATASET"/frames"
FRAME_LIST="./data/"$DATASET"/frame_lists"
ANNOT_DIR="./data/"$DATASET"/annotations/"
COCO_ANN_PATH="./data/"$DATASET"/annotations/grasp_long-term_"$TEST_FOLD".json"
CHECKPOINT="./data/GraSP/pretrained_models/"$TRAIN_FOLD"/"$TASK".pyth"

#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/GraSP/TAPIS/tapis:$PYTHONPATH

mkdir -p $OUTPUT_DIR

python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 4 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TEST.ENABLE True \
TRAIN.ENABLE False \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.TRAIN_LISTS $TRAIN_FOLD".csv" \
ENDOVIS_DATASET.TEST_LISTS $TEST_FOLD".csv" \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.TRAIN_GT_BOX_JSON "grasp_long-term_"$TRAIN_FOLD".json" \
ENDOVIS_DATASET.TEST_GT_BOX_JSON "grasp_long-term_"$TEST_FOLD".json" \
ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
TRAIN.BATCH_SIZE 96 \
TEST.BATCH_SIZE 96 \
OUTPUT_DIR $OUTPUT_DIR 