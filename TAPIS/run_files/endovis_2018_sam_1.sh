# Experiment setup
EXP_NAME="MATIS_sam"
TASK="TOOLS" 
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/SAM_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018/annotations"
COCO_ANN_PATH="./data/endovis_2018/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018/features/region_features_mask_train.pth" 
FF_VAL="./data/endovis_2018/features/region_features_mask_val.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
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
FEATURES.MODEL sam \
MODEL.TRANS False \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
ENDOVIS_DATASET.INCLUDE_GT True \
TASKS.PRESENCE_RECOGNITION True \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

##################################################################################################################################3

# Experiment setup
EXP_NAME="MATIS_sam_mlp"
TASK="TOOLS" 
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/SAM_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018/annotations"
COCO_ANN_PATH="./data/endovis_2018/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018/features/region_features_mask_train.pth" 
FF_VAL="./data/endovis_2018/features/region_features_mask_val.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
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
FEATURES.MODEL sam \
FEATURES.SAM_MLP True \
MODEL.TRANS False \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
ENDOVIS_DATASET.INCLUDE_GT True \
TASKS.PRESENCE_RECOGNITION True \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

##################################################################################################################################3

# Experiment setup
EXP_NAME="MATIS_sam_prompt"
TASK="TOOLS" 
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/SAM_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018/annotations"
COCO_ANN_PATH="./data/endovis_2018/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018/features/region_features_mask_train.pth" 
FF_VAL="./data/endovis_2018/features/region_features_mask_val.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
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
FEATURES.MODEL sam \
FEATURES.SAM_PROMPT True \
MODEL.TRANS False \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
ENDOVIS_DATASET.INCLUDE_GT True \
TASKS.PRESENCE_RECOGNITION True \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

##################################################################################################################################3

# Experiment setup
EXP_NAME="MATIS_sam_box"
TASK="TOOLS" 
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/SAM_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018/annotations"
COCO_ANN_PATH="./data/endovis_2018/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018/features/region_features_mask_train.pth" 
FF_VAL="./data/endovis_2018/features/region_features_mask_val.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
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
FEATURES.MODEL sam \
FEATURES.SAM_BOX True \
MODEL.TRANS False \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
ENDOVIS_DATASET.INCLUDE_GT False \
TASKS.PRESENCE_RECOGNITION True \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

##################################################################################################################################3

# Experiment setup
EXP_NAME="MATIS_sam_mlp_prompt"
TASK="TOOLS" 
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/SAM_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018/annotations"
COCO_ANN_PATH="./data/endovis_2018/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018/features/region_features_mask_train.pth" 
FF_VAL="./data/endovis_2018/features/region_features_mask_val.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
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
FEATURES.MODEL sam \
FEATURES.SAM_MLP True \
FEATURES.SAM_PROMPT True \
MODEL.TRANS False \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
ENDOVIS_DATASET.INCLUDE_GT True \
TASKS.PRESENCE_RECOGNITION True \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

##################################################################################################################################3

# Experiment setup
EXP_NAME="MATIS_sam_mlp_box"
TASK="TOOLS" 
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/SAM_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018/annotations"
COCO_ANN_PATH="./data/endovis_2018/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018/features/region_features_mask_train.pth" 
FF_VAL="./data/endovis_2018/features/region_features_mask_val.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
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
FEATURES.MODEL sam \
FEATURES.SAM_MLP True \
FEATURES.SAM_BOX True \
MODEL.TRANS False \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
ENDOVIS_DATASET.INCLUDE_GT False \
TASKS.PRESENCE_RECOGNITION True \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

##################################################################################################################################3

# Experiment setup
EXP_NAME="MATIS_sam_prompt_box"
TASK="TOOLS" 
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/SAM_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018/annotations"
COCO_ANN_PATH="./data/endovis_2018/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018/features/region_features_mask_train.pth" 
FF_VAL="./data/endovis_2018/features/region_features_mask_val.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
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
FEATURES.MODEL sam \
FEATURES.SAM_BOX True \
FEATURES.SAM_PROMPT True \
MODEL.TRANS False \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
ENDOVIS_DATASET.INCLUDE_GT False \
TASKS.PRESENCE_RECOGNITION True \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

##################################################################################################################################3

# Experiment setup
EXP_NAME="MATIS_sam_mlp_prompt_box"
TASK="TOOLS" 
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/SAM_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018/annotations"
COCO_ANN_PATH="./data/endovis_2018/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018/features/region_features_mask_train.pth" 
FF_VAL="./data/endovis_2018/features/region_features_mask_val.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
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
FEATURES.MODEL sam \
FEATURES.SAM_MLP True \
FEATURES.SAM_PROMPT True \
FEATURES.SAM_BOX True \
MODEL.TRANS False \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
ENDOVIS_DATASET.INCLUDE_GT False \
TASKS.PRESENCE_RECOGNITION True \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 