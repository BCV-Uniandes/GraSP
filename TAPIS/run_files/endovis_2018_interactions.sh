# Experiment setup
EXP_NAME="baseline_100-epoch_e4-wd_125-blr_0000125-stlr_e5-enlr"
TASK="ACTIONS" 
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MATIS/data/endovis_2018/models/matis_pretrained_model.pyth"
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/MVIT_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018_Interactions/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018_interactions/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018_interactions/annotations"
COCO_ANN_PATH="./data/endovis_2018_interactions/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018_interactions/features/region_features_decoder_train.pth" 
FF_VAL="./data/endovis_2018_interactions/features/region_features_decoder_val.pth"

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
SOLVER.MAX_EPOCH 100 \
SOLVER.WEIGHT_DECAY 1e-4 \
SOLVER.BASE_LR 0.125 \
SOLVER.WARMUP_START_LR 0.0000125 \
SOLVER.COSINE_END_LR 1e-5 \
DATA.JUST_CENTER False \
TRAIN.BATCH_SIZE 30 \
TEST.BATCH_SIZE 30 \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

############################################################################

# Experiment setup
EXP_NAME="baseline_100-epoch_e4-wd_0125-blr_0000125-stlr_e5-enlr"
TASK="ACTIONS" 
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MATIS/data/endovis_2018/models/matis_pretrained_model.pyth"
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/MVIT_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018_Interactions/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018_interactions/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018_interactions/annotations"
COCO_ANN_PATH="./data/endovis_2018_interactions/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018_interactions/features/region_features_decoder_train.pth" 
FF_VAL="./data/endovis_2018_interactions/features/region_features_decoder_val.pth"

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
SOLVER.MAX_EPOCH 100 \
SOLVER.WEIGHT_DECAY 1e-4 \
SOLVER.BASE_LR 0.0125 \
SOLVER.WARMUP_START_LR 0.0000125 \
SOLVER.COSINE_END_LR 1e-5 \
DATA.JUST_CENTER False \
TRAIN.BATCH_SIZE 30 \
TEST.BATCH_SIZE 30 \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 

############################################################################

# Experiment setup
EXP_NAME="baseline_200-epoch_e4-wd_0125-blr_0000125-stlr_e5-enlr"
TASK="ACTIONS" 
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MATIS/data/endovis_2018/models/matis_pretrained_model.pyth"
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/MVIT_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018_Interactions/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018_interactions/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018_interactions/annotations"
COCO_ANN_PATH="./data/endovis_2018_interactions/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018_interactions/features/region_features_decoder_train.pth" 
FF_VAL="./data/endovis_2018_interactions/features/region_features_decoder_val.pth"

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
SOLVER.MAX_EPOCH 200 \
SOLVER.WEIGHT_DECAY 1e-4 \
SOLVER.BASE_LR 0.0125 \
SOLVER.WARMUP_START_LR 0.0000125 \
SOLVER.COSINE_END_LR 1e-5 \
DATA.JUST_CENTER False \
TRAIN.BATCH_SIZE 30 \
TEST.BATCH_SIZE 30 \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR

############################################################################

# Experiment setup
EXP_NAME="baseline_200-epoch_e4-wd_125-blr_0000125-stlr_e5-enlr"
TASK="ACTIONS" 
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MATIS/data/endovis_2018/models/matis_pretrained_model.pyth"
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/MVIT_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018_Interactions/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018_interactions/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018_interactions/annotations"
COCO_ANN_PATH="./data/endovis_2018_interactions/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018_interactions/features/region_features_decoder_train.pth" 
FF_VAL="./data/endovis_2018_interactions/features/region_features_decoder_val.pth"

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
SOLVER.MAX_EPOCH 100 \
SOLVER.WEIGHT_DECAY 1e-4 \
SOLVER.BASE_LR 0.125 \
SOLVER.WARMUP_START_LR 0.0000125 \
SOLVER.COSINE_END_LR 1e-5 \
DATA.JUST_CENTER False \
TRAIN.BATCH_SIZE 30 \
TEST.BATCH_SIZE 30 \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR

# ############################################################################

# # Experiment setup
# EXP_NAME="baseline_100-epoch_e5-wd_125-blr"
# TASK="ACTIONS" 
# # CHECKPOINT="/media/SSD0/nayobi/Endovis/MATIS/data/endovis_2018/models/matis_pretrained_model.pyth"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

# #-------------------------
# CONFIG_PATH="configs/EndoVis-2018/MVIT_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/EndoVis_2018_Interactions/"$TASK"/"$EXP_NAME
# FRAME_LIST="./data/endovis_2018_interactions/annotations/frame_lists"
# ANNOT_DIR="./data/endovis_2018_interactions/annotations"
# COCO_ANN_PATH="./data/endovis_2018_interactions/annotations/val_coco_anns.json"
# FF_TRAIN="./data/endovis_2018_interactions/features/region_features_decoder_train.pth" 
# FF_VAL="./data/endovis_2018_interactions/features/region_features_decoder_val.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# SOLVER.MAX_EPOCH 100 \
# SOLVER.WEIGHT_DECAY 1e-5 \
# SOLVER.BASE_LR 0.125 \
# DATA.JUST_CENTER False \
# TRAIN.BATCH_SIZE 30 \
# TEST.BATCH_SIZE 30 \
# DATA_LOADER.NUM_WORKERS 5 \
# OUTPUT_DIR $OUTPUT_DIR 

# ############################################################################

# # Experiment setup
# EXP_NAME="baseline_100-epoch_e6-wd_125-blr"
# TASK="ACTIONS" 
# # CHECKPOINT="/media/SSD0/nayobi/Endovis/MATIS/data/endovis_2018/models/matis_pretrained_model.pyth"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

# #-------------------------
# CONFIG_PATH="configs/EndoVis-2018/MVIT_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/EndoVis_2018_Interactions/"$TASK"/"$EXP_NAME
# FRAME_LIST="./data/endovis_2018_interactions/annotations/frame_lists"
# ANNOT_DIR="./data/endovis_2018_interactions/annotations"
# COCO_ANN_PATH="./data/endovis_2018_interactions/annotations/val_coco_anns.json"
# FF_TRAIN="./data/endovis_2018_interactions/features/region_features_decoder_train.pth" 
# FF_VAL="./data/endovis_2018_interactions/features/region_features_decoder_val.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# SOLVER.MAX_EPOCH 100 \
# SOLVER.WEIGHT_DECAY 1e-6 \
# SOLVER.BASE_LR 0.125 \
# DATA.JUST_CENTER False \
# TRAIN.BATCH_SIZE 30 \
# TEST.BATCH_SIZE 30 \
# DATA_LOADER.NUM_WORKERS 5 \
# OUTPUT_DIR $OUTPUT_DIR 

# ############################################################################

# # Experiment setup
# EXP_NAME="baseline_100-epoch_e2-wd_125-blr"
# TASK="ACTIONS" 
# # CHECKPOINT="/media/SSD0/nayobi/Endovis/MATIS/data/endovis_2018/models/matis_pretrained_model.pyth"
# CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

# #-------------------------
# CONFIG_PATH="configs/EndoVis-2018/MVIT_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/EndoVis_2018_Interactions/"$TASK"/"$EXP_NAME
# FRAME_LIST="./data/endovis_2018_interactions/annotations/frame_lists"
# ANNOT_DIR="./data/endovis_2018_interactions/annotations"
# COCO_ANN_PATH="./data/endovis_2018_interactions/annotations/val_coco_anns.json"
# FF_TRAIN="./data/endovis_2018_interactions/features/region_features_decoder_train.pth" 
# FF_VAL="./data/endovis_2018_interactions/features/region_features_decoder_val.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -W ignore tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# SOLVER.MAX_EPOCH 100 \
# SOLVER.WEIGHT_DECAY 1e-2 \
# SOLVER.BASE_LR 0.125 \
# DATA.JUST_CENTER False \
# TRAIN.BATCH_SIZE 30 \
# TEST.BATCH_SIZE 30 \
# DATA_LOADER.NUM_WORKERS 5 \
# OUTPUT_DIR $OUTPUT_DIR 

############################################################################

# Experiment setup
EXP_NAME="baseline_100-epoch_e4-wd_125-blr_adam"
TASK="ACTIONS" 
# CHECKPOINT="/media/SSD0/nayobi/Endovis/MATIS/data/endovis_2018/models/matis_pretrained_model.pyth"
CHECKPOINT="/media/SSD0/nayobi/Endovis/ISBI_TAPIR/models/K400_MVIT_B_16x4_CONV.pyth"

#-------------------------
CONFIG_PATH="configs/EndoVis-2018/MVIT_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/EndoVis_2018_Interactions/"$TASK"/"$EXP_NAME
FRAME_LIST="./data/endovis_2018_interactions/annotations/frame_lists"
ANNOT_DIR="./data/endovis_2018_interactions/annotations"
COCO_ANN_PATH="./data/endovis_2018_interactions/annotations/val_coco_anns.json"
FF_TRAIN="./data/endovis_2018_interactions/features/region_features_decoder_train.pth" 
FF_VAL="./data/endovis_2018_interactions/features/region_features_decoder_val.pth"

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
SOLVER.MAX_EPOCH 100 \
SOLVER.WEIGHT_DECAY 1e-4 \
SOLVER.BASE_LR 0.125 \
SOLVER.OPTIMIZING_METHOD adam \
DATA.JUST_CENTER False \
TRAIN.BATCH_SIZE 30 \
TEST.BATCH_SIZE 30 \
DATA_LOADER.NUM_WORKERS 5 \
OUTPUT_DIR $OUTPUT_DIR 