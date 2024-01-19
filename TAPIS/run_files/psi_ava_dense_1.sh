# # Experiment setup
# FOLD="fold1" 
# EXP_NAME="MVITv1_dense_dino_swinl"
# TASK="ACTIONS"
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_dino_swinl.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_dino_swinl.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_dino_swinl.json" \
# ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_dino_swinl.json" \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# OUTPUT_DIR $OUTPUT_DIR 

# #################################################################################################################### 

# # Experiment setup
# FOLD="fold2" 
# EXP_NAME="MVITv1_dense_dino_swinl"
# TASK="ACTIONS"
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_dino_swinl.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_dino_swinl.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_dino_swinl.json" \
# ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_dino_swinl.json" \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# OUTPUT_DIR $OUTPUT_DIR 

# ########################################################################################################################

# Experiment setup
FOLD="fold2" 
EXP_NAME="MVITv1_decoder_dense_mf_swinl"
TASK="ACTIONS"
ARCH="MVIT"
CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava_dense"
EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask2former_swinl.pth" 
FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask2former_swinl.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
DATA.NUM_FRAMES 16 \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask2former_swinl.json" \
ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask2former_swinl.json" \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TEST.DATASET psi_ava_dense \
TRAIN.DATASET psi_ava_dense \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 28 \
MODEL.DECODER True \
OUTPUT_DIR $OUTPUT_DIR 

# ########################################################################################################################

# Experiment setup
FOLD="fold1" 
EXP_NAME="MVITv1_decoder_dense_mf_swinl_12"
TASK="ACTIONS"
ARCH="MVIT"
CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava_dense"
EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask2former_swinl.pth" 
FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask2former_swinl.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
DATA.NUM_FRAMES 12 \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask2former_swinl.json" \
ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask2former_swinl.json" \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TEST.DATASET psi_ava_dense \
TRAIN.DATASET psi_ava_dense \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 28 \
MODEL.DECODER True \
OUTPUT_DIR $OUTPUT_DIR 

# ########################################################################################################################

# Experiment setup
FOLD="fold2" 
EXP_NAME="MVITv1_decoder_dense_mf_swinl_8"
TASK="ACTIONS"
ARCH="MVIT"
CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava_dense"
EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask2former_swinl.pth" 
FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask2former_swinl.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
DATA.NUM_FRAMES 8 \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask2former_swinl.json" \
ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask2former_swinl.json" \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TEST.DATASET psi_ava_dense \
TRAIN.DATASET psi_ava_dense \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 28 \
MODEL.DECODER True \
OUTPUT_DIR $OUTPUT_DIR 

# ########################################################################################################################

# Experiment setup
FOLD="fold2" 
EXP_NAME="MVITv1_decoder_dense_mf_swinl_4"
TASK="ACTIONS"
ARCH="MVIT"
CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava_dense"
EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask2former_swinl.pth" 
FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask2former_swinl.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
DATA.NUM_FRAMES 4 \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask2former_swinl.json" \
ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask2former_swinl.json" \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TEST.DATASET psi_ava_dense \
TRAIN.DATASET psi_ava_dense \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 28 \
MODEL.DECODER True \
OUTPUT_DIR $OUTPUT_DIR 

# ########################################################################################################################

# Experiment setup
FOLD="fold2" 
EXP_NAME="MVITv1_decoder_dense_mf_swinl_20"
TASK="ACTIONS"
ARCH="MVIT"
CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

#-------------------------
DATA_VER="psi-ava_dense"
EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
FRAME_DIR="./data" 
OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features_mask2former_swinl.pth" 
FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features_mask2former_swinl.pth"

TYPE="pytorch"
#-------------------------
# Run experiment

export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

mkdir -p $OUTPUT_DIR

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
TRAIN.ENABLE True \
DATA.NUM_FRAMES 20 \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON "train_coco_preds_mask2former_swinl.json" \
ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON "val_coco_preds_mask2former_swinl.json" \
BN.NUM_BATCHES_PRECISE 72 \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.VAL_FEATURES_PATH $FF_VAL \
TEST.DATASET psi_ava_dense \
TRAIN.DATASET psi_ava_dense \
TRAIN.BATCH_SIZE 18 \
TEST.BATCH_SIZE 28 \
MODEL.DECODER True \
OUTPUT_DIR $OUTPUT_DIR 

# # ########################################################################################################################

# # Experiment setup
# FOLD="fold2" 
# BASE_LR=0.125
# GRAD_NORM=1.0
# LR_POLICY=cosine
# END_LR=1e-4
# MAX_EPOCH=100
# WARMUP_EPOCHS=5.0
# WEIGHT_DECAY=1e-4
# START_LR=0.000125
# OPTIMIZER=sgd
# EXP_NAME="blr-"$BASE_LR"_gn-"$GRAD_NORM"_pol-"$LR_POLICY"_elr-"$END_LR"_eps-"$MAX_EPOCH"_weps-"$WARMUP_EPOCHS"_slr-"$START_LR"_op-"$OPTIMIZER
# TASK="ACTIONS" 
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/Optim_Dense/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# SOLVER.BASE_LR $BASE_LR \
# SOLVER.CLIP_GRAD_L2NORM $GRAD_NORM \
# SOLVER.LR_POLICY $LR_POLICY \
# SOLVER.COSINE_END_LR $END_LR \
# SOLVER.MAX_EPOCH $MAX_EPOCH \
# SOLVER.WARMUP_EPOCHS $WARMUP_EPOCHS \
# SOLVER.WEIGHT_DECAY $WEIGHT_DECAY \
# SOLVER.WARMUP_START_LR  $START_LR \
# SOLVER.OPTIMIZING_METHOD $OPTIMIZER \
# OUTPUT_DIR $OUTPUT_DIR 

# # ########################################################################################################################

# # Experiment setup
# FOLD="fold2" 
# BASE_LR=0.0125
# GRAD_NORM=1.0
# LR_POLICY=cosine
# END_LR=1e-4
# MAX_EPOCH=100
# WARMUP_EPOCHS=5.0
# WEIGHT_DECAY=1e-4
# START_LR=0.000125
# OPTIMIZER=sgd
# EXP_NAME="blr-"$BASE_LR"_gn-"$GRAD_NORM"_pol-"$LR_POLICY"_elr-"$END_LR"_eps-"$MAX_EPOCH"_weps-"$WARMUP_EPOCHS"_slr-"$START_LR"_op-"$OPTIMIZER
# TASK="ACTIONS" 
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/Optim_Dense/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# SOLVER.BASE_LR $BASE_LR \
# SOLVER.CLIP_GRAD_L2NORM $GRAD_NORM \
# SOLVER.LR_POLICY $LR_POLICY \
# SOLVER.COSINE_END_LR $END_LR \
# SOLVER.MAX_EPOCH $MAX_EPOCH \
# SOLVER.WARMUP_EPOCHS $WARMUP_EPOCHS \
# SOLVER.WEIGHT_DECAY $WEIGHT_DECAY \
# SOLVER.WARMUP_START_LR  $START_LR \
# SOLVER.OPTIMIZING_METHOD $OPTIMIZER \
# OUTPUT_DIR $OUTPUT_DIR 

# # ########################################################################################################################

# # Experiment setup
# FOLD="fold2" 
# BASE_LR=0.00125
# GRAD_NORM=1.0
# LR_POLICY=cosine
# END_LR=1e-4
# MAX_EPOCH=100
# WARMUP_EPOCHS=5.0
# WEIGHT_DECAY=1e-4
# START_LR=0.000125
# OPTIMIZER=sgd
# EXP_NAME="blr-"$BASE_LR"_gn-"$GRAD_NORM"_pol-"$LR_POLICY"_elr-"$END_LR"_eps-"$MAX_EPOCH"_weps-"$WARMUP_EPOCHS"_slr-"$START_LR"_op-"$OPTIMIZER
# TASK="ACTIONS" 
# ARCH="MVIT"
# CHECKPOINT="/home/nayobi/Endovis/Final-TAPIR/data/PSIAVA_v2/weights/"$FOLD"/checkpoint_best_phases.pyth"

# #-------------------------
# DATA_VER="psi-ava_dense"
# EXPERIMENT_NAME=$EXP_NAME"/"$FOLD
# CONFIG_PATH="configs/PSI-AVA/"$ARCH"_"$TASK".yaml"
# FRAME_DIR="./data" 
# OUTPUT_DIR="outputs/PSI_AVAv2/"$TASK"/Optim_Dense/"$EXPERIMENT_NAME
# FRAME_LIST="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/frame_lists"
# ANNOT_DIR="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns"
# COCO_ANN_PATH="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/coco_anns/val_coco_anns.json"
# FF_TRAIN="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/train/bbox_features.pth" 
# FF_VAL="./data/PSIAVA_v2/data_annotations/"$DATA_VER"/"$FOLD"/val/bbox_features.pth"

# TYPE="pytorch"
# #-------------------------
# # Run experiment

# export PYTHONPATH=/home/nayobi/Endovis/Final-TAPIR/tapir:$PYTHONPATH

# mkdir -p $OUTPUT_DIR

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -B tools/run_net.py \
# --cfg $CONFIG_PATH \
# NUM_GPUS 1 \
# TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
# TRAIN.CHECKPOINT_EPOCH_RESET True \
# TRAIN.CHECKPOINT_TYPE $TYPE \
# TEST.ENABLE False \
# TRAIN.ENABLE True \
# TEST.DATASET psi_ava_dense \
# TRAIN.DATASET psi_ava_dense \
# DATA.NUM_FRAMES 16 \
# ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
# ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
# ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
# ENDOVIS_DATASET.COCO_ANN_DIR $COCO_ANN_PATH \
# BN.NUM_BATCHES_PRECISE 72 \
# FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
# FEATURES.VAL_FEATURES_PATH $FF_VAL \
# TRAIN.BATCH_SIZE 24 \
# TEST.BATCH_SIZE 28 \
# SOLVER.BASE_LR $BASE_LR \
# SOLVER.CLIP_GRAD_L2NORM $GRAD_NORM \
# SOLVER.LR_POLICY $LR_POLICY \
# SOLVER.COSINE_END_LR $END_LR \
# SOLVER.MAX_EPOCH $MAX_EPOCH \
# SOLVER.WARMUP_EPOCHS $WARMUP_EPOCHS \
# SOLVER.WEIGHT_DECAY $WEIGHT_DECAY \
# SOLVER.WARMUP_START_LR  $START_LR \
# SOLVER.OPTIMIZING_METHOD $OPTIMIZER \
# OUTPUT_DIR $OUTPUT_DIR 