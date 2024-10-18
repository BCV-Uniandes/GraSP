# TAPIS: Transformers for Actions, Phases, Steps and Instrument Segmentation

<div align="center">
  <img src="../Images/TAPIS.jpg"/>
</div><br/>

We present the Transformers for Actions, Phases, Steps, and Instrument Segmentation (TAPIS) model, a generalized architecture designed to tackle all the proposed tasks in the GraSP benchmark. Our method utilizes a localized instrument segmentation baseline applied on independent keyframes that act as a region proposal network and provide pixel-precise instrument masks and their corresponding segment embeddings. Further, our model uses a global video feature extractor on time windows centered on a keyframe to compute a class embedding and a sequence of spatio-temporal embeddings. A *frame classification head* uses the class embedding to classify the middle frame of the time window into a phase or a step, and a *region classification head* interrelates the global spatio-temporal features with the localized region embeddings for atomic action prediction or instrument region classification. In the following subsections, we explain the details of our proposed architecture.

## Previous works

This work is an extended and consolidated version of three previous works:

- [Towards Holistic Surgical Scene Understanding](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_42), **MICCAI 2022, Oral.** Code [here](https://github.com/BCV-Uniandes/TAPIR).
- **Winner solution** of the [2022 SAR-RARP50 challenge](https://arxiv.org/abs/2401.00496)
- [MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation](https://ieeexplore.ieee.org/document/10230819), **ISBI 2023, Oral.** Code [here](https://github.com/BCV-Uniandes/MATIS).

## Installation
Please follow these steps to run TAPIS:

```sh
$ conda create --name tapis python=3.8 -y
$ conda activate tapis
$ conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

$ conda install av -c conda-forge
$ pip install -U iopath
$ pip install -U opencv-python
$ pip install -U pycocotools
$ pip install 'git+https://github.com/facebookresearch/fvcore'
$ pip install 'git+https://github.com/facebookresearch/fairscale'
$ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

$ git clone https://github.com/BCV-Uniandes/GraSP
$ cd GraSP/TAPIS
$ pip install -r requirements.txt
```

## Data Preparation

In this [link](http://157.253.243.19/TAPIS/TAPIS.tar.gz), you will find a compressed archive with our preprocessed data files, region proposals, and pre-trained models. We provide a README file with instructions about the data structures and the files in the link. Download and uncompress these files with the following command.

```sh
$ wget http://157.253.243.19/TAPIS/TAPIS.tar.gz
$ tar -xzvf TAPIS.tar.gz
```
Then, locate the extracted files in a directory named GraSP inside the data directory of this repository. Please also include the video frames. In the end, the repository must have the following structure.

```tree
TAPIS
|
|__configs
|   ...
|__data
|   |__GraSP
|       |__annotations
|       |   |__fold1
|       |   |   |__train_anns.json
|       |   |   |__train_preds.json
|       |   |   |__train_long-term_anns.json
|       |   |   |__fold2_anns.json
|       |   |   |__fold2_preds.json
|       |   |   |__fold2_long-term_anns.json
|       |   |__fold2
|       |   |   ...
|       |   |__train
|       |       ...
|       |
|       |__features
|       |   |__fold1
|       |   |   |__train_region_features.pth
|       |   |   |__fold2_preds_region_features.pth
|       |   |__fold2
|       |   |   ...
|       |   |__train
|       |       ...
|       |
|       |__frame_lists
|       |   |__fold1.csv
|       |   |__fold2.csv
|       |   |__train.csv
|       |   |__test.csv
|       |
|       |__frames
|       |   |__CASE001
|       |   |   |__000000000.jpg
|       |   |   |__000000002.jpg
|       |   |   ...
|       |   |__CASE002
|       |   |   ...
|       |   ...
|       |
|       |__pretrained_models
|           |__ACTIONS
|           |   |__actions_m2f-swinl_fold1.pyth
|           |   |__actions_m2f-swinl_fold2.pyth
|           |   |__actions_m2f-swinl_train.pyth
|           |__INSTRUMENTS
|           |   ...
|           |__PHASES
|           |   ...
|           |__STEPS
|           |   ...
|           |__SEGMENTATION_BASELINE
|               ...
|__region_proposals
|__run_files
|__tapis
|__tools
```

Feel free to use soft/hard linking to other paths or to modify the directory structure, names, or locations of the files. However, you may also have to alter the .yaml config files or the bash running scripts. 

### Alternative Download Methods

If you cannot download the files from our server, you can find the compressed archive in this [Google Drive Link](https://drive.google.com/file/d/1Lf_OeSUzSRxyvcVSsH8_GkNLTYMmPnhV/view?usp=sharing).

## Running the code

| Task | cross-val mAP | test mAP | config | run file | model path |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Phases | 72.87 $\pm$ 1.66 | 74.06 | [PHASES](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/configs/GraSP/TAPIS_PHASES.yaml) | [phases](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/run_files/grasp_phases.sh) | *TAPIS/pretrained_models/PHASES* |
| Steps | 49.165 $\pm$ 0.004 | 49.45 | [STEPS](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/configs/GraSP/TAPIS_STEPS.yaml) | [steps](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/run_files/grasp_steps.sh) | *TAPIS/pretrained_models/STEPS* |
| Instruments | 90.28 $\pm$ 0.83 | 89.09 | [INSTRUMENTS](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/configs/GraSP/TAPIS_INSTRUMENTS.yaml) | [instruments](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/run_files/grasp_instruments.sh) | *TAPIS/pretrained_models/INSTRUMENTS* |
| Actions | 34.27 $\pm$ 1.76 | 39.50 | [ACTIONS](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/configs/GraSP/TAPIS_ACTIONS.yaml) | [actions](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/run_files/grasp_actions.sh) | *TAPIS/pretrained_models/ACTIONS* |

We provide bash scripts with the default parameters to evaluate each GraSP task. Please first download our preprocessed data files and pretrained models as instructed earlier and run the following commands to run evaluation on each task:

```sh
# Run the script corresponding to the desired task to evaluate
$ sh run_files/grasp_<actions/instruments/phases/steps>
```

### Training TAPIS

You can easily modify the bash scripts to train our models. Just set ```TRAIN.ENABLE True``` on the desired script to enable training, and set ```TEST.ENABLE False``` to avoid testing before training. You might also want to modify ```TRAIN.CHECKPOINT_FILE_PATH``` to the model weights you want to use as initialization. You can modify the [config files](https://github.com/BCV-Uniandes/GraSP/tree/main/TAPIS/configs/GraSP) or the [bash scripts](https://github.com/BCV-Uniandes/GraSP/tree/main/TAPIS/run_files) to alter the architecture design, training schedule, video input design, etc. We provide documentation for each hyperparameter in the [defaults script](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/tapis/config/defaults.py).

### Evaluation metrics

Although our codes are configured to evaluate the model's performance after each epoch, you can easily evaluate your model's predictions using our evaluation codes and implementations. For this purpose, you can run the [evaluate script](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/tapis/evaluate.py) and provide the required paths in the arguments as documented in the script. You can run this script on the output files of the [detectron2](https://github.com/facebookresearch/detectron2) library using the ```--filter``` argument, or you can provide your predictions in the following format:

```tree
[
      {"<frame/name>":
            
            {
             # For long-term tasks
             "<phase/step>_score_dist": [class_1_score, ..., class_N_score],

             # For short-term tasks
             "instances": 
             [
                 {
                  "bbox": [x_min, y_min, x_max, y_max],
                  "<instruments/actions>_score_dist": [class_1_score, ..., class_N_score],
                  
                  # For instrument segmentation
                  "segment" <Segmentation in RLE format>
                 } 
             ]
            }
      },
      ...
]
```

You can run the ```evaluate.py``` script as follows:

```sh
$ python evaluate.py --coco_anns_path /path/to/coco/annotations/json \
--pred-path /path/to/predictions/json or pth \
--output_path /path/to/output/directory \
--tasks <instruments/actions/phases/steps> \
--metrics <mAP/mAP@0.5IoU_box/mAP@0.5IoU_segm/mIoU> \
(optional) --masks-path /path/to/segmentation/masks \
# Optional for detectron2 outputs
--filter \
--slection <topk/thresh/cls_thresh/...> \
--selection_info <filtering info> 
```

## Instrument Segmentation Baseline

Our instrument segmentation baseline is wholly based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), so we recommend checking their repo for details on their implementation. 

### Installation

To run our baseline, first go to the region proposal directory and install the corresponding dependencies. You must have already installed all the required dependencies of the main TAPIS code. The following is an example of how to install dependencies correctly.

```sh
$ conda activate tapis
$ cd ./region_proposals
$ pip install -r requirements.txt
$ cd mask2former/modeling/pixel_decoder/ops
$ sh make.sh
$ cd ../../../..
```
### Running the Segmentation Baseline

The original Mask2Former code does not accept segmentation annotations in RLE format; hence, to run our baseline, you must first transform our RLE masks into Polygons using the [rle_to_polygon.py](https://github.com/BCV-Uniandes/GraSP/tree/main/TAPIS/region_proposals/rle_to_polygon.py) script as follows:

```sh
$ python rle_to_polygon.py --data_path /path/to/GraSP/annotations
```

Then to run the training code run the [train_net.py](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/region_proposals/train_net.py) script indicating the path to a configuration file in the [configs directory](https://github.com/BCV-Uniandes/GraSP/tree/main/TAPIS/region_proposals/configs/grasp) with the ```--config-file``` argument. You should also indicate the path to the GraSP dataset with the ```DATASETS.DATA_PATH``` option, the path to the pretrained weights with the ```MODEL.WEIGHTS``` option, and the desired output path with the ```OUTPUT_DIR``` option. Download the pretrained Mask2Former weights for instance segmentation in the COCO dataset from the [Mask2Former](https://github.com/facebookresearch/Mask2Former) repo. Use the following command to train our baseline:

```sh
$ python train_net.py --num-gpus <number of GPUs> \
--config-file configs/grasp/<config file name>.yaml \
DATASETS.DATA_PATH path/to/grasp/dataset \
MODEL.WEIGHTS path/to/pretrained/model/weights \
OUTPUT_DIR output/path
```

You can modify most hyperparameters by changing the values in the configuration files or using command options; please check the [Detectron2](https://github.com/facebookresearch/detectron2) library and the original [Mask2Former](https://github.com/facebookresearch/Mask2Former) repo for further details on configuration files and options.<br/>

To run the evaluation code, use the ```--eval-only``` argument and the TAPIS model weights provided in the [data link](http://157.253.243.19/TAPIS). Run the following command to evaluate our baseline:

```sh
$ python train_net.py --num-gpus <number of GPUs> --eval-only \
--config-file configs/grasp/<config file name>.yaml \
DATASETS.DATA_PATH path/to/grasp/dataset \
MODEL.WEIGHTS path/to/pretrained/model/weights \
OUTPUT_DIR output/path
```

**Note:** You can easily **run our segmentation baseline in a custom dataset** by modifying the ```register_surgical_dataset``` function in the [train_net.py](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/region_proposals/train_net.py) script to register the dataset in a COCO JSON format. Once again, we recommend checking the [Detectron2](https://github.com/facebookresearch/detectron2) library and the original [Mask2Former](https://github.com/facebookresearch/Mask2Former) for more details on registering your dataset.

### Region Features

Our code stores the region features corresponding to the predicted segments in the same results files in the output directory. However, you can use the [match_annots_n_preds.py](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/data/match_annots_n_preds.py) script to filter predictions, assign region features to ground truth instances for training, and parse predictions into necessary files for TAPIS. Use the code as follows:

```sh
$ python match_annots_n_preds.py 
```

### MATIS Baseline for Endovis 2017 and 2018

You can also run the segmentation baseline for the [Endovis 2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/) and [Endovis 2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/) datasets, as done in our previous [MATIS](https://arxiv.org/abs/2303.09514) paper. We recommend checking the paper and the [MATIS repo](https://github.com/BCV-Uniandes/MATIS). <br/>

To run our segmentation baseline in the Endovis 2017 and 2018 datasets, please download the preprocessed frames, instances annotations, and pretrained models from this [link](http://157.253.243.19/MATIS/) as instructed in the [MATIS repo](https://github.com/BCV-Uniandes/MATIS). Then run the segmentation baseline as previously instructed but using the provided configuration files for [Endovis 2017](https://github.com/BCV-Uniandes/GraSP/tree/main/TAPIS/region_proposals/configs/endovis_2017) or [Endovis 2018](https://github.com/BCV-Uniandes/GraSP/tree/main/TAPIS/region_proposals/configs/endovis_2018), and indicating the path to the downloaded data with the ```DATASETS.DATA_PATH``` option

## Contact

If you have any doubts, questions, issues, corrections, or comments, please email n.ayobi@uniandes.edu.co.

## Citing TAPIS 

If you find GraSP or TAPIS useful for your research (or its previous versions, PSI-AVA, TAPIR, and MATIS), please include the following BibTex citations in your papers.

```BibTeX
@article{ayobi2024pixelwise,
      title={Pixel-Wise Recognition for Holistic Surgical Scene Understanding}, 
      author={Nicol{\'a}s Ayobi and Santiago Rodr{\'i}guez and Alejandra P{\'e}rez and Isabela Hern{\'a}ndez and Nicol{\'a}s Aparicio and Eug{\'e}nie Dessevres and Sebasti{\'a}n Peña and Jessica Santander and Juan Ignacio Caicedo and Nicol{\'a}s Fern{\'a}ndez and Pablo Arbel{\'a}ez},
      year={2024},
      url={https://arxiv.org/abs/2401.11174},
      eprint={2401.11174},
      journal={arXiv},
      primaryClass={cs.CV}
}

@InProceedings{ayobi2023matis,
      author={Nicol{\'a}s Ayobi and Alejandra P{\'e}rez-Rond{\'o}n and Santiago Rodr{\'i}guez and Pablo Arbel{\'a}es},
      booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)}, 
      title={MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation}, 
      year={2023},
      pages={1-5},
      doi={10.1109/ISBI53787.2023.10230819}
}

@InProceedings{valderrama2020tapir,
      author={Natalia Valderrama and Paola Ruiz and Isabela Hern{\'a}ndez and Nicol{\'a}s Ayobi and Mathilde Verlyck and Jessica Santander and Juan Caicedo and Nicol{\'a}s Fern{\'a}ndez and Pablo Arbel{\'a}ez},
      title={Towards Holistic Surgical Scene Understanding},
      booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
      year={2022},
      publisher={Springer Nature Switzerland},
      address={Cham},
      pages={442--452},
      isbn={978-3-031-16449-1}
}
```
