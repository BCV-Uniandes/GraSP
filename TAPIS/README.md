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

In this [link](http://157.253.243.19/TAPIS), you will find our preprocessed data files, region proposals, and pre-trained models. We provide a README file with instructions about the data structures and the files in the link. Download these files and locate them in a directory named GraSP in the data directory of this repository. Please also include the video frames. In the end, the repository must have the following structure.

```tree
TAPIS
|
|--configs
|   ...
|--data
|   |--GraSP
|       |--annotations
|       |   |--fold1
|       |   |   |--train_anns.json
|       |   |   |--train_preds.json
|       |   |   |--train_long-term_anns.json
|       |   |   |--fold2_anns.json
|       |   |   |--fold2_preds.json
|       |   |   |--fold2_long-term_anns.json
|       |   |--fold2
|       |   |   ...
|       |   |--train
|       |       ...
|       |
|       |--features
|       |   |--fold1
|       |   |   |--train_region_features.pth
|       |   |   |--fold2_preds_region_features.pth
|       |   |--fold2
|       |   |   ...
|       |   |--train
|       |       ...
|       |
|       |--frame_lists
|       |   |--fold1.csv
|       |   |--fold2.csv
|       |   |--train.csv
|       |   |--test.csv
|       |
|       |--frames
|       |   |--CASE001
|       |   |   |--000000000.jpg
|       |   |   |--000000002.jpg
|       |   |   ...
|       |   |--CASE002
|       |   |   ...
|       |   ...
|       |
|       |--pretrained_models
|           |--ACTIONS
|           |   |--actions_m2f-swinl_fold1.pyth
|           |   |--actions_m2f-swinl_fold2.pyth
|           |   |--actions_m2f-swinl_train.pyth
|           |--INSTRUMENTS
|           |   ...
|           |--PHASES
|           |   ...
|           |--STEPS
|           |   ...
|           |--SEGMENTATION_BASELINE
|               ...
|--region_proposals
|--run_files
|--tapis
|--tools
```

Feel free to use soft/hard linking to other paths or to modify the directory structure, the names, or the locations of the files, but then you may also have to modify the .yaml config files or the bash running scripts. 

## Running the code

| Task | cross-val mAP | test mAP | config | run file | model |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Phases | 72.87 $\pm$ 1.66 | 74.06 | [PHASES](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/configs/GraSP/TAPIS_PHASES.yaml) | [phases](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/run_files/grasp_phases.sh) | [phases](http://157.253.243.19/TAPIS/pretrained_models/PHASES/) |
| Steps | 49.165 $\pm$ 0.004 | 49.45 | [STEPS](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/configs/GraSP/TAPIS_STEPS.yaml) | [steps](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/run_files/grasp_steps.sh) | [steps](http://157.253.243.19/TAPIS/pretrained_models/STEPS/) |
| Instruments | 90.28 $\pm$ 0.83 | 89.09 | [INSTRUMENTS](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/configs/GraSP/TAPIS_INSTRUMENTS.yaml) | [instruments](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/run_files/grasp_instruments.sh) | [instruments](http://157.253.243.19/TAPIS/pretrained_models/INSTRUMENTS/) |
| Actions | 34.27 $\pm$ 1.76 | 39.50 | [ACTIONS](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/configs/GraSP/TAPIS_ACTIONS.yaml) | [actions](https://github.com/BCV-Uniandes/GraSP/blob/main/TAPIS/run_files/grasp_actions.sh) | [actions](http://157.253.243.19/TAPIS/pretrained_models/ACTIONS/) |

We provide bash scripts with the default parameters to evaluate each GraSP task. Please first download our preprocessed data files and pretrained models as instructed earlier and run the following commands to run evaluation on each task:

```sh
# Run the script corresponding to the desired task to evaluate
$ sh run_files/grasp_{actions/instruments/phases/steps}
```

### Training TAPIS

You can easily modify the bash scripts to train our models. Just set ```TRAIN.ENABLE True``` on the desired script to enable training, and set ```TEST.ENABLE False``` to avoid testing before training. You might also want to modify ```TRAIN.CHECKPOINT_FILE_PATH``` to the model weights you want to use as initialization. You can modify the [config files]() or the [bash scripts]() to modify the architecture design, training schedule, video input design, etc. We provide documentation for each hyperparameter in the [defaults script]().

### Evaluation metrics

Although our codes are configured to evaluate the model's performance after each epoch, you can also manually evaluate your model's predictions using our evaluation codes and implementations. For this purpose, you can run the [evaluate script]() and provide the required paths in the arguments as documented in the script. 

## Instrument Segmentation Baseline

Our instrument segmentation baseline is completely based on [Mask2Former](), so we recommend checking their repo for details on their implementation. 

### Installation

To run our baseline, first go to the region proposal directory and install the corresponding dependencies. You must have already installed all the required dependencies of the main TAPIS code. The following is an example of how to install dependencies correctly.

```sh
$ conda activate tapis
$ cd ./region_proposals
$ pip install -r requirements.txt
$ cd mask2former/modeling/pixel_decoder/ops
$ sh make.sh
```
### Running the Segmentation Baseline

Coming soon!

## Running MATIS or TAPIS

Coming soon!

## Contact

If you have any doubts, questions, issues, corrections, or comments, please email n.ayobi@uniandes.edu.co.

## Citing TAPIS 

If you find GraSP or TAPIS useful for your research, please include the following BibTex citations in your papers.

```BibTeX
@article{ayobi2024pixelwise,
      title={Pixel-Wise Recognition for Holistic Surgical Scene Understanding}, 
      author={Nicolás Ayobi and Santiago Rodríguez and Alejandra Pérez and Isabela Hernández and Nicolás Aparicio and Eugénie Dessevres and Sebastián Peña and Jessica Santander and Juan Ignacio Caicedo and Nicolás Fernández and Pablo Arbeláez},
      year={2024},
      url={https://arxiv.org/abs/2401.11174},
      eprint={2401.11174},
      journal={arXiv},
      primaryClass={cs.CV}
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