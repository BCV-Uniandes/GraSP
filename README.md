# Pixel-wise Recognition for Holistic Surgical Scene Understanding

We present the Holistic and Multi-Granular Surgical Scene Understanding of Prostatectomies (GraSP) dataset, a curated benchmark that models surgical scene understanding as a hierarchy of complementary tasks with varying levels of granularity. Our approach enables a multi-level comprehension of surgical activities, encompassing long-term tasks such as surgical phases and steps recognition and short-term tasks including surgical instrument segmentation and atomic visual actions detection. To exploit our proposed benchmark, we introduce the Transformers for Actions, Phases, Steps, and Instrument Segmentation (TAPIS) model, a general architecture that combines a global video feature extractor with localized region proposals from an instrument segmentation model to tackle the multi-granularity of our benchmark. Through extensive experimentation, we demonstrate the impact of including segmentation annotations in short-term recognition tasks, highlight the varying granularity requirements of each task, and establish TAPIS's superiority over previously proposed baselines and conventional CNN-based models. Additionally, we validate the robustness of our method across multiple public benchmarks, confirming the reliability and applicability of our dataset. This work represents a significant step forward in Endoscopic Vision, offering a novel and comprehensive framework for future research towards a holistic understanding of surgical procedures.

This repository provides instructions to download the GraSP dataset and run the PyTorch implementation of TAPIS, both presented in the paper Pixel-Wise Recognition for Holistic Surgical Scene Understanding.

## Paper

[Pixel-wise Recognition for Holistic Surgical Scene Understanding](https://arxiv.org/abs/2401.11174) <br/>
[Nicolás Ayobi](https://nayobi.github.io/)<sup>1</sup>, Santiago Rodríguez<sup>1*</sup>, Alejandra Pérez<sup>1*</sup>, Isabela Hernández<sup>1*</sup>, Nicolás Aparicio<sup>1</sup>, Eugégia Dessevres<sup>1</sup>, Sebastián Peña<sup>2</sup>, Jessica Santander<sup>2</sup>, Juan Ignacio Caicedo<sup>2</sup>, Nicolás Fernández<sup>3,4</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup> <br/>
<sup>*</sup>Equal contribution.<br/>
<sup>1 </sup> Center  for  Research  and  Formation  in  Artificial  Intelligence .([CINFONIA](https://cinfonia.uniandes.edu.co/)),  Universidad  de  los  Andes,  Bogotá 111711, Colombia. <br/>
<sup>2 </sup> Fundación Santafé de Bogotá, Bogotá, Colombia<br/>
<sup>3 </sup> Seattle Children’s Hospital, Seattle, USA <br/>
<sup>4 </sup> University of Washington, Seattle, USA <br/>

**Preprint available at [ArXiv](https://arxiv.org/abs/2401.11174) "Pixel-wise Recognition for Holistic Surgical Scene Understanding" with code 2401.11174**<br/>

Visit the project in our [website](https://cinfonia.uniandes.edu.co/publications/pixel-wise-recognition-for-holistic-surgical-scene-understanding/).

### Previous works

This work is an extended and consolidated version of three previous works:

- [Towards Holistic Surgical Scene Understanding](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_42), **MICCAI 2022, Oral.** Code [here](https://github.com/BCV-Uniandes/TAPIR).
- **Winner solution** of the [2022 SAR-RARP50 challenge](https://arxiv.org/abs/2401.00496)
- [MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation](https://ieeexplore.ieee.org/document/10230819), **ISBI 2023, Oral.** Code [here](https://github.com/BCV-Uniandes/MATIS).

Please check these works.

## GraSP

<div align="center">
  <img src="Images/dataset.jpg" width="70%" height="70%"/>
</div><br/>

In this [link](http://157.253.243.19/PSI-AVA/GraSP), you will find the sampled frames of the original Radical Prostatectomy videos and the annotations that compose the Holistic and Multi-Granular Surgical Scene Understanding of Prostatectomies (GraSP) dataset. The data in the link has the following organization:

```tree
GraSP:
|
|--GraSP_1fps
|         |---frames
|         |    |---CASE001
|         |    |    |--00000.jpg
|         |    |    |--00001.jpg
|         |    |    |--00002.jpg
|         |    |    ...
|         |    |---CASE002
|         |    |    ...
|         |    ...
|         |
|         |---original_frames
|         |    |---CASE001
|         |    |    |--00000.jpg
|         |    |    |--00001.jpg
|         |    |    |--00002.jpg
|         |    |    ...
|         |    |---CASE002
|         |    |    ...
|         |    ...
|         |
|         |---annotations
|              |--segmentations
|              |    |---CASE001
|              |    |    |--00000.png
|              |    |    |--00001.png
|              |    |    |--00002.png
|              |    |    ...
|              |    |---CASE002
|              |    |    ...
|              |    ...
|              |    
|              |--grasp_dense_long-term_fold1.json
|              |--grasp_dense_long-term_fold2.json
|              |--grasp_dense_long-term_train.json
|              |--grasp_dense_long-term_test.json
|              |--grasp_dense_short-term_fold1.json
|              |--grasp_dense_short-term_fold2.json
|              |--grasp_dense_short-term_train.json
|              |--grasp_dense_short-term_test.json
|
|--GraSP_30fps
|         |---frames
|         |    |---CASE001
|         |    |    |--000000000.jpg
|         |    |    |--000000001.jpg
|         |    |    |--000000002.jpg
|         |    |    ...
|         |    |---CASE002
|         |    |    ...
|         |    ...
|         |
|         |---annotations
|              |--grasp_dense_short-term_fold1.json
|              |--grasp_dense_short-term_fold2.json
|              |--grasp_dense_short-term_train.json
|              |--grasp_dense_short-term_test.json
|
|--1fps_to_30fps_association.json
|--README.txt
```

In the [GraSP_1fps directory](http://157.253.243.19/PSI-AVA/GraSP/GraSP_1fps), you will find our video frames sampled at 1fps and all the annotations for long-term tasks (surgical phases and steps recognition) and short-term tasks (instrument segmentation and atomic action detection). Additionally, in the [GraSP_30fps directory](http://157.253.243.19/PSI-AVA/GraSP/GraSP_30fps), you will find all the frames of our videos sampled at 30fps and their corresponding annotations for short-term tasks. We provide a README file in the link, briefly explaining our annotations. We recommend downloading the data recursively with the following command:

```sh
$ wget -r http://157.253.243.19/PSI-AVA/GraSP
```

If you only need the dataset with the video frames sampled at 1fps or 30fps, you can download the directory of the version that you need:


```sh
# 1fps version
$ wget -r http://157.253.243.19/PSI-AVA/GraSP/GraSP_1fps

# 30fps version
$ wget -r http://157.253.243.19/PSI-AVA/GraSP/GraSP_30fps
```

## [TAPIS](./TAPIS/)

Go to the [TAPIS directory](./TAPIS/) to find our source codes and instructions to run the code.

## Citing GraSP

If you find GraSP or TAPIS useful for your research, please include the following BibTex citations in your papers.

```BibTeX
@misc{ayobi2024pixelwise,
      title={Pixel-Wise Recognition for Holistic Surgical Scene Understanding}, 
      author={Nicol{\'a}s Ayobi and Santiago Rodr{\'i}guez and Alejandra P{\'e}rez and Isabela Hern{\'a}ndez and Nicol{\'a}s Aparicio and Eug{\'e}nie Dessevres and Sebasti{\'a}n Peña and Jessica Santander and Juan Ignacio Caicedo and Nicol{\'a}s Fern{\'a}ndez and Pablo Arbel{\'a}ez},
      year={2024},
      eprint={2401.11174},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{valderrama2020tapir,
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