# Pixel-wise Recognition for Holistic Surgical Scene Understanding

[Nicolás Ayobi](https://nayobi.github.io/)<sup>1,2</sup>, Santiago Rodríguez<sup>1,2*</sup>, Alejandra Pérez<sup>1,2*</sup>, Isabela Hernández<sup>1,2*</sup>, Nicolás Aparicio<sup>1,2</sup>, Eugénie Dessevres<sup>1,2</sup>, Sebastián Peña<sup>3</sup>, Jessica Santander<sup>3</sup>, Juan Ignacio Caicedo<sup>3</sup>, Nicolás Fernández<sup>4,5</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1,2</sup> <br/>
<br/>
<font size="1"><sup>*</sup>Equal contribution.</font><br/>
<font size="1"><sup>1 </sup> Center  for  Research  and  Formation  in  Artificial  Intelligence ([CinfonIA](https://cinfonia.uniandes.edu.co/)), Bogotá, Colombia.</font><br/>
<font size="1"><sup>2 </sup> Universidad  de  los  Andes,  Bogotá, Colombia.</font><br/>
<font size="1"><sup>3 </sup> Fundación Santafé de Bogotá, Bogotá, Colombia</font><br/>
<font size="1"><sup>4 </sup> Seattle Children’s Hospital, Seattle, USA</font><br/>
<font size="1"><sup>5 </sup> University of Washington, Seattle, USA</font><br/>

- Preprint available at [**arXiv**](https://arxiv.org/abs/2401.11174)<br/>
- Visit the project on our [**website**](https://cinfonia.uniandes.edu.co/publications/pixel-wise-recognition-for-holistic-surgical-scene-understanding/).

## Abstract

<div align="center">
  <img src="Images/dataset.jpg"/>
</div><br/>

We present the Holistic and Multi-Granular Surgical Scene Understanding of Prostatectomies (GraSP) dataset, a curated benchmark that models surgical scene understanding as a hierarchy of complementary tasks with varying levels of granularity. Our approach enables a multi-level comprehension of surgical activities, encompassing long-term tasks such as surgical phases and steps recognition and short-term tasks including surgical instrument segmentation and atomic visual actions detection. To exploit our proposed benchmark, we introduce the Transformers for Actions, Phases, Steps, and Instrument Segmentation (TAPIS) model, a general architecture that combines a global video feature extractor with localized region proposals from an instrument segmentation model to tackle the multi-granularity of our benchmark. Through extensive experimentation, we demonstrate the impact of including segmentation annotations in short-term recognition tasks, highlight the varying granularity requirements of each task, and establish TAPIS's superiority over previously proposed baselines and conventional CNN-based models. Additionally, we validate the robustness of our method across multiple public benchmarks, confirming the reliability and applicability of our dataset. This work represents a significant step forward in Endoscopic Vision, offering a novel and comprehensive framework for future research towards a holistic understanding of surgical procedures.

This repository provides instructions for downloading the [**GraSP** dataset](https://github.com/BCV-Uniandes/GraSP?tab=readme-ov-file#grasp) and running the PyTorch implementation of [**TAPIS**](https://github.com/BCV-Uniandes/GraSP/tree/main/TAPIS), both presented in the paper Pixel-Wise Recognition for Holistic Surgical Scene Understanding.

## Previous works

This work is an extended and consolidated version of three previous works:

- [Towards Holistic Surgical Scene Understanding](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_42), **MICCAI 2022, Oral.** Code [here](https://github.com/BCV-Uniandes/TAPIR).
- **Winner solution** of the [2022 SAR-RARP50 challenge](https://arxiv.org/abs/2401.00496)
- [MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation](https://ieeexplore.ieee.org/document/10230819), **ISBI 2023, Oral.** Code [here](https://github.com/BCV-Uniandes/MATIS).

Please check these works!

## GraSP

In this [link](http://157.253.243.19/GraSP), you will find all the files that compose the entire Holistic and Multi-Granular Surgical Scene Understanding of Prostatectomies (GraSP) dataset. These files include the original Radical Prostatectomy videos, our sampled preprocessed and raw frames, and the gathered annotations for all four semantic tasks. The data in the link has the following organization:

```tree
GraSP:
|
|__GraSP_30fps.tar.gz
|__GraSP_1fps.tar.hz
|__raw_frames_30fps.tar.gz
|__raw_frames_1fps.tar.gz
|__videos.tar.gz
|__1fps_to_30fps_association.json
|__README.txt
```

These files contain the following aspects and versions of our dataset: 

1) ```GraSP_30fps.tar.gz``` Is a compressed archive with all the preprocessed frames sampled at 30fps and all annotations for all tasks in our benchmark. **This is the complete dataset used for model training and evaluation.**
2) ```GraSP_1fps.tar.gz``` Is a compressed archive with a lighter version of the dataset with preprocessed frames sampled at 1fps and the annotations for these frames. 
3) ```raw_frames_30fps.tar.gz``` Is a compressed archive with all original frames sampled at 30 fps before frame preprocessing.
4) ```raw_frames_1fps.tar.gz``` Is a compressed archive with original frames sampled at 1 fps before frame preprocessing.
5) ```videos.tar.gz``` Is a compressed archive with our dataset's original raw Radical Prostatctomy videos.
6) ```1fps_to_30fps_association.json``` Contains the name association between frames sampled at 1fps and frames sampled at 30fps.
7) ```README.txt``` Information file with a summary of files' contents.

You can download all files recursively by running the following command in a Linux terminal:

```sh
$ wget -r http://157.253.243.19/GraSP
```

However, **we recommend downloading just the files you require** by running:

```sh
$ wget http://157.253.243.19/GraSP/<file_name>
```

For example, you can only download the video files by running:

```sh
$ wget http://157.253.243.19/GraSP/videos.tar.gz
```

**Note:** All compressed archives **contain a README file** with further details and instructions on the data's structure and format. 

### Alternative Download Methods

In case you cannot download the files from our servers, you can download the **lighter version of our dataset (the version sampled at 1fps)** from this [**Google Drive link**](https://drive.google.com/file/d/1vfRHtNj0ZWvSascgJbEPkQX4uayRvs0I/view?usp=sharing), and the [1fps to 30fps association file](https://drive.google.com/file/d/1E4_W7DWS84G9ywo5GCHb7tets_2pe9Cz/view?usp=sharing). This alternative downloading option only includes the 1fps version of the dataset for now, but we are currently working to upload all the files to the cloud.

### Main Dataset to Run our Models

The ```GraSP_30fps.tar.gz``` is the only archive **necessary to run our code**. Hence, to download and uncompress this file, run the following command in a Linux terminal:

```sh
$ wget http://157.253.243.19/GraSP/GraSP_30fps.tar.gz
$ tar -xzvf GraSP_30fps.tar.gz
```

After decompressing the archive, the GraSP_30fps directory must have the following structure:

```tree
GraSP_30fps
|
|___frames
|    |
|    |___CASE001
|    |    |__000000000.jpg
|    |    |__000000001.jpg
|    |    |__000000002.jpg
|    |    ...
|    |___CASE002
|    |    ...
|    ...
|    |
|    |___README.txt
|
|___annotations
     |__segmentations
     |       |
     |       |__CASE001
     |       |      |__000000068.png
     |       |      |__000001642.png
     |       |      |__000003218.png
     |       |      ...
     |       ...
     |       |__CASE053
     |              |__000000015.png
     |              |__000001065.png
     |              |__000002115.png
     |              ...
     |
     |__grasp_long-term_fold1.json
     |__grasp_long-term_fold2.json
     |__grasp_long-term_train.json
     |__grasp_long-term_test.json
     |__grasp_short-term_fold1.json
     |__grasp_short-term_fold2.json
     |__grasp_short-term_train.json
     |__grasp_short-term_test.json
     |__README.txt
```


## [TAPIS](./TAPIS/)

Go to the [TAPIS directory](./TAPIS/) to find our source codes and instructions for running our TAPIS model.

## Contact

If you have any doubts, questions, issues, corrections, or comments, please email n.ayobi@uniandes.edu.co.

## Citing GraSP

If you find GraSP or TAPIS useful for your research (or its previous versions, PSI-AVA, TAPIR, and MATIS), please include the following BibTex citations in your papers.

```BibTeX
@article{ayobi2024pixelwise,
      title={Pixel-Wise Recognition for Holistic Surgical Scene Understanding}, 
      author={Nicol{\'a}s Ayobi and Santiago Rodr{\'i}guez and Alejandra P{\'e}rez and Isabela Hern{\'a}ndez and Nicol{\'a}s Aparicio and Eug{\'e}nie Dessevres and Sebasti{\'a}n Peña and Jessica Santander and Juan Ignacio Caicedo and Nicol{\'a}s Fernández and Pablo Arbel{\'a}ez},
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