# ILPose  
Incremental Object 6D Pose Estimation

[Project Page](https://qm-ipalab.github.io/ILPose/) 


## Overview
This repository is the implementation code of the paper "Incremental Object 6D Pose Estimation".
In this repo, we provide our full implementation code of training/evaluation/inference of the ILPose model.

## Requirements

* Python 2.7/3.5/3.6
* [PyTorch 0.4.1](https://pytorch.org/) 
* PIL
* scipy
* numpy
* logging
* CUDA 7.5/8.0/9.0

## Dataset

This work is evaluated on two datasets:
* [Linemod](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7)
* [YCB-Video](https://rse-lab.cs.washington.edu/projects/posecnn/)
Download preprocessed LineMOD dataset and YCB Video dataset.


## Keyframe selection
Please go to /selection/ folder and run:
```bash
python keyframe_selection.py
```

## Training
Please go to /training/ folder and run:
```bash
python train.py 
```
The model is sequentially trained on different objects, and the trained model is saved at /trained_models/ folder.




**Checkpoints and Resuming**: 
After the training of each epoch, a `model_current.pth` checkpoint will be saved. 
You can use it to resume the training. 
We test the model after each epoch and save the model has the best performance so far, as `model_(epoch)_(best_score)_.pth`, which can be used for the evaluation.

## Evaluation
Please go to /eval/ folder and run:
```bash
python eval.py
```




