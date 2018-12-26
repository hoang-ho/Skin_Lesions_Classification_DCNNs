# Skin Lesions Classification with Deep Convolutional Neural Network

This is a 40-hour project for CIS 5526 Machine Learning. For full description and analysis please refer to Project_Report.pdf

## Files Description

* Final report: Project_Report.pdf

* Exploratory data analysis: Skin_Cancer_EDA.ipynb

* Baseline model: Baseline_CNN.ipynb

* Fine-tuning the last convolutional block of VGG16: Fine_Tuning_VGG16.ipynb

* Fine-tuning the top 2 inception blocks of InceptionV3: Fine_Tuning_InceptionV3.ipynb

* Fine-tuning the Inception-ResNet-C of Inception-ResNet V2: Fine_Tuning_InceptionResNet.ipynb

* Fine-tuning the last dense block of DenseNet 201: Fine_Tuning_DenseNet.ipynb

* Fine-tuning all layers of pretrained Inception V3 on ImageNet: Retraining_InceptionV3.ipynb

* Fine-tuning all layers of pretrained DenseNet 201 on ImageNet: Retraining_DenseNet.ipynb

* Ensemble model of the fully fine-tuned Inception V3 and DenseNet 201 (best result): Ensemble_Models.ipynb 

## Results

| Models        | Validation           | Test            |  Depth          | # Params          |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:| 
|   Baseline   | 77.48% |76.54% | 11 layers | 2,124,839 |
|  Fine-tuned VGG16 (from last block)    |  79.82%      |   79.64%  | 23 layers | 14,980,935 |
|  Fine-tuned Inception V3 (from the last 2 inception blocks) |  79.935%   |  79.94% | 315 layers | 22,855,463 |
|  Fine-tuned Inception-ResNet V2 (from the Inception-ResNet-C) | 80.82% | 82.53% | 784 layers | 55,127,271 |
|  Fine-tuned DenseNet 201 (from the last dense block) | **85.8%** | **83.9%**  |  711 layers | 19,309,127 |
|  Fine-tuned Inception V3 (all layers) | 86.92% | 86.826% | | |
|  Fine-tuned DenseNet 201 (all layers)  | **86.696%** | **87.725%** | | |
|  Ensemble of fully-fine-tuned Inception V3 and DenseNet 201 | **88.8%** | **88.52%** | | |


## The Dataset

[The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T,)
