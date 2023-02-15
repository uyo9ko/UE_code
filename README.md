# The Project
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1br0nJcuTF2YJCvbvnu0Etsm4S7mLk5Sc?usp=sharing)

This project is UE



## Prerequisites

In order to use this project, you will need the following:

- Python 3.9 or higher

## Installation

To install the required packages, run the following command:
```
pip install -r requirements.txt
```


## Test
Put your test image in folder `samples`, then run this colab noteboook 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1br0nJcuTF2YJCvbvnu0Etsm4S7mLk5Sc?usp=sharing)

Pretrained model weights on UIEB dataset can be download here : https://drive.google.com/file/d/1E7edgJ83wuLrTRMO7HWd1zL_Z-e2_Zsq/view?usp=share_link 



## Train
The train dataset should be organized as follows:
```
Data
├── Train
│   ├── raw
│   └── reference
├── Test
│   ├── raw
│   └── reference
```
To train the model, run the following command:
```
python train.py --data_set uieb --data_path your_path --batch_size 8 --image_size 256
```


