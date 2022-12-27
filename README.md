# The Project
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1br0nJcuTF2YJCvbvnu0Etsm4S7mLk5Sc?usp=sharing)

This project is UE



## Prerequisites

In order to use this project, you will need the following:

- Python 3.9 or higher
- pytorch-lightning 

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
│   ├── train
│   └── gt
├── Test
│   ├── test
│   └── gt
```
To train the model, run the following command:
```
python train.py
```


## Additional Resources

For more information on how the model works, see the following paper:

- "...." (xxxx, 2022)

## Contact Information

If you have any questions or encounter any issues, please don't hesitate to contact me at [uyo9ko@gmail.com].
