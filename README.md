# Hover-Trans: Anatomy-aware HoVer-Transformer for ROI-free Breast Cancer Diagnosis in Ultrasound Images
![network](https://github.com/yuhaomo/HoVerTrans/blob/main/network.png)
## Introduction
The implementation of: <br>
[**Hover-Trans: Anatomy-aware HoVer-Transformer for ROI-free Breast Cancer Diagnosis in Ultrasound Images**](https://arxiv.org/abs/2205.08390)
## Requirements
- Pytorch
## Setup
### Installation
Clone the repo and install required packages:
```
git clone https://github.com/yuhaomo/HoVerTrans.git
cd HoVerTrans
pip install -r requirements.txt
```
### Dataset
-  You can download our dataset ([GDPH&SYSUCC](https://1drv.ms/u/s!AgOtqK2ZncKlgoxsmt-UYbEwMyZY2g?e=INNhyK)) and unpack them into the ./data folder.
```
./data
└─GDPH&SYSUCC
      ├─label.csv
      └─img
          ├─benign(0).png
          ├─benign(1).png
          ├─benign(2).png
          ├─malignant(0).png
          ├─malignant(1).png
          ...
```
- The format of the label.csv is as follows:
```
| name             | label |
| benign(0).png    |   0   |
| benign(1).png    |   0   |
| benign(2).png    |   0   |
| malignant(0).png |   1   |
| malignant(1).png |   1   |
...
```
### Training
```
python train.py --data_path ./data/GDPH&SYSUCC/img --csv_path ./data/GDPH&SYSUCC/label.csv -s --batch_size 32 --class_num 2 --epochs 250 --lr 0.0001 
```
## Citation
If you find this repository useful or use this dataset, please consider citing our work:
```
@misc{https://doi.org/10.48550/arxiv.2205.08390,
  doi = {10.48550/ARXIV.2205.08390},
  url = {https://arxiv.org/abs/2205.08390},
  author = {Mo, Yuhao and Han, Chu and Liu, Yu and Liu, Min and Shi, Zhenwei and Lin, Jiatai and Zhao, Bingchao and Huang, Chunwang and Qiu, Bingjiang and Cui, Yanfen and Wu, Lei and Pan, Xipeng and Xu, Zeyan and Huang, Xiaomei and Liu, Zaiyi and Wang, Ying and Liang, Changhong},
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {HoVer-Trans: Anatomy-aware HoVer-Transformer for ROI-free Breast Cancer Diagnosis in Ultrasound Images},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
