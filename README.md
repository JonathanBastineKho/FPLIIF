# FPLIIF

## 1. Download data

First, run the download script to get the dataset:

```bash
wget https://entuedu-my.sharepoint.com/:u:/g/personal/jonathan017_e_ntu_edu_sg/IQBt_TbAeGBGTIDtRK2iDzQ0ATeBSEUoJimLsa8Hts6KC9c?e=67tHkv
```

## 2. Install dependencies

Install the package in editable mode and the required dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

## 3. Run training

Train the model:

```bash
python train.py \
  --data_dir data \
  --dataset_name CelebAMaskHQ
```