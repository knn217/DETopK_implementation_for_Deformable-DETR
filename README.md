# DETopK

By Nhan Duy Quan; Dang Nguyen Chau

This repository is a fork of Deformable DETR to implement training method from the paper [DETopK: training for DETR based models with Top K predictions noise]

## Introduction

**TL; DR.** DETR models introduced the concept of object queries, this research aims to study the concept to improve DETR based models.

## Training

Setting up and using the model is not different from Deformable DETR.

The dataset should also be organized as instructed in Deformable DETR:
```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```


But training use a different command:

* Training from scratch: 
```bash
python main.py --top_K_matching --set_top_K=5 --top_K_matching_method='closest' --top_K_scale=0.5 --num_feature_levels=4 --batch_size 1 --lr 6.25e-6 --coco_path data/coco --output_dir output
```
* Continue from a previous training session
```bash
python main.py --top_K_matching --set_top_K=5 --top_K_matching_method='closest' --top_K_scale=0.5 --num_feature_levels=4 --batch_size 1 --lr 6.25e-6 --coco_path data/coco --output_dir output --resume output/checkpoint.pth
```

The paper actually trains with a dataset clled miniCOCO2017 which you can download from: https://github.com/giddyyupp/coco-minitrain data/miniCOCO2017

If you want to train with this dataset instead, download the dataset into the same directory as coco:
```
code_root/
└── data/
    |── coco/
	|	├── ...
	|
	|
	|── miniCOCO2017/
		|── ...
```

And then change the coco_path param from ```--coco_path data/coco``` to ```--coco_path data/miniCOCO2017```

```--top_K_matching_method``` can be either 'closest' or 'bipartite'. All other params can be tweaked to your likings
