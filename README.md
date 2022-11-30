# Face embedding trainer

This repository contains the framework for training deep embeddings for face recognition. The trainer is intended for the face recognition exercise of the [EE488B Deep Learning for Visual Understanding](https://mm.kaist.ac.kr/teaching/) course. This is an adaptation of the [speaker recognition model trainer](https://github.com/clovaai/voxceleb_trainer).

### Dependencies
```
pip install -r requirements.txt
```

### Training examples

- Softmax:
```
python ./trainEmbedNet.py --model ResNet18 --trainfunc softmax --save_path exps/exp1 --nClasses 2000 --batch_size 200 --gpu 8
```


### Training method I used => batch 64 for korface
```
python ./trainEmbedNet.py --model ResNet18 --trainfunc softmax --save_path pretrain_models/kor_VGG --batch_size 64 --max_epoch 200 --train_path data/kor_VGGface2/train --test_path data/kor_VGGface2/val --test_list data/kor_VGGface2/val_pairs.csv --gpu 1
```
### Training method I used => batch 200 for korface
```
python ./trainEmbedNet.py --model ResNet18 --trainfunc softmax --save_path pretrain_models/kor_VGG_batch200 --batch_size 200 --max_epoch 100 --train_path data/kor_VGGface2/train --test_path data/kor_VGGface2/val --test_list data/kor_VGGface2/val_pairs.csv --gpu 0
```
### Train using KOR VGGface preprocessed model => batch64
```
python ./trainEmbedNet.py --model ResNet18 --trainfunc softmax --save_path exps/pre_VGG --batch_size 64 --max_epoch 100 --train_path data/proj_data/train --test_path data/proj_data/val --test_list data/proj_data/val_pairs.csv  --gpu 1 --initial_model pretrain_models/kor_VGG/model000000200.model
```
### Train using KOR VGGface preprocessed model => batch200
```
python ./trainEmbedNet.py --model ResNet18 --trainfunc softmax --save_path exps/pre_VGG_batch200 --batch_size 200 --max_epoch 100 --train_path data/proj_data/train --test_path data/proj_data/val --test_list data/proj_data/val_pairs.csv  --gpu 1 --initial_model pretrain_models/kor_VGG/model000000200.model
```
### Pretrain data
```
python ./trainEmbedNet.py --model ResNet18 --trainfunc softmax --save_path pretrain_models/pretrain_ver1.0 --batch_size 200 --max_epoch 100 --train_path data/pretrain/train --test_interval 0 --test_list data/pretrain/val_pairs.csv --gpu 1 --train_ext wav
```

GPU ID must be specified using `--gpu` flag.

Use `--mixedprec` flag to enable mixed precision training. This is recommended for Tesla V100, GeForce RTX 20 series or later models.

### Implemented loss functions
```
Softmax (softmax)
Triplet (triplet)
```

For softmax-based losses, `nPerClass` should be 1, and `nClasses` must be specified. For metric-based losses, `nPerClass` should be 2 or more. 

### Implemented models
```
ResNet18
```
### Adding new models and loss functions

You can add new models and loss functions to `models` and `loss` directories respectively. See the existing definitions for examples.

### Data

The test list should contain labels and image pairs, one line per pair, as follows. `1` is a target and `0` is an imposter.
```
1,id10001/00001.jpg,id10001/00002.jpg
0,id10001/00003.jpg,id10002/00001.jpg
```

The folders in the training set should contain images for each identity (i.e. `identity/image.jpg`).

The input transformations can be changed in the code.

### Inference

In order to save pairwise similarity scores to file, use `--output` flag.