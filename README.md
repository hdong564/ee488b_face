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

### pretrain cropped vgg iresnet(too many params)
```
python ./trainEmbedNet.py --model iResNet101 --trainfunc supcontrast --save_path pretrained_model/ires-supc_b200 --nClasses 2000 --batch_size 200 --gpu 1 --train_path data/resized_vggface2
```
### pretrain cropped vgg EfficientNet(too many params)
```
python ./trainEmbedNet.py --model EfficientNet --trainfunc supcontrast --save_path pretrained_model/EfficientNet_Supcon --train_path data/resized_vggface2 --test_path data/kor_VGGface2/val --test_list data/kor_VGGface2/val_pairs.csv --max_epoch 30  --test_interval 3 --nOut 256 --gpu 1 --batch_size 32
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