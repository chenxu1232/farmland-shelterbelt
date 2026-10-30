# Classification of Farmland Protective Forests

This repository contains Python source code for semantic segmentation experiments related to farmland protective forest classification.

## Repository Structure

- `123/`: Research implementation, training, prediction, and evaluation code.
- `nets/`: SegFormer backbone and network training modules.
- `utils/`: Data loading, training utilities, callbacks, and evaluation metrics.

## Reference Environment

The following versions were observed in the AutoDL environment:

- Python 3.8.10
- PyTorch 2.0.0+cu118
- Torchvision 0.15.1+cu118
- NumPy 1.24.2
- Pillow 9.4.0
- Matplotlib 3.7.1
- tqdm 4.61.2
- timm 0.9.2

The complete dependency list and clean-environment installation procedure remain to be verified.

## Model Configuration

The active configuration identified in `123/train.py` includes:

| Parameter | Value |
|---|---|
| Backbone | SegFormer-B5 |
| Input resolution | 512 x 512 |
| num_classes | 2 |
| Initial learning rate | 0.0001 |
| Minimum learning rate | 0.000001 |
| Optimizer | AdamW |
| Learning rate schedule | Cosine |
| Initial epoch | 0 |
| Maximum epoch | 100 |
| Freeze training | False |
| Freeze batch size setting | 8 |
| Unfreeze batch size setting | 4 |

These values describe the current source-code configuration and are not independently verified as the settings used for every reported experiment.

## Dataset

The training implementation uses a VOC-style segmentation dataset.

The source code references:

- `VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt`
- `VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt`

The dataset is not included in this repository.

Dataset access instructions and the complete directory layout will be documented separately.

## Pretrained Weights

The current training configuration references:

`model_data/segformer_b5_backbone_weights.pth`

Pretrained weights are not included in this repository.

## Reproducibility Status

The source files have been published, but complete reproducibility from a clean environment has not yet been verified.

Training commands, inference commands, full dependencies, dataset instructions, and checkpoint access information require further validation.

## License and Citation

License information, third-party code attribution, and the final paper citation will be added after verification.
