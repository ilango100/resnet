# Resnet

This repository contains tensorflow implementation of Residual Neural Network. For comparison, three network architectures are implemented:
- PlainNet: Conventional stacking of Convolutional layers
- [ResNet](https://arxiv.org/abs/1512.03385): Stacking of Residual "blocks"
- [ResNetV2](https://arxiv.org/abs/1603.05027): Stacking Residual blocks with Identity mappings

## Usage

To run sample training on sin function dataset, run
```
$ python sin_function.py
```
This generates the comparison graphs between PlainNet and ResNet.

To run the training on CIFAR-10 with a particular architecture, run
```
$ python train.py <Network> -n <nblocks>
```
**Eg**: `python train.py ResNetV2 -n 4`

Apart from the defaults, you can also specify command line parameters to tweak the whole network:
```
$ python train.py --help
```
to see all the options.

## Results on CIFAR 10

The results obtained on CIFAR 10 dataset are on this [tensorboard log](https://tensorboard.dev/experiment/n2VONYJsRRC1nGLI18aLag). To replicate the experiment, run in shell:
```
for net in PlainNet ResNet ResNetV2; do
	for size in 2 4 6; do
		python train.py $net -n $size
	done
done
```

