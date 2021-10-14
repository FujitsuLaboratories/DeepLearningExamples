# pytorch + oneDNN example

## About this
- Sample programs to check the effects of oneDNN.
- AlexNet for a training model, CIFAR-10 for a training data.

## File structure
- README.md： this file
- READMEjp.md： README(Japanese)
- main.py： main program
- alexnet.py： training model (AlexNet)

## Requirements
- python 3.x
- torch
- torchvision
- argparse

## Advance Preparation
- Install libraries with pip.
```
$ pip install torch torchvision argparse tqdm
```

## Usage
```
$ python3 main.py --onednn -e 10
```

## Options
- -e, --epochs num(int): number of epochs(default=2)
- -o, --onednn(bool): use oneDNN
- -h, --help: help messages

## Input files
- CIFAR-10 dataset (automaticaly acquired)

## Output files
- None(standard output)

## Copyright
COPYRIGHT Fujitsu Limited 2021
