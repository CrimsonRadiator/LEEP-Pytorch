
# Pytorch implementation of LEEP
This repository has a Pytorch implementaion of [LEEP: A New Measure to Evaluate Transferability of Learned Representations](https://arxiv.org/abs/2002.12462)

## Dependencies
Code was tested with `torch==1.8.1`, but both newer and older versions should work alright.

## How to use it
[Here](https://colab.research.google.com/drive/11h5TTvRp3Gjw1Y7kO9S8kr2KsM64iBrm?usp=sharing) is a Google Colab notebook presenting LEEP usage on CIFAR 10.

Function call for calculating LEEP from the above notebook:
```
leep(model=model, data_loader=train_dataloader, number_of_target_labels=10, device=0)
```
Here are some important assumptions:
1. `data_loader` returns `(images, labels)` pairs, where labels are integers corresponding to the class label in the dataset
2. `data_loader` has `drop_last` property set to `False` - we need to calculate score for all of the images
