"Visualize the scale distribution of instances."

import argparse
import copy
import datetime
import logging
import os
import socket
import openpifpaf
import matplotlib.pyplot as plt
import numpy as np

import torch

# from . import datasets, encoder, logger, network, optimize, plugin, show, visualizer
# from . import __version__
import openpifpaf.datasets as datasets



def main():
    datamodule = datasets.factory('cocokp')

    print('square_edge: ', datamodule.square_edge)
    print('extended_scale: ', datamodule.extended_scale)
    print('orientation_invariant: ', datamodule.orientation_invariant)
    print('upsample_stride: ', datamodule.upsample_stride)

    # print(type(datamodule))
    # print(len(train_loader))
    datamodule.square_edge = 513
    datamodule.extended_scale = False
    datamodule.orientation_invariant = 0.1
    datamodule.upsample_stride = 2

    print('square_edge: ', datamodule.square_edge)
    print('extended_scale: ', datamodule.extended_scale)
    print('orientation_invariant: ', datamodule.orientation_invariant)
    print('upsample_stride: ', datamodule.upsample_stride)


    train_loader = datamodule.vis_train_loader()
    val_loader = datamodule.vis_val_loader()

    train_instance_scales = []
    val_instance_scales = []



    for batch, (image, anns, _) in enumerate(train_loader):
        if batch >= 1000:
            break
        # print(type(data))
        # print(data)
        for ann in anns:
            # print(ann['bbox'])
            # print(ann['bbox'][0][2])
            train_instance_scales.append((ann['bbox'][0][2].item() * ann['bbox'][0][3].item())**0.5)

    for batch, (image, anns, _) in enumerate(val_loader):
        if batch >= 1000:
            break
        for ann in anns:
            val_instance_scales.append((ann['bbox'][0][2].item() * ann['bbox'][0][3].item())**0.5)

    #draw a histogram of train_instance_scales
    #put the histogram in figure 1
    plt.figure()
    plt.hist(train_instance_scales, bins=600)
    plt.title("COCOKP Train Instance Scales")
    plt.xticks(np.arange(0, 600, 30))
    plt.xticks(rotation=90)
    # plt.show()
    #save the histogram
    plt.savefig('/scratch/jiguo/visualization/train_instance_scales.png')

    #draw a histogram of val_instance_scales
    #put the histogram in a new figure
    plt.figure()
    plt.hist(val_instance_scales, bins=600)
    plt.title("COCOKP Val Instance Scales")
    #set the x axis bins
    plt.xticks(np.arange(0, 600, 30))
    #rotate the x axis bins
    plt.xticks(rotation=90)
    # plt.show()
    #save the histogram
    plt.savefig('/scratch/jiguo/visualization/val_instance_scales.png')



if __name__ == '__main__':
    main()