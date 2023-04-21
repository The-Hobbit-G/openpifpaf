"Visualize the scale distribution of instances."

import argparse
import copy
import datetime
import logging
import os
import socket
import openpifpaf
import matplotlib.pyplot as plt

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
        # print(type(data))
        # print(data)
        for ann in anns:
            print(ann['bbox'])
            train_instance_scales.append(ann['bbox'][2] * ann['bbox'][3])

    for batch, (image, anns, _) in enumerate(val_loader):
        for ann in anns:
            val_instance_scales.append(ann['bbox'][2] * ann['bbox'][3])

    #draw a histogram of train_instance_scales
    plt.hist(train_instance_scales, bins=50)
    plt.title("COCOKP Train Instance Scales")
    # plt.show()
    #save the histogram
    plt.savefig('/scratch/izar/jiguo/visualization/train_instance_scales.png')

    #draw a histogram of val_instance_scales
    plt.hist(val_instance_scales, bins=50)
    plt.title("COCOKP Val Instance Scales")
    # plt.show()
    #save the histogram
    plt.savefig('/scratch/izar/jiguo/visualization/val_instance_scales.png')



if __name__ == '__main__':
    main()