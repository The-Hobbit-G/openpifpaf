import argparse
import logging
import dataclasses
from typing import List
from abc import ABC, abstractmethod
import copy

import openpifpaf

import torch

from .. import headmeta, metric

LOG = logging.getLogger(__name__)


class DataModule:
    """
    Base class to extend OpenPifPaf with custom data.

    This class gives you all the handles to train OpenPifPaf on a new dataset.
    Create a new class that inherits from this to handle a new datasets.


    1. Define the PifPaf heads you would like to train. \
    For example, \
    CIF (Composite Intensity Fields) to detect keypoints, and \
    CAF (Composite Association Fields) to associate joints \

    2. Add class variables, such as annotations, training/validation image paths.

    """

    #: Data loader batch size.
    batch_size = 1

    #: Data loader number of workers.
    _loader_workers = None

    #: A list of head metas for this dataset.
    #: Set as instance variable (not class variable) in derived classes
    #: so that different instances of head metas are created for different
    #: instances of the data module. Head metas contain the base stride which
    #: might be different for different data module instances.
    #: When loading a checkpoint, entries in this list will be matched by
    #: name and dataset to entries in the checkpoint and overwritten here.
    head_metas: List[headmeta.Base] = None


    #A list of strides that should be explicitly specified by the user and used for encoder
    head_stride: List[int] = [4,8,16]

    #A flag for specifying whether to use fpn and adjust the targets correspondingly
    use_fpn: bool = False


    #classmethod is a built-in decorator that can be used to define a method that operates on the class rather than on instances of the class. 
    #A classmethod receives the class itself as its first argument instead of the instance, and it can access and modify the class-level data.
    #This method takes the class itself as its first argument (usually called cls by convention) instead of self. 
    #Inside the method, we can access and modify class-level data using the class name (e.g. cls.class_attr1).
    #we don't need to create an instance of DataModule to call this method. Instead, we have called it on the class itself.

    @classmethod
    def set_loader_workers(cls, value):
        cls._loader_workers = value

    @property
    def loader_workers(self):
        if self._loader_workers is not None:
            return self._loader_workers

        # Do not propose more than 16 loaders. More loaders use more
        # shared memory. When shared memory is exceeded, all jobs
        # on that machine crash.
        return min(16, self.batch_size)  ###change back to 16 later!!!

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        r"""
        Command line interface (CLI) to extend argument parser for your custom dataset.

        Make sure to use unique CLI arguments for your dataset.
        For clarity, we suggest to start every CLI argument with the name of your new dataset,
        i.e. \-\-<dataset_name>-train-annotations.

        All PifPaf commands will still work.
        E.g. to load a model, there is no need to implement the command \-\-checkpoint
        """

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

    def metrics(self) -> List[metric.Base]:
        """Define a list of metrics to be used for eval."""
        raise NotImplementedError

    def train_loader(self) -> torch.utils.data.DataLoader:
        """
        Loader of the training dataset.

        A Coco Data loader is already available, or a custom one can be created and called here.
        To modify preprocessing steps of your images (for example scaling image during training):

        1. chain them using torchvision.transforms.Compose(transforms)
        2. pass them to the preprocessing argument of the dataloader"""
        raise NotImplementedError

    def val_loader(self) -> torch.utils.data.DataLoader:
        """
        Loader of the validation dataset.

        The augmentation and preprocessing should be the same as for train_loader.
        The only difference is the set of data. This allows to inspect the
        train/val curves for overfitting.

        As in the train_loader, the annotations should be encoded fields
        so that the loss function can be computed.
        """
        raise NotImplementedError

    def eval_loader(self) -> torch.utils.data.DataLoader:
        """
        Loader of the evaluation dataset.

        For local runs, it is common that the validation dataset is also the
        evaluation dataset. This is then changed to test datasets (without
        ground truth) to produce predictions for submissions to a competition
        server that holds the private ground truth.

        This loader shouldn't have any data augmentation. The images should be
        as close as possible to the real application.
        The annotations should be the ground truth annotations similarly to
        what the output of the decoder is expected to be.
        """
        raise NotImplementedError

    @staticmethod
    def distributed_sampler(loader: torch.utils.data.DataLoader):
        LOG.info('Replacing sampler of %s with DistributedSampler.', loader)
        distributed_sampler = torch.utils.data.DistributedSampler(
            loader.dataset, shuffle=True, drop_last=True)

        return torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            drop_last=True,
            shuffle=False,
            sampler=distributed_sampler,
            pin_memory=loader.pin_memory,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
        )


    def multiencoder_process(self):
        preprocess_compose = copy.deepcopy(self._preprocess())
        if self.use_fpn:
            assert len(self.head_stride) >= 1
            if isinstance(preprocess_compose.preprocess_list[-1], openpifpaf.transforms.Encoders \
                          or openpifpaf.transforms.pair.Encoders):
                ori_encoders = preprocess_compose.pop(-1)

                '''
                #TODO: Consider the situation where enc in ori_encoders.encoders could be openpifpaf.encoder.SingleImage
                if type(ori_encoders) == openpifpaf.transforms.Encoders:
                    new_encoders = [openpifpaf.transforms.Encoders([dataclasses.replace(enc,meta = dataclasses.replace(enc.meta, base_stride = hs)) if type(enc) != openpifpaf.encoder.SingleImage \
                                                                    else dataclasses.replace(enc, wrapped = dataclasses.replace(enc.wrapped, meta = dataclasses.replace(enc.wrapped.meta, base_stride = hs)))]\
                                                                            for enc in ori_encoders.encoders) for hs in self.head_stride]
                else:
                    new_encoders = [openpifpaf.transforms.pair.Encoders([dataclasses.replace(enc,meta = dataclasses.replace(enc.meta, base_stride = hs)) if type(enc) != openpifpaf.encoder.SingleImage \
                                                                    else dataclasses.replace(enc, wrapped = dataclasses.replace(enc.wrapped, meta = dataclasses.replace(enc.wrapped.meta, base_stride = hs)))]\
                                                                            for enc in ori_encoders.encoders) for hs in self.head_stride]
                #eg. [encoder.Cif(headmeta.Cif(base_stride=16)),encoder.Caf(headmeta.Caf(base_stride=16)),...]-->[[encoder.Cif(headmeta.Cif(base_stride=4)),encoder.Caf(headmeta.Caf(base_stride=4,)),...],
                # [encoder.Cif(headmeta.Cif(base_stride=8)),encoder.Caf(headmeta.Caf(base_stride=8,)),...],[encoder.Cif(headmeta.Cif(base_stride=16)),encoder.Caf(headmeta.Caf(base_stride=16)),...]]
                #Now the final element of preprocess_compose changes from type<openpifpaf.transforms.Encoders/openpifpaf.transforms.pair.Encoders> to 
                #List[type<openpifpaf.transforms.Encoders/openpifpaf.transforms.pair.Encoders>]
                '''

                #Since base_stride is set to init=False and cannot be applied replace, we try the following implementation
                new_encoders = []
                for h_index,hs in enumerate(self.head_stride):
                    if h_index == len(self.head_stride)-1:
                        h_index = -1
                    new_encs = []
                    for enc in ori_encoders.encoders:
                        if type(enc) != openpifpaf.encoder.SingleImage:
                            new_meta = copy.deepcopy(enc.meta)
                            new_meta.base_stride = hs
                            new_bmin = enc.bmin
                            # new_enc = dataclasses.replace(enc,meta = new_meta) 
                            # can't use dataclasses to renew the Caf,TCaf encoder because if we do so, we are replacing just one attribute(meta)
                            # And all their old attributes are kept(including rescaler, the __post_init__ function won't renew the rescaler so the stride of the rescaler will never change)
                            
                            #adjust the Gaussian kernel size at each stage
                            new_enc = enc.__class__(meta=new_meta, bmin = new_bmin, use_fpn = True, head_index = h_index)
                            if isinstance(enc, openpifpaf.encoder.Cif) or isinstance(enc, openpifpaf.encoder.CifDet):
                                if type(enc.side_length)==list: 
                                    assert(len(enc.side_length)==len(self.head_stride))
                                    new_side_length = copy.deepcopy(enc.side_length)[h_index]
                                    # new_enc = enc.__class__(meta=new_meta, bmin = new_bmin, side_length=new_side_length, use_fpn = True, head_index = h_index)
                                    new_enc.side_length = new_side_length
                                    
                            elif isinstance(enc, openpifpaf.encoder.Caf):
                                if type(enc.min_size)==list: 
                                    assert(len(enc.min_size)==len(self.head_stride))
                                    new_min_size = copy.deepcopy(enc.min_size)[h_index]
                                    # new_enc = enc.__class__(meta=new_meta, bmin = new_bmin, min_size=new_min_size, use_fpn = True, head_index = h_index)
                                    new_enc.min_size = new_min_size
                                # else:
                                #     new_enc = enc.__class__(meta=new_meta, bmin = new_bmin, use_fpn = True, head_index = h_index)
                        else:
                            new_meta = copy.deepcopy(enc.wrapped.meta)
                            new_meta.base_stride = hs
                            new_bmin = enc.wrapped.bmin
                            # new_enc = dataclasses.replace(enc, wrapped = dataclasses.replace(enc.wrapped, meta = new_meta))
                            new_enc = dataclasses.replace(enc, wrapped = enc.wrapped.__class__(meta=new_meta,bmin=new_bmin,use_fpn = True, head_index = h_index))
                            #adjust the Gaussian kernel size at each stage
                            if isinstance(enc.wrapped, openpifpaf.encoder.Cif) or isinstance(enc.wrapped, openpifpaf.encoder.CifDet):
                                if type(enc.wrapped.side_length)==list: 
                                    assert(len(enc.wrapped.side_length)==len(self.head_stride))
                                    new_side_length = copy.deepcopy(enc.wrapped.side_length)[h_index]
                                    # new_enc = dataclasses.replace(enc, wrapped = enc.wrapped.__class__(meta=new_meta,bmin=new_bmin,side_length=new_side_length,use_fpn = True, head_index = h_index))
                                    new_enc.wrapped.side_length = new_side_length
                                # else:
                                #     new_enc = dataclasses.replace(enc, wrapped = enc.wrapped.__class__(meta=new_meta,bmin=new_bmin,use_fpn = True, head_index = h_index))
                            elif isinstance(enc.wrapped, openpifpaf.encoder.Caf):
                                if type(enc.wrapped.min_size)==list: 
                                    assert(len(enc.wrapped.min_size)==len(self.head_stride))
                                    new_min_size = copy.deepcopy(enc.wrapped.min_size)[h_index]
                                    # new_enc = dataclasses.replace(enc, wrapped = enc.wrapped.__class__(meta=new_meta,bmin=new_bmin,min_size=new_min_size,use_fpn = True, head_index = h_index))
                                    new_enc.wrapped.min_size = new_min_size
                                # else:
                                #     new_enc = dataclasses.replace(enc, wrapped = enc.wrapped.__class__(meta=new_meta,bmin=new_bmin,use_fpn = True, head_index = h_index))
                        new_encs.append(new_enc)
                    new_encoder = ori_encoders.__class__(new_encs)
                    new_encoders.append(new_encoder)


                preprocess_compose.append(new_encoders)
        # self._preprocess() = preprocess_compose
        return preprocess_compose


    @abstractmethod
    def _preprocess(self):
        pass