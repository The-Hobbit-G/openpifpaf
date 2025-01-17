import argparse

import torch

import openpifpaf

from .cocokp import CocoKp
from .constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    COCODET_KEYPOINTS, #for detection with cifcaf
    COCODET_FULL_KEYPOINTS, #for detection with cifcaf
    COCO_PERSON_SKELETON,
    COCODET_SKELETON,  #for detection with cifcaf
    COCODET_FULL_SKELETON, #for detection with cifcaf
    COCO_PERSON_SIGMAS,
    COCO_DET_SIGMAS, #for detection with cifcaf
    COCO_DET_FULL_SIGMAS, #for detection with cifcaf
    COCO_PERSON_SCORE_WEIGHTS,
    COCO_UPRIGHT_POSE,
    DENSER_COCO_PERSON_CONNECTIONS,
    HFLIP,
)
from .dataset import CocoDataset

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

TEST_CATEGOERY = ['person']


class CocoDet(openpifpaf.datasets.DataModule):
    # cli configurable
    train_annotations = 'data-mscoco/annotations/instances_train2017.json'
    val_annotations = 'data-mscoco/annotations/instances_val2017.json'
    # val_annotations = 'data-mscoco/annotations/person_keypoints_val2017.json'
    eval_annotations = val_annotations
    train_image_dir = 'data-mscoco/images/train2017/'
    val_image_dir = 'data-mscoco/images/val2017/'
    eval_image_dir = val_image_dir


    ##test cifcafdet with just one categoty
    # train_annotations = 'data-mscoco/annotations/person_keypoints_train2017.json'
    # val_annotations = 'data-mscoco/annotations/person_keypoints_val2017.json'
    # # val_annotations = train_annotations
    # eval_annotations = val_annotations
    # train_image_dir = 'data-mscoco/images/train2017/'
    # val_image_dir = 'data-mscoco/images/val2017/'
    # # val_image_dir = train_image_dir
    # eval_image_dir = val_image_dir

    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1

    #add bmin for cocodet with cifcaf
    bmin = 0.1

    eval_annotation_filter = True

    '''
    def __init__(self):
        super().__init__()
        cifdet = openpifpaf.headmeta.CifDet('cifdet', 'cocodet', COCO_CATEGORIES)
        cifdet.upsample_stride = self.upsample_stride
        self.head_metas = [cifdet]
    '''

    def __init__(self):
        super().__init__()
        #Use full corners and skeletons for cifcaf with sigmas(joint scales)
        cifdet = openpifpaf.headmeta.Cif('cif', 'cocodet',
                                      keypoints=COCODET_FULL_KEYPOINTS,
                                      sigmas=COCO_DET_FULL_SIGMAS,
                                      draw_skeleton=COCODET_FULL_SKELETON,
                                      categories=COCO_CATEGORIES,)
        cifdet.upsample_stride = self.upsample_stride
        cafdet = openpifpaf.headmeta.Caf('caf', 'cocodet',
                                      keypoints=COCODET_FULL_KEYPOINTS,
                                      sigmas=COCO_DET_FULL_SIGMAS,
                                      skeleton=COCODET_FULL_SKELETON,
                                      categories=COCO_CATEGORIES,)
        cafdet.upsample_stride = self.upsample_stride


        #Use patial corners and skeletons for cifcafdet
        # cifdet = openpifpaf.headmeta.Cif('cif', 'cocodet',
        #                               keypoints=COCODET_KEYPOINTS,
        #                               sigmas=None,
        #                               draw_skeleton=COCODET_SKELETON,
        #                               categories=COCO_CATEGORIES,)
        # cifdet.upsample_stride = self.upsample_stride
        # cafdet = openpifpaf.headmeta.Caf('caf', 'cocodet',
        #                               keypoints=COCODET_KEYPOINTS,
        #                               sigmas=None,
        #                               skeleton=COCODET_SKELETON,
        #                               categories=COCO_CATEGORIES,)
        # cafdet.upsample_stride = self.upsample_stride

        #Use patial corners and skeletons for cifcafdet with sigmas(joint scales)
        # cifdet = openpifpaf.headmeta.Cif('cif', 'cocodet',
        #                               keypoints=COCODET_KEYPOINTS,
        #                               sigmas=COCO_DET_SIGMAS,
        #                               draw_skeleton=COCODET_SKELETON,
        #                               categories=COCO_CATEGORIES,)
        # cifdet.upsample_stride = self.upsample_stride
        # cafdet = openpifpaf.headmeta.Caf('caf', 'cocodet',
        #                               keypoints=COCODET_KEYPOINTS,
        #                               sigmas=COCO_DET_SIGMAS,
        #                               skeleton=COCODET_SKELETON,
        #                               categories=COCO_CATEGORIES,)
        # cafdet.upsample_stride = self.upsample_stride

        self.head_metas = [cifdet, cafdet]
    
    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module CocoDet')

        group.add_argument('--cocodet-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--cocodet-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--cocodet-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--cocodet-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--cocodet-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--cocodet-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--cocodet-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--cocodet-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--cocodet-no-augmentation',
                           dest='cocodet_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--cocodet-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

        group.add_argument('--cocodet-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        #add bmin for cocodet with cifcaf
        group.add_argument('--cocodet-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocodet specific
        cls.train_annotations = args.cocodet_train_annotations
        cls.val_annotations = args.cocodet_val_annotations
        cls.train_image_dir = args.cocodet_train_image_dir
        cls.val_image_dir = args.cocodet_val_image_dir

        cls.square_edge = args.cocodet_square_edge
        cls.extended_scale = args.cocodet_extended_scale
        cls.orientation_invariant = args.cocodet_orientation_invariant
        cls.blur = args.cocodet_blur
        cls.augmentation = args.cocodet_augmentation
        cls.rescale_images = args.cocodet_rescale_images
        cls.upsample_stride = args.cocodet_upsample

        #add bmin for cocodet with cifcaf
        cls.bmin = args.cocodet_bmin

        cls.eval_annotation_filter = args.coco_eval_annotation_filter

    '''
    def _preprocess(self):
        enc = openpifpaf.encoder.CifDet(self.head_metas[0])

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders([enc]),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.5 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.7 * self.rescale_images,
                             1.5 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(10.0)],
                [self.orientation_invariant, 0.2],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.MinSize(min_side=4.0),
            openpifpaf.transforms.UnclippedArea(threshold=0.75),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders([enc]),
        ])
    '''
    
    def _preprocess(self):
        # enc = openpifpaf.encoder.CifDet(self.head_metas[0])
        encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin)]

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.5 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.7 * self.rescale_images,
                             1.5 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(10.0)],
                [self.orientation_invariant, 0.2],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.MinSize(min_side=4.0),
            openpifpaf.transforms.UnclippedArea(threshold=0.75),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])
    
    def train_loader(self):
        '''
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            category_ids=[],
        )
        '''

        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self.multiencoder_process(),
            annotation_filter=True,
            category_ids=[],
        )
        

        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        '''
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            category_ids=[],
        )
        '''

        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self.multiencoder_process(),
            annotation_filter=True,
            category_ids=[],
        )
        

        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @staticmethod
    def _eval_preprocess():
        return openpifpaf.transforms.Compose([
            *CocoKp.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToDetAnnotations(COCO_CATEGORIES),
                openpifpaf.transforms.ToCrowdAnnotations(COCO_CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoDataset(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [openpifpaf.metric.Coco(
            pycocotools.coco.COCO(self.eval_annotations),
            max_per_image=100,
            category_ids=[],
            iou_type='bbox',
        )]
    




    #for visulization of scale distribution
    def vis_preprocess(self):
        # encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
        #             openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin)]
        # if len(self.head_metas) > 2:
        #     encoders.append(openpifpaf.encoder.Caf(self.head_metas[2], bmin=self.bmin))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.5 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.7 * self.rescale_images,
                             1.5 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(10.0)],
                [self.orientation_invariant, 0.2],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.MinSize(min_side=4.0),
            openpifpaf.transforms.UnclippedArea(threshold=0.75),
            openpifpaf.transforms.TRAIN_TRANSFORM,
        ])
    
    def vis_train_loader(self):
        '''
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        '''
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self.vis_preprocess(),
            annotation_filter=True,
            category_ids=[],
        )

        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True,
            pin_memory=True, num_workers=0, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def vis_val_loader(self):
        '''
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        '''
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self.vis_preprocess(),
            annotation_filter=True,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=True,
            pin_memory=True, num_workers=0, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)
