import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

from .annrescaler import AnnRescaler
from .. import headmeta
from ..visualizer import Cif as CifVisualizer
from ..utils import create_sink, mask_valid_area

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Cif:
    meta: headmeta.Cif
    rescaler: AnnRescaler = None  ##If we want CIF to process with different stide, we have to specify the rescaler with the corresponding stides
    v_threshold: int = 0
    bmin: float = 0.1  #: in pixels
    visualizer: CifVisualizer = None

    side_length: ClassVar[int] = 4
    padding: ClassVar[int] = 10

    use_fpn: bool = False
    head_index: int = None

    def __call__(self, image, anns, meta):
        return CifGenerator(self)(image, anns, meta)


class CifGenerator():
    def __init__(self, config: Cif):
        self.config = config

        self.rescaler = config.rescaler or AnnRescaler(
            config.meta.stride, config.meta.pose)
        self.visualizer = config.visualizer or CifVisualizer(config.meta)

        self.intensities = None
        self.fields_reg = None
        self.fields_bmin = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.sink = create_sink(config.side_length)
        self.s_offset = (config.side_length - 1.0) / 2.0

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        ###This operation maps the gd keypoint coordinates into its coordinates in the featuremap to formulate the target for regression part(vectoral part) in CifCaf
        keypoint_sets = self.rescaler.keypoint_sets(anns)
        bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.side_length - 1) / 2)
        valid_area = self.rescaler.valid_area(meta)
        LOG.debug('valid area: %s, pif side length = %d', valid_area, self.config.side_length)

        n_fields = len(self.config.meta.keypoints)
        self.init_fields(n_fields, bg_mask)
        self.fill(keypoint_sets)
        fields = self.fields(valid_area)
        #fields is the final target of the CIF head(mapped from the gd through a load of pre-processing)

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.config.padding
        field_h = bg_mask.shape[0] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_bmin = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_scale = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan

    def fill(self, keypoint_sets):
        for keypoints in keypoint_sets:
            self.fill_keypoints(keypoints)
        #keypoint_sets is a list of keypoints from different objects in the image
        #keypoints is a set of keypoints from the same object

    def fill_keypoints(self, keypoints):
        scale = self.rescaler.scale(keypoints) #each keypoints from the keypoint_sets is a set of keypoints from the same object
        #and (scale>8*self.config.meta.stride or scale<=4*self.config.meta.stride):
        # print('head index: {}, scale: {}'.format(self.config.head_index,scale))
        if self.config.use_fpn:
            # if (self.config.head_index == 0 and scale>8) or (self.config.head_index == -1 and scale<=4)\
            #     or (self.config.head_index != 0 and self.config.head_index != -1 and (scale>8 or scale<=4)):
            #     # print('ignore')
            #     return

            # if (self.config.head_index == 0 and scale>16) or (self.config.head_index == -1 and scale<=8)\
            #     or (self.config.head_index != 0 and self.config.head_index != -1 and (scale>16 or scale<=8)):
            #     # print('ignore')
            #     return

            if (self.config.head_index == 0 and scale>8/self.config.meta.upsample_stride) or (self.config.head_index == -1 and scale<=4/self.config.meta.upsample_stride)\
                or (self.config.head_index != 0 and self.config.head_index != -1 and (8/self.config.meta.upsample_stride or scale<=4/self.config.meta.upsample_stride)):
                # print('ignore')
                return

        #xyv is the coordinate of one keypoint in the keypoints of one instance in the featuremap
        for f, xyv in enumerate(keypoints):
            if xyv[2] <= self.config.v_threshold:
                continue

            joint_scale = (
                scale
                if self.config.meta.sigmas is None
                else scale * self.config.meta.sigmas[f]
            )

            self.fill_coordinate(f, xyv, joint_scale)

    def fill_coordinate(self, f, xyv, scale):
        ij = np.round(xyv[:2] - self.s_offset).astype(np.intc) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.config.side_length, miny + self.config.side_length 
        #side_length determines the size of Gaussian kernel length and s_offset is half of the side_length
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return

        offset = xyv[:2] - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(2, 1, 1)

        # mask
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        mask_peak = np.logical_and(mask, sink_l < 0.7)
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        #print out ij, minx, miny, maxx, maxy, offset, sink_reg, sink_l, mask, mask_peak in a nice form
        print('keypoint: {}'.format(xyv))
        print('padding: {}'.format(self.config.padding))
        print('ij: {}'.format(ij))
        print('minx: {}'.format(minx))
        print('miny: {}'.format(miny))
        print('maxx: {}'.format(maxx))
        print('maxy: {}'.format(maxy))
        print('offset: {}'.format(offset))
        print('sink_reg: {}'.format(sink_reg))
        print('sink_l: {}'.format(sink_l))
        print('mask: {}'.format(mask))
        print('mask_peak: {}'.format(mask_peak))
        #print out self.fields_reg_l[f, miny:maxy, minx:maxx][mask]


        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0
        self.intensities[f, miny:maxy, minx:maxx][mask_peak] = 1.0

        # update regression
        patch = self.fields_reg[f, :, miny:maxy, minx:maxx]
        patch[:, mask] = sink_reg[:, mask]

        # update bmin
        bmin = self.config.bmin / self.config.meta.stride
        self.fields_bmin[f, miny:maxy, minx:maxx][mask] = bmin

        # neglect extremely large object(100 is just a pre-defined threshold)
        # if scale >= 100:
        #     scale = np.nan

        # update scale
        assert np.isnan(scale) or 0.0 < scale < 100.0
        self.fields_scale[f, miny:maxy, minx:maxx][mask] = scale

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg = self.fields_reg[:, :, p:-p, p:-p]
        fields_bmin = self.fields_bmin[:, p:-p, p:-p]
        fields_scale = self.fields_scale[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale, valid_area, fill_value=np.nan)

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg,
            np.expand_dims(fields_bmin, 1),
            np.expand_dims(fields_scale, 1),
        ], axis=1))
