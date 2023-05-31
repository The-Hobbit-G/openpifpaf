import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

from .annrescaler import AnnRescalerDet
from .. import headmeta
from ..visualizer import CifDet as CifDetVisualizer
from ..utils import create_sink, mask_valid_area, create_sink_2d

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class CifDet:
    meta: headmeta.CifDet
    rescaler: AnnRescalerDet = None
    v_threshold: int = 0
    bmin: float = 1.0  #: in pixels
    visualizer: CifDetVisualizer = None

    # side_length: ClassVar[int] = 5
    side_length = 5
    padding: ClassVar[int] = 10

    use_fpn: bool = False
    head_index: int = None

    def __call__(self, image, anns, meta):
        return CifDetGenerator(self)(image, anns, meta)


class CifDetGenerator():
    def __init__(self, config: CifDet):
        self.config = config

        self.rescaler = config.rescaler or AnnRescalerDet(
            config.meta.stride, len(config.meta.categories))
        self.visualizer = config.visualizer or CifDetVisualizer(config.meta)

        self.intensities = None
        self.fields_reg = None
        self.fields_wh = None
        self.fields_reg_bmin = None
        self.fields_wh_bmin = None
        self.fields_reg_l = None

        if config.side_length >= 1:
            self.sink = create_sink(config.side_length)
            self.s_offset = (config.side_length - 1.0) / 2.0

    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        detections = self.rescaler.detections(anns)
        bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.side_length - 1) / 2)
        valid_area = self.rescaler.valid_area(meta)
        LOG.debug('valid area: %s, pif side length = %d', valid_area, self.config.side_length)

        n_fields = len(self.config.meta.categories)
        self.init_fields(n_fields, bg_mask)
        self.fill(detections)
        fields = self.fields(valid_area)

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[-1] + 2 * self.config.padding
        field_h = bg_mask.shape[-2] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_wh = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_bmin = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_wh_bmin = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][bg_mask == 0] = np.nan

    def fill(self, detections):
        for category_id, bbox in detections:
            xy = bbox[:2] + 0.5 * bbox[2:]  #center of the bbox
            wh = bbox[2:] #width and height of the bbox
            #TODO: assign boxes of different scales to different levels of FPN
            scale = np.sqrt(wh[0]*wh[1])
            if self.config.use_fpn:
                # if (self.config.head_index == 0 and scale>8) or (self.config.head_index == -1 and scale<=4)\
                #     or (self.config.head_index != 0 and self.config.head_index != -1 and (scale>8 or scale<=4)):
                #     # print('ignore')
                #     continue

                # if (self.config.head_index == 0 and scale>16) or (self.config.head_index == -1 and scale<=8)\
                #     or (self.config.head_index != 0 and self.config.head_index != -1 and (scale>16 or scale<=8)):
                #     # print('ignore')
                #     continue

                if (self.config.head_index == 0 and scale>8/self.config.meta.upsample_stride) or (self.config.head_index == -1 and scale<=4/self.config.meta.upsample_stride)\
                or (self.config.head_index != 0 and self.config.head_index != -1 and (scale>8/self.config.meta.upsample_stride or scale<=4/self.config.meta.upsample_stride)):
                    # print('ignore')
                    continue

            if self.config.side_length == -1:
                self.fill_detection_full(category_id - 1, xy, wh)
            elif self.config.side_length == -2:
                self.fill_detection_max4(category_id - 1, xy, wh)
            else: 
                self.fill_detection(category_id - 1, xy, wh)

    def fill_detection(self, f, xy, wh):
        ij = np.round(xy - self.s_offset).astype(np.intc) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.config.side_length, miny + self.config.side_length
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return

        offset = xy - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(2, 1, 1)

        # mask
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        core_radius = (self.config.side_length - 1) / 2.0
        mask_fringe = np.logical_and(
            sink_l > core_radius,
            sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx],
        )
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0
        self.intensities[f, miny:maxy, minx:maxx][mask_fringe] = np.nan

        # update regression
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = sink_reg[:, mask]

        # update wh
        assert wh[0] > 0.0
        assert wh[1] > 0.0
        self.fields_wh[f, :, miny:maxy, minx:maxx][:, mask] = np.expand_dims(wh, 1)

        # update bmin
        half_scale = 0.5 * min(wh[0], wh[1])
        bmin = max(0.1 * half_scale, self.config.bmin / self.config.meta.stride)
        self.fields_reg_bmin[f, miny:maxy, minx:maxx][mask] = bmin
        self.fields_wh_bmin[f, miny:maxy, minx:maxx][mask] = bmin



    def fill_detection_full(self, f, xy, wh):
        '''
        Use a wxh field pointing towards the center
        '''
        xy_offset = [-0.5, -0.5]
        #xy_offset = [0, 0]
        # minx, miny = xy[0]-wh[0], xy[1]-wh[1]
        # maxx, maxy = np.max(kps[kps[:,2]>0, 0]), np.max(kps[kps[:,2]>0, 1])
        # w = np.round(maxx - minx + 0.5).astype(np.int)

        # h = np.round(maxy- miny + 0.5).astype(np.int)
        w = np.round(wh[0] + 0.5).astype(np.int)
        h = np.round(wh[1] + 0.5).astype(np.int)

        # xyv = kps[-1]
        xy_offset = [(w - 1.0) / 2.0, (h - 1.0) / 2.0]

        ij = np.round(xy - xy_offset).astype(np.int) + self.config.padding
        offset = xy - (ij + xy_offset - self.config.padding)
        minx, miny = int(ij[0]), int(ij[1])

        # if not self.obutterfly:
        #     if w<=0:
        #         raise Exception('w= ', w, ' is negative or zero')
        #     if h<=0:
        #         raise Exception('h= ', h, ' is negative or zero')
        # else:
        if w==0:
            w = 1
        if h==0:
            h=1

        maxx, maxy = minx + w, miny + h

        sink = create_sink_2d(w, h)

        if minx + w/2 < 0 or maxx - w/2 > self.intensities.shape[2] or \
           miny + h/2 < 0 or maxy - h/2 > self.intensities.shape[1]:
            return

        minx_n = max(0, minx)
        miny_n = max(0, miny)
        maxx_n = min(maxx, self.intensities.shape[2])
        maxy_n = min(maxy, self.intensities.shape[1])
        sink = sink[:, (miny_n-miny):(miny_n-miny) + (maxy_n-miny_n), (minx_n-minx):(minx_n-minx) + (maxx_n-minx_n)] # sink[:, 0: maxy_n-miny_n, 0: maxx_n-minx_n]
        minx = minx_n
        maxx = maxx_n
        miny = miny_n
        maxy = maxy_n
        offset = offset.reshape(2, 1, 1)

        # update intensity
        # self.intensities[f, miny:maxy, minx:maxx] = 1.0
        # update regression
        sink_reg = sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]

        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0
        # self.intensities[f, miny:maxy, minx:maxx][mask_fringe] = np.nan

        # update regression
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = sink_reg[:, mask]

        # update wh
        assert wh[0] > 0.0
        assert wh[1] > 0.0
        self.fields_wh[f, :, miny:maxy, minx:maxx][:, mask] = np.expand_dims(wh, 1)

        # update bmin
        half_scale = 0.5 * min(wh[0], wh[1])
        bmin = max(0.1 * half_scale, self.config.bmin / self.config.meta.stride)
        self.fields_reg_bmin[f, miny:maxy, minx:maxx][mask] = bmin
        self.fields_wh_bmin[f, miny:maxy, minx:maxx][mask] = bmin

    

    def fill_detection_max4(self, f, xy, wh):
        '''
        Use a 4x4 field with ignore region surrounding it.
        '''
        xy_offset = [-0.5, -0.5]

        w = np.round(wh[0] + 0.5).astype(np.int)
        h = np.round(wh[1] + 0.5).astype(np.int)

        # xyv = kps[-1]

        xy_offset = [(4 - 1.0) / 2.0, (4 - 1.0) / 2.0]
        
        ij = np.round(xy - xy_offset).astype(np.int) + self.config.padding
        offset = xy - (ij + xy_offset - self.config.padding)
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + 4, miny + 4

        if minx + 2 < 0 or maxx - 2 > self.intensities.shape[2] or \
           miny + 2 < 0 or maxy - 2 > self.intensities.shape[1]:
            return

        if w > 16 or h>16:
            sigma_ignore, sigma_eff = 0.5, 0.2
            w_ignore, h_ignore = np.round(sigma_ignore*w + 0.5).astype(np.int), np.round(sigma_ignore*h+ 0.5).astype(np.int)
            xy_offset = [(w_ignore - 1.0) / 2.0, (h_ignore - 1.0) / 2.0]
            ij = np.round(xy - xy_offset).astype(np.int) + self.config.padding
            if w>16:
                minx_ignore = int(ij[0])
            else:
                minx_ignore = minx
                w_ignore = 4
            if h>16:
                miny_ignore = int(ij[1])
            else:
                miny_ignore = miny
                h_ignore = 4
            maxx_ignore, maxy_ignore = minx_ignore + w_ignore, miny_ignore + h_ignore
            self.intensities[f, miny_ignore:maxy_ignore, minx_ignore:maxx_ignore] = np.nan
        w = 4
        h = 4

        sink = create_sink_2d(w, h)

        minx_n = max(0, minx)
        miny_n = max(0, miny)
        maxx_n = min(maxx, self.intensities.shape[2])
        maxy_n = min(maxy, self.intensities.shape[1])
        sink = sink[:, (miny_n-miny):(miny_n-miny) + (maxy_n-miny_n), (minx_n-minx):(minx_n-minx) + (maxx_n-minx_n)]
        minx = minx_n
        maxx = maxx_n
        miny = miny_n
        maxy = maxy_n
        offset = offset.reshape(2, 1, 1)

         # update intensity
        # self.intensities[f, miny:maxy, minx:maxx] = 1.0
        # update regression
        sink_reg = sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]

        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0
        # self.intensities[f, miny:maxy, minx:maxx][mask_fringe] = np.nan

        # update regression
        self.fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = sink_reg[:, mask]

        # update wh
        assert wh[0] > 0.0
        assert wh[1] > 0.0
        self.fields_wh[f, :, miny:maxy, minx:maxx][:, mask] = np.expand_dims(wh, 1)

        # update bmin
        half_scale = 0.5 * min(wh[0], wh[1])
        bmin = max(0.1 * half_scale, self.config.bmin / self.config.meta.stride)
        self.fields_reg_bmin[f, miny:maxy, minx:maxx][mask] = bmin
        self.fields_wh_bmin[f, miny:maxy, minx:maxx][mask] = bmin








    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg = self.fields_reg[:, :, p:-p, p:-p]
        fields_wh = self.fields_wh[:, :, p:-p, p:-p]
        fields_reg_bmin = self.fields_reg_bmin[:, p:-p, p:-p]
        fields_wh_bmin = self.fields_wh_bmin[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_wh[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_wh[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg_bmin, valid_area, fill_value=np.nan)
        mask_valid_area(fields_wh_bmin, valid_area, fill_value=np.nan)

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg,
            fields_wh,
            np.expand_dims(fields_reg_bmin, 1),
            np.expand_dims(fields_wh_bmin, 1),
        ], axis=1))
