import dataclasses
import logging
from typing import ClassVar, List, Tuple

import numpy as np
import torch

from .annrescaler import AnnRescaler
from .. import headmeta
from ..visualizer import Caf as CafVisualizer
from ..utils import mask_valid_area

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Caf:
    meta: headmeta.Caf
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 0.1  #: in pixels
    visualizer: CafVisualizer = None
    fill_plan: List[Tuple[int, int, int]] = None

    # min_size: ClassVar[int] = 3
    min_size = 3
    fixed_size: ClassVar[bool] = False
    aspect_ratio: ClassVar[float] = 0.0
    padding: ClassVar[int] = 10

    use_fpn: bool = False
    head_index: int = None

    def __post_init__(self):
        if self.rescaler is None:
            self.rescaler = AnnRescaler(self.meta.stride, self.meta.pose)

        if self.visualizer is None:
            self.visualizer = CafVisualizer(self.meta)

        if self.fill_plan is None:
            self.fill_plan = [
                (caf_i, joint1i - 1, joint2i - 1)
                for caf_i, (joint1i, joint2i) in enumerate(self.meta.skeleton)
            ] 
            # caf_i is the index of the i_th joint connection and joint1i and joint2i are the indices of the joints connected by the i_th joint connection
            # -1 because python index from 0

    def __call__(self, image, anns, meta):
        return CafGenerator(self)(image, anns, meta)


class AssociationFiller:
    def __init__(self, config: Caf):
        self.config = config
        self.rescaler = config.rescaler
        self.visualizer = config.visualizer

        self.sparse_skeleton_m1 = (
            np.asarray(config.meta.sparse_skeleton) - 1
            if getattr(config.meta, 'sparse_skeleton', None) is not None
            else None
        )

        if self.config.fixed_size:
            assert self.config.aspect_ratio == 0.0

        LOG.debug('only_in_field_of_view = %s, caf min size = %d',
                  config.meta.only_in_field_of_view,
                  self.config.min_size)

        self.field_shape = None
        self.fields_reg_l = None

    def init_fields(self, bg_mask):
        raise NotImplementedError
    
    def init_cafdet_fields(self, bg_mask):
        raise NotImplementedError

    def all_fill_values(self, keypoint_sets, anns):
        """Values in the same order and length as keypoint_sets."""
        raise NotImplementedError
    
    def all_fill_values_cafdet(self, detections, anns):
        raise NotImplementedError

    def fill_field_values(self, field_i, fij, fill_values):
        raise NotImplementedError

    def fields_as_tensor(self, valid_area):
        raise NotImplementedError

    def __call__(self, image, anns, meta):
        if self.config.meta.dataset == 'cocokp':
            width_height_original = image.shape[2:0:-1]

            keypoint_sets = self.rescaler.keypoint_sets(anns)
            bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                            crowd_margin=(self.config.min_size - 1) / 2)
            self.field_shape = (
                self.config.meta.n_fields,
                bg_mask.shape[0] + 2 * self.config.padding,
                bg_mask.shape[1] + 2 * self.config.padding,
            )
            valid_area = self.rescaler.valid_area(meta)
            LOG.debug('valid area: %s', valid_area)

            self.init_fields(bg_mask)
            self.fields_reg_l = np.full(self.field_shape, np.inf, dtype=np.float32)
            p = self.config.padding
            self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0

            fill_values = self.all_fill_values(keypoint_sets, anns) #returns a list of [(keypoints at the corresponding stride of an instance, scale of the visiable part of that instance)]
            self.fill(keypoint_sets, fill_values)
            fields = self.fields_as_tensor(valid_area)

            self.visualizer.processed_image(image)
            self.visualizer.targets(fields, annotation_dicts=anns)
        else:
            width_height_original = image.shape[2:0:-1]
            detections = self.rescaler.cif_detections(anns)
            bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.min_size - 1) / 2)
            #TODO: figure out field shape
            n_fields = self.config.meta.n_fields
            self.field_shape = (
                n_fields,
                bg_mask.shape[0] + 2 * self.config.padding,
                bg_mask.shape[1] + 2 * self.config.padding,
            )
            valid_area = self.rescaler.valid_area(meta)
            LOG.debug('valid area: %s', valid_area)

            self.init_cafdet_fields(bg_mask)
            self.fields_reg_l = [np.full(self.field_shape, np.inf, dtype=np.float32) for _ in range(len(self.config.meta.categories))]
            p = self.config.padding
            for i in range(len(self.config.meta.categories)):
                self.fields_reg_l[i][:, p:-p, p:-p][:, bg_mask == 0] = 1.0
            cafdet_fill_values = self.all_fill_values_cafdet(detections, anns)
            self.fill_cafdet(detections, cafdet_fill_values)
            fields = [self.cafdet_fields_as_tensor(valid_area, i) for i in range(len(self.config.meta.categories))]

            #TODO: visualize the fields
        return fields

    def fill(self, keypoint_sets, fill_values):
        for keypoints, fill_value in zip(keypoint_sets, fill_values):
            self.fill_keypoints(keypoints, fill_value)

    def fill_cafdet(self, detections, cafdet_fill_values):
        for detection, fill_value in zip(detections, cafdet_fill_values):
            self.fill_caf_detections(detection, fill_value) #detection contain (category_id, [center,top_left,bottom_right]) of one object

    def shortest_sparse(self, joint_i, keypoints):
        shortest = np.inf
        for joint1i, joint2i in self.sparse_skeleton_m1:
            if joint_i not in (joint1i, joint2i):
                continue

            joint1 = keypoints[joint1i]
            joint2 = keypoints[joint2i]
            if joint1[2] <= self.config.v_threshold or joint2[2] <= self.config.v_threshold:
                continue

            d = np.linalg.norm(joint1[:2] - joint2[:2])
            shortest = min(d, shortest)

        return shortest

    def fill_keypoints(self, keypoints, fill_values):
        for field_i, joint1i, joint2i in self.config.fill_plan:
            joint1 = keypoints[joint1i] #keypoints coordinate of the first joint connected by the i_th joint connection
            joint2 = keypoints[joint2i] #keypoints coordinate of the second joint connected by the i_th joint connection
            #check if the joint is visible
            if joint1[2] <= self.config.v_threshold or joint2[2] <= self.config.v_threshold:
                continue

            # check if there are shorter connections in the sparse skeleton
            if self.sparse_skeleton_m1 is not None:
                d = (np.linalg.norm(joint1[:2] - joint2[:2])
                     / self.config.meta.dense_to_sparse_radius)
                if self.shortest_sparse(joint1i, keypoints) < d \
                   and self.shortest_sparse(joint2i, keypoints) < d:
                    continue

            # if there is no continuous visual connection, endpoints outside
            # the field of view cannot be inferred
            # LOG.debug('fov check: j1 = %s, j2 = %s', joint1, joint2)
            out_field_of_view_1 = (
                joint1[0] < 0
                or joint1[1] < 0
                or joint1[0] > self.field_shape[2] - 1 - 2 * self.config.padding
                or joint1[1] > self.field_shape[1] - 1 - 2 * self.config.padding
            )
            out_field_of_view_2 = (
                joint2[0] < 0
                or joint2[1] < 0
                or joint2[0] > self.field_shape[2] - 1 - 2 * self.config.padding
                or joint2[1] > self.field_shape[1] - 1 - 2 * self.config.padding
            )
            if out_field_of_view_1 and out_field_of_view_2:
                continue
            if self.config.meta.only_in_field_of_view:
                if out_field_of_view_1 or out_field_of_view_2:
                    continue

            self.fill_association(field_i, joint1, joint2, fill_values)

    def fill_caf_detections(self, detection, fill_value):
        category_id, bbox = detection
        cafdet_keypoints = [np.asarray([bbox[2*i],bbox[2*i+1]]) for i in range(self.config.meta.n_fields)]

        for field_i, joint1i, joint2i in self.config.fill_plan:
            joint1 = cafdet_keypoints[joint1i] #keypoints coordinate of the first joint connected by the i_th joint connection
            joint2 = cafdet_keypoints[joint2i] #keypoints coordinate of the second joint connected by the i_th joint connection
            #check if the joint is visible
            # if joint1[2] <= self.config.v_threshold or joint2[2] <= self.config.v_threshold:
            #     continue

            # check if there are shorter connections in the sparse skeleton
            if self.sparse_skeleton_m1 is not None:
                d = (np.linalg.norm(joint1[:2] - joint2[:2])
                     / self.config.meta.dense_to_sparse_radius)
                if self.shortest_sparse(joint1i, cafdet_keypoints) < d \
                   and self.shortest_sparse(joint2i, cafdet_keypoints) < d:
                    continue

            # if there is no continuous visual connection, endpoints outside
            # the field of view cannot be inferred
            # LOG.debug('fov check: j1 = %s, j2 = %s', joint1, joint2)
            out_field_of_view_1 = (
                joint1[0] < 0
                or joint1[1] < 0
                or joint1[0] > self.field_shape[2] - 1 - 2 * self.config.padding
                or joint1[1] > self.field_shape[1] - 1 - 2 * self.config.padding
            )
            out_field_of_view_2 = (
                joint2[0] < 0
                or joint2[1] < 0
                or joint2[0] > self.field_shape[2] - 1 - 2 * self.config.padding
                or joint2[1] > self.field_shape[1] - 1 - 2 * self.config.padding
            )
            if out_field_of_view_1 and out_field_of_view_2:
                continue
            if self.config.meta.only_in_field_of_view:
                if out_field_of_view_1 or out_field_of_view_2:
                    continue

            self.fill_cafdet_association(field_i, joint1, joint2, fill_value, category_id)

    def fill_association(self, field_i, joint1, joint2, fill_values):
        # offset between joints
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = max(self.config.min_size, int(offset_d * self.config.aspect_ratio))

        # meshgrid: coordinate matrix
        xyv = np.stack(np.meshgrid(
            np.linspace(-0.5 * (s - 1), 0.5 * (s - 1), s),
            np.linspace(-0.5 * (s - 1), 0.5 * (s - 1), s),
        ), axis=-1).reshape(-1, 2)   #creat a grid for each point along the connection between two joints

        # set fields
        num = max(2, int(np.ceil(offset_d))) #number of points that we need do construct paf along the connection between two joints
        fmargin = (s / 2) / (offset_d + np.spacing(1))
        fmargin = np.clip(fmargin, 0.25, 0.4)
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0 - fmargin, num=num) #the fraction of the connection between two joints that we need to construct paf
        if self.config.fixed_size:
            frange = [0.5]
        filled_ij = set()
        for f in frange:
            for xyo in xyv:
                fij = np.round(joint1[:2] + f * offset + xyo).astype(np.intc) + self.config.padding   
                #adding one fraction at a time to the first joint coordinate to get the coordinate of the point that we need to construc paf(fij) along the connection
                #+xyo allows us to estimate the size of the connection between two joints
                if fij[0] < 0 or fij[0] >= self.field_shape[2] or \
                   fij[1] < 0 or fij[1] >= self.field_shape[1]:
                    continue

                # convert to hashable coordinate and check whether
                # it was processed before
                fij_int = (int(fij[0]), int(fij[1]))
                if fij_int in filled_ij:
                    continue
                filled_ij.add(fij_int)

                # mask
                # perpendicular distance computation:
                # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
                # Coordinate systems for this computation is such that
                # joint1 is at (0, 0).
                fxy = fij - self.config.padding
                f_offset = fxy - joint1[:2]
                sink_l = np.fabs(
                    offset[1] * f_offset[0]
                    - offset[0] * f_offset[1]
                ) / (offset_d + 0.01) #sunk_l is the projection length of f_offset on the connection between two joints which is offset. (A /dot B / ||B|| = the projection length of A on B)
                if sink_l > self.fields_reg_l[field_i, fij[1], fij[0]]:
                    continue
                self.fields_reg_l[field_i, fij[1], fij[0]] = sink_l

                self.fill_field_values(field_i, fij, fill_values)

    def fill_cafdet_association(self, field_i, joint1, joint2, fill_value, category_id):
        # offset between joints
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = max(self.config.min_size, int(offset_d * self.config.aspect_ratio))

        # meshgrid: coordinate matrix
        xyv = np.stack(np.meshgrid(
            np.linspace(-0.5 * (s - 1), 0.5 * (s - 1), s),
            np.linspace(-0.5 * (s - 1), 0.5 * (s - 1), s),
        ), axis=-1).reshape(-1, 2)   #creat a grid for each point along the connection between two joints

        # set fields
        num = max(2, int(np.ceil(offset_d))) #number of points that we need do construct paf along the connection between two joints
        fmargin = (s / 2) / (offset_d + np.spacing(1))
        fmargin = np.clip(fmargin, 0.25, 0.4)
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0 - fmargin, num=num) #the fraction of the connection between two joints that we need to construct paf
        if self.config.fixed_size:
            frange = [0.5]
        filled_ij = set()
        for f in frange:
            for xyo in xyv:
                fij = np.round(joint1[:2] + f * offset + xyo).astype(np.intc) + self.config.padding   
                #adding one fraction at a time to the first joint coordinate to get the coordinate of the point that we need to construc paf(fij) along the connection
                #+xyo allows us to estimate the size of the connection between two joints
                if fij[0] < 0 or fij[0] >= self.field_shape[2] or \
                   fij[1] < 0 or fij[1] >= self.field_shape[1]:
                    continue

                # convert to hashable coordinate and check whether
                # it was processed before
                fij_int = (int(fij[0]), int(fij[1]))
                if fij_int in filled_ij:
                    continue
                filled_ij.add(fij_int)

                # mask
                # perpendicular distance computation:
                # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
                # Coordinate systems for this computation is such that
                # joint1 is at (0, 0).
                fxy = fij - self.config.padding
                f_offset = fxy - joint1[:2]
                sink_l = np.fabs(
                    offset[1] * f_offset[0]
                    - offset[0] * f_offset[1]
                ) / (offset_d + 0.01) #sunk_l is the projection length of f_offset on the connection between two joints which is offset. (A /dot B / ||B|| = the projection length of A on B)
                if sink_l > self.fields_reg_l[category_id-1][field_i, fij[1], fij[0]]:
                    continue
                self.fields_reg_l[category_id-1][field_i, fij[1], fij[0]] = sink_l

                self.fill_cafdet_field_values(field_i, fij, fill_value, category_id)


class CafGenerator(AssociationFiller):
    def __init__(self, config: Caf):
        super().__init__(config)

        self.skeleton_m1 = np.asarray(config.meta.skeleton) - 1

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_bmin1 = None
        self.fields_bmin2 = None
        self.fields_scale1 = None
        self.fields_scale2 = None

    def init_fields(self, bg_mask):
        reg_field_shape = (self.field_shape[0], 2, self.field_shape[1], self.field_shape[2])

        self.intensities = np.zeros(self.field_shape, dtype=np.float32)
        self.fields_reg1 = np.full(reg_field_shape, np.nan, dtype=np.float32)
        self.fields_reg2 = np.full(reg_field_shape, np.nan, dtype=np.float32)
        self.fields_bmin1 = np.full(self.field_shape, np.nan, dtype=np.float32)
        self.fields_bmin2 = np.full(self.field_shape, np.nan, dtype=np.float32)
        self.fields_scale1 = np.full(self.field_shape, np.nan, dtype=np.float32)
        self.fields_scale2 = np.full(self.field_shape, np.nan, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan

    def init_cafdet_fields(self, bg_mask):
        reg_field_shape = (self.field_shape[0], 2, self.field_shape[1], self.field_shape[2])

        self.intensities = [np.zeros(self.field_shape, dtype=np.float32) for _ in range(len(self.config.meta.categories))]
        self.fields_reg1 = [np.full(reg_field_shape, np.nan, dtype=np.float32) for _ in range(len(self.config.meta.categories))]
        self.fields_reg2 = [np.full(reg_field_shape, np.nan, dtype=np.float32) for _ in range(len(self.config.meta.categories))]
        self.fields_bmin1 = [np.full(self.field_shape, np.nan, dtype=np.float32) for _ in range(len(self.config.meta.categories))]
        self.fields_bmin2 = [np.full(self.field_shape, np.nan, dtype=np.float32) for _ in range(len(self.config.meta.categories))]
        self.fields_scale1 = [np.full(self.field_shape, np.nan, dtype=np.float32) for _ in range(len(self.config.meta.categories))]
        self.fields_scale2 = [np.full(self.field_shape, np.nan, dtype=np.float32) for _ in range(len(self.config.meta.categories))]

        # bg_mask
        p = self.config.padding
        for i in range(len(self.config.meta.categories)):
            self.intensities[i][:, p:-p, p:-p][:, bg_mask == 0] = np.nan


    def all_fill_values(self, keypoint_sets, anns):
        return [(kps, self.rescaler.scale(kps)) for kps in keypoint_sets]
    
    def all_fill_values_cafdet(self, detections, anns):
        return [(detection, self.rescaler.detections_scale(detection)) for detection in detections]

    def fill_field_values(self, field_i, fij, fill_values):
        joint1i, joint2i = self.skeleton_m1[field_i]
        keypoints, scale = fill_values

        if self.config.use_fpn:
            # if (self.config.head_index == 0 and scale>8) or (self.config.head_index == -1 and scale<=4)\
            #     or (self.config.head_index != 0 and self.config.head_index != -1 and (scale>8 or scale<=4)):
            #     return

            # if (self.config.head_index == 0 and scale>16) or (self.config.head_index == -1 and scale<=8)\
            #     or (self.config.head_index != 0 and self.config.head_index != -1 and (scale>16 or scale<=8)):
            #     return
            
            if (self.config.head_index == 0 and scale>8/self.config.meta.upsample_stride) or (self.config.head_index == -1 and scale<=4/self.config.meta.upsample_stride)\
                or (self.config.head_index != 0 and self.config.head_index != -1 and (8/self.config.meta.upsample_stride or scale<=4/self.config.meta.upsample_stride)):
                # print('ignore')
                return
        # update intensity
        self.intensities[field_i, fij[1], fij[0]] = 1.0

        # update regressions
        fxy = fij - self.config.padding
        self.fields_reg1[field_i, :, fij[1], fij[0]] = keypoints[joint1i][:2] - fxy
        self.fields_reg2[field_i, :, fij[1], fij[0]] = keypoints[joint2i][:2] - fxy

        # update bmin
        bmin = self.config.bmin / self.config.meta.stride
        self.fields_bmin1[field_i, fij[1], fij[0]] = bmin
        self.fields_bmin2[field_i, fij[1], fij[0]] = bmin

        # update scale
        if self.config.meta.sigmas is None:
            scale1, scale2 = scale, scale
        else:
            scale1 = scale * self.config.meta.sigmas[joint1i]
            scale2 = scale * self.config.meta.sigmas[joint2i]

        # neglect very large instance
        # if scale1>=100:
        #     scale1=np.nan
        # if scale2>=100:
        #     scale2=np.nan

    
        assert np.isnan(scale1) or 0.0 < scale1 < 100.0
        self.fields_scale1[field_i, fij[1], fij[0]] = scale1
        assert np.isnan(scale2) or 0.0 < scale2 < 100.0
        self.fields_scale2[field_i, fij[1], fij[0]] = scale2

    def fill_cafdet_field_values(self, field_i, fij, fill_value):

        joint1i, joint2i = self.skeleton_m1[field_i]
        detection, scale = fill_value

        category_id, bbox = detection
        cafdet_keypoints = [np.asarray([bbox[2*i],bbox[2*i+1]]) for i in range(self.config.meta.n_fields)]

        if self.config.use_fpn:
            # if (self.config.head_index == 0 and scale>8) or (self.config.head_index == -1 and scale<=4)\
            #     or (self.config.head_index != 0 and self.config.head_index != -1 and (scale>8 or scale<=4)):
            #     return

            # if (self.config.head_index == 0 and scale>16) or (self.config.head_index == -1 and scale<=8)\
            #     or (self.config.head_index != 0 and self.config.head_index != -1 and (scale>16 or scale<=8)):
            #     return
            
            if (self.config.head_index == 0 and scale>8/self.config.meta.upsample_stride) or (self.config.head_index == -1 and scale<=4/self.config.meta.upsample_stride)\
                or (self.config.head_index != 0 and self.config.head_index != -1 and (8/self.config.meta.upsample_stride or scale<=4/self.config.meta.upsample_stride)):
                # print('ignore')
                return
        # update intensity
        self.intensities[category_id-1][field_i, fij[1], fij[0]] = 1.0

        # update regressions
        fxy = fij - self.config.padding
        self.fields_reg1[category_id-1][field_i, :, fij[1], fij[0]] = cafdet_keypoints[joint1i][:2] - fxy
        self.fields_reg2[category_id-1][field_i, :, fij[1], fij[0]] = cafdet_keypoints[joint2i][:2] - fxy

        # update bmin
        bmin = self.config.bmin / self.config.meta.stride
        self.fields_bmin1[category_id-1][field_i, fij[1], fij[0]] = bmin
        self.fields_bmin2[category_id-1][field_i, fij[1], fij[0]] = bmin

        # update scale
        if self.config.meta.sigmas is None:
            scale1, scale2 = scale, scale
        else:
            scale1 = scale * self.config.meta.sigmas[joint1i]
            scale2 = scale * self.config.meta.sigmas[joint2i]

        # neglect very large instance
        # if scale1>=100:
        #     scale1=np.nan
        # if scale2>=100:
        #     scale2=np.nan

    
        assert np.isnan(scale1) or 0.0 < scale1 < 100.0
        self.fields_scale1[category_id-1][field_i, fij[1], fij[0]] = scale1
        assert np.isnan(scale2) or 0.0 < scale2 < 100.0
        self.fields_scale2[category_id-1][field_i, fij[1], fij[0]] = scale2


    def fields_as_tensor(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg1 = self.fields_reg1[:, :, p:-p, p:-p]
        fields_reg2 = self.fields_reg2[:, :, p:-p, p:-p]
        fields_bmin1 = self.fields_bmin1[:, p:-p, p:-p]
        fields_bmin2 = self.fields_bmin2[:, p:-p, p:-p]
        fields_scale1 = self.fields_scale1[:, p:-p, p:-p]
        fields_scale2 = self.fields_scale2[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg1[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg1[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin2, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale2, valid_area, fill_value=np.nan)

        # print('caf fields reg is nan: {}'.format(np.isnan(fields_reg1).all()))

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg1,
            fields_reg2,
            np.expand_dims(fields_bmin1, 1),
            np.expand_dims(fields_bmin2, 1),
            np.expand_dims(fields_scale1, 1),
            np.expand_dims(fields_scale2, 1),
        ], axis=1))

    def cafdet_fields_as_tensor(self, valid_area, c_id):
        #c_id iterate in range(len(categories)) so we don't need to -1
        p = self.config.padding
        intensities = self.intensities[c_id][:, p:-p, p:-p]
        fields_reg1 = self.fields_reg1[c_id][:, :, p:-p, p:-p]
        fields_reg2 = self.fields_reg2[c_id][:, :, p:-p, p:-p]
        fields_bmin1 = self.fields_bmin1[c_id][:, p:-p, p:-p]
        fields_bmin2 = self.fields_bmin2[c_id][:, p:-p, p:-p]
        fields_scale1 = self.fields_scale1[c_id][:, p:-p, p:-p]
        fields_scale2 = self.fields_scale2[c_id][:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg1[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg1[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg2[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin2, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale1, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale2, valid_area, fill_value=np.nan)

        # print('caf fields reg is nan: {}'.format(np.isnan(fields_reg1).all()))

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg1,
            fields_reg2,
            np.expand_dims(fields_bmin1, 1),
            np.expand_dims(fields_bmin2, 1),
            np.expand_dims(fields_scale1, 1),
            np.expand_dims(fields_scale2, 1),
        ], axis=1))
