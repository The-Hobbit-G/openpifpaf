# Modification 2020 RangiLyu
# Copyright 2018-2019 Open-MMLab.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.functional as F

from ..module.conv import ConvModule
from ..module.init_weights import xavier_init


class FPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        activation=None,
    ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        ##This implementation is from mmdetection where the end_level is included in the output, max(end_level) == self.num_ins-1
        # if end_level == -1 or end_level == self.num_ins-1:
        #     self.backbone_end_level = self.num_ins
        #     assert num_outs >= self.num_ins - start_level
        # else:
        #     # if end_level < inputs, no extra level is allowed
        #     self.backbone_end_level = end_level + 1
        #     assert end_level <= len(in_channels)
        #     assert num_outs == end_level - start_level + 1

        ##Our current implementation is from nanodet where the end_level is not included in the output, max(end_level) == self.num_ins
        ##The output of basenet will be formulated based on this version
        if end_level == -1 or end_level == self.num_ins:
            self.backbone_end_level = self.num_ins
            end_level = self.num_ins  #enable the modified version to work for the original version and the swin case where the last level is not used
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            # self.backbone_end_level = end_level
            self.backbone_end_level = self.num_ins #do lateral till the last level to avoid the problem of the last level not used
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'


        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=activation,
                inplace=False,
            )

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)


        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
         
        # for lateral in laterals:
        #     print(lateral.shape)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # laterals[i - 1] += F.interpolate(
            #     laterals[i], scale_factor=2, mode="bilinear"
            # )
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i-1].size()[2:], mode="bilinear"
            )

        # build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            # laterals[i]
            # for i in range(used_backbone_levels)
        ]

        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(self.start_level, self.end_level)
        #     # use only the selected levels and the remaining levels are for lateral connection and top-down path
        # ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)


# if __name__ == '__main__':
