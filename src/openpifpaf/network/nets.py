import logging
import torch
from functools import partial


LOG = logging.getLogger(__name__)


def multi_apply(func, *args, **kwargs):
    #When *args is used as an argument to a function, it tells Python to unpack the contents of the tuple and pass them as separate positional arguments to the function.
    #In this case, if args is a tuple containing multiple arguments, and the * operator is used to unpack this tuple into separate arguments that are passed to the map() function.
    #map function will apply pfunc to each element unpacked from the args(eg. if args = (op1,op2,op3), map will apply pfunc to each of the op and return a iterator)
    #zip(*map_results) will first unpack all the map_results and then zip the corresponding element together(eg. map_results = [[1,2],[3,4],[5,6]], zip(*map_results) will be (1,3,5),(2,4,6))
    #Then map(list, zip(*map_results)) will map these zipped elements to list type(eg.(1,3,5),(2,4,6)-->[1,3,5],[2,4,6])
    #Finally, these lists are put in a tuple
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

class Shell(torch.nn.Module):
    #By using the * syntax, any arguments that come after it must be specified as keyword arguments. 
    #In the case of the Shell class, this means that the process_input and process_heads arguments can only be passed as named parameters
    #like: net = Shell(base_net, head_nets, process_input=my_input_processor, process_heads=my_head_processor)
    def __init__(self, base_net, head_nets, neck_net = None, *,
                 process_input=None, process_heads=None):
        super().__init__()

        self.base_net = base_net
        self.neck_net = neck_net
        self.head_nets = None
        self.process_input = process_input
        self.process_heads = process_heads

        self.set_head_nets(head_nets)

    @property
    def head_metas(self):
        if self.head_nets is None:
            return None
        return [hn.meta for hn in self.head_nets]

    def set_head_nets(self, head_nets):
        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)

        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.base_net.stride


        self.head_nets = head_nets

    def forward(self, image_batch, *, head_mask=None):
        if self.process_input is not None:
            image_batch = self.process_input(image_batch)

        x = self.base_net(image_batch)

        if self.neck_net is not None:
            assert type(x) == tuple
            x = self.neck_net(x) ##now x becomes a tuple(the outputs from different stage of the FPN in a bottom-up!!! fasion)
            # print(type(head_mask),type(head_mask[0]))
            # assert type(head_mask[0]) == list
            if head_mask is not None:
                # head_outputs = tuple(multi_apply(hn,x) if m else None for hn, m in zip(self.head_nets, head_mask[0]))  
                ###we don't wanna the output to be in the form tuple(Cif(stage1,stage2,...),Caf(stage1,stage2,...),...)
                head_outputs = tuple(tuple(hn(x_single) if m else None for hn, m in zip(self.head_nets, head_mask)) for x_single in x)
                ###Rather we wanna the output to be in the form tuple(tuple(Cif(stage1),Caf(stage1),...),tuple(Cif(stage2),Caf(stage2),...),...)
                #Plus, now head_mask if a list of the original head_mask without fpn, but each element in this list contains the same heads, so we simply use head_mask[0]
            else:
                # head_outputs = tuple(multi_apply(hn,x) for hn in self.head_nets)
                head_outputs = tuple(tuple(hn(x_single) for hn in self.head_nets) for x_single in x )

            if self.process_heads is not None:
                # head_outputs = self.process_heads(head_outputs)
                head_outputs = multi_apply(self.process_heads,head_outputs)
        else:
            if head_mask is not None:
                head_outputs = tuple(hn(x) if m else None for hn, m in zip(self.head_nets, head_mask))
            else:
                head_outputs = tuple(hn(x) for hn in self.head_nets)

            if self.process_heads is not None:
                head_outputs = self.process_heads(head_outputs)


        return head_outputs


class CrossTalk(torch.nn.Module):
    def __init__(self, strength=0.2):
        super().__init__()
        self.strength = strength

    def forward(self, image_batch):
        if self.training and self.strength:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk
        return image_batch


def model_defaults(net_cpu):
    for m in net_cpu.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # avoid numerical instabilities
            # (only seen sometimes when training with GPU)
            # Variances in pretrained models can be as low as 1e-17.
            # m.running_var.clamp_(min=1e-8)
            # m.eps = 1e-3  # tf default is 0.001
            # m.eps = 1e-5  # pytorch default

            # This epsilon only appears inside a sqrt in the denominator,
            # i.e. the effective epsilon for division is much bigger than the
            # given eps.
            # See equation here:
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            m.eps = max(m.eps, 1e-3)  # mobilenetv3 actually has 1e-3

            # smaller step size for running std and mean update
            m.momentum = 0.01  # tf default is 0.99
            # m.momentum = 0.1  # pytorch default

        elif isinstance(m, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            m.eps = 1e-4

        elif isinstance(m, (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d)):
            m.eps = 1e-4
            m.momentum = 0.01
