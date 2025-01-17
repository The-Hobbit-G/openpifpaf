"""Train a neural net."""

import argparse
import copy
import hashlib
import logging
import shutil
import time
import psutil
import resource
import cv2
import numpy as np

import torch

from ..profiler import TorchProfiler
from .nets import multi_apply
from . import nets

LOG = logging.getLogger(__name__)






class Trainer():
    epochs = None
    n_train_batches = None
    n_val_batches = None

    clip_grad_norm = 0.0
    clip_grad_value = 0.0
    log_interval = 11
    val_interval = 1

    fix_batch_norm = False
    stride_apply = 1
    ema_decay = 0.01
    train_profile = None
    distributed_reduce_loss = True

    def __init__(self, model, loss, optimizer, out, *,
                 checkpoint_shell=None,
                 lr_scheduler=None,
                 device=None,
                 model_meta_data=None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.out = out
        self.checkpoint_shell = checkpoint_shell
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_meta_data = model_meta_data

        self.ema = None
        self.ema_restore_params = None

        self.n_clipped_grad = 0
        self.max_norm = 0.0

        if self.train_profile and (not torch.distributed.is_initialized()
                                   or torch.distributed.get_rank() == 0):
            # monkey patch to profile self.train_batch()
            self.train_batch = TorchProfiler(self.train_batch, out_name=self.train_profile)

        LOG.info({
            'type': 'config',
            'field_names': self.loss.field_names,
        })

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('trainer')
        group.add_argument('--epochs', type=int,
                           help='number of epochs to train')
        group.add_argument('--train-batches', default=None, type=int,
                           help='number of train batches')
        group.add_argument('--val-batches', default=None, type=int,
                           help='number of val batches')

        group.add_argument('--clip-grad-norm', default=cls.clip_grad_norm, type=float,
                           help='clip grad norm: specify largest change for single param')
        group.add_argument('--clip-grad-value', default=cls.clip_grad_value, type=float,
                           help='clip grad value: specify largest change for single param')
        group.add_argument('--log-interval', default=cls.log_interval, type=int,
                           help='log loss every n steps')
        group.add_argument('--val-interval', default=cls.val_interval, type=int,
                           help='validation run every n epochs')

        group.add_argument('--stride-apply', default=cls.stride_apply, type=int,
                           help='apply and reset gradients every n batches')
        assert not cls.fix_batch_norm
        group.add_argument('--fix-batch-norm',
                           default=False, const=True, type=int, nargs='?',
                           help='fix batch norm running statistics (optionally specify epoch)')
        group.add_argument('--ema', default=cls.ema_decay, type=float,
                           help='ema decay constant')
        group.add_argument('--profile', default=cls.train_profile,
                           help='enables profiling. specify path for chrome tracing file')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.epochs = args.epochs
        cls.n_train_batches = args.train_batches
        cls.n_val_batches = args.val_batches

        cls.clip_grad_norm = args.clip_grad_norm
        cls.clip_grad_value = args.clip_grad_value
        cls.log_interval = args.log_interval
        cls.val_interval = args.val_interval

        cls.fix_batch_norm = args.fix_batch_norm
        cls.stride_apply = args.stride_apply
        cls.ema_decay = args.ema
        cls.train_profile = args.profile

    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step_ema(self):
        if self.ema is None:
            return

        for p, ema_p in zip(self.model.parameters(), self.ema):
            ema_p.mul_(1.0 - self.ema_decay).add_(p.data, alpha=self.ema_decay)

    def apply_ema(self):
        if self.ema is None:
            return

        LOG.info('applying ema')
        self.ema_restore_params = copy.deepcopy(
            [p.data for p in self.model.parameters()])
        for p, ema_p in zip(self.model.parameters(), self.ema):
            p.data.copy_(ema_p)

    def ema_restore(self):
        if self.ema_restore_params is None:
            return

        LOG.info('restoring params from before ema')
        for p, ema_p in zip(self.model.parameters(), self.ema_restore_params):
            p.data.copy_(ema_p)
        self.ema_restore_params = None

    def loop(self,
             train_scenes: torch.utils.data.DataLoader,
             val_scenes: torch.utils.data.DataLoader,
             start_epoch=0):
        if start_epoch >= self.epochs:
            raise Exception('start epoch ({}) >= total epochs ({})'
                            ''.format(start_epoch, self.epochs))

        if self.lr_scheduler is not None:
            assert self.lr_scheduler.last_epoch == start_epoch * len(train_scenes)

        for epoch in range(start_epoch, self.epochs):
            if epoch == 0:
                self.write_model(0, final=False)
            if hasattr(train_scenes.sampler, 'set_epoch'):
                train_scenes.sampler.set_epoch(epoch)
            if hasattr(val_scenes.sampler, 'set_epoch'):
                val_scenes.sampler.set_epoch(epoch)


            #TODO: a temp test for val epoch(will be removed after debugging)
            # if epoch == 0:
            #     self.val(val_scenes, epoch+1)


            self.train(train_scenes, epoch)

            #evaluate the model after the training is done or at specified epochs
            if (epoch + 1) % self.val_interval == 0 \
               or epoch + 1 == self.epochs:
                self.write_model(epoch + 1, epoch + 1 == self.epochs)
                self.val(val_scenes, epoch + 1)

    # pylint: disable=method-hidden,too-many-branches,too-many-statements
    def train_batch(self, data, targets, apply_gradients=True):
        if self.device.type != 'cpu':
            # print(type(data))
            assert data.is_pinned(), 'input data must be pinned'
            if targets[0] is not None:
                # assert targets[0].is_pinned(), 'input targets must be pinned'
                for target in targets[0]:
                    if target is not None:
                        assert target.is_pinned()

            # print(type(targets),type(targets[0]),type(targets[1]),type(targets[2]))
            # print(type(targets),type(targets[0]),type(targets[1]))
            # print(len(targets),len(targets[0]),len(targets[1]))
            # print(type(data),len(data))
            
            #for FPN case

            # target_formulation_start = time.time()


            if type(targets[0]) == list and ((isinstance(self.model,nets.Shell) and self.model.neck_net is not None) or \
                                             (isinstance(self.model,torch.nn.parallel.DistributedDataParallel) and self.model.module.neck_net is not None)):
                with torch.autograd.profiler.record_function('to-device'):
                    data = data.to(self.device, non_blocking=True)
                    targets = tuple([[head.to(self.device, non_blocking=True)
                            if head is not None else None
                            for head in target] if target is not None else None for target in targets]) #now the target would be tuple([tensors,...],[tensors,...],...)
            #for cifcafdet case
            elif type(targets[0]) == list and ((isinstance(self.model,nets.Shell) and self.model.neck_net is None) or \
                                             (isinstance(self.model,torch.nn.parallel.DistributedDataParallel) and self.model.module.neck_net is None)):
                with torch.autograd.profiler.record_function('to-device'):
                    data = data.to(self.device, non_blocking=True)
                    assert (len(targets) == 2) and (len(targets[0]) == len(targets[1]))
                    #concatenate all cifdet targets and all cafdet targets at dim=1 dimention into two tensors and put them in self.device
                    targets = tuple([torch.cat([targets[0][i] for i in range(len(targets[0])) if targets[0][i] is not None],dim=1).to(self.device, non_blocking=True),\
                                     torch.cat([targets[1][i] for i in range(len(targets[1])) if targets[1][i] is not None],dim=1).to(self.device, non_blocking=True)])  #cifdet targets,cafdet targets
                    
                    
                    # print(targets[0].size(),targets[1].size())
                    # print(targets[0][0].size(),targets[1][0].size())
                    # targets = tuple([[targets[0][i].to(self.device, non_blocking=True) if targets[0][i] is not None else None,\
                    #                   targets[1][i].to(self.device, non_blocking=True) if targets[1][i] is not None else None]\
                    #                     for i in range(len(targets[0]))])
                    #in this case, the new targets would be tuple([cifdet_target1,cafdet_target1],[cifdet_target2,cafdet_target2],...)
                    #concatenate all cifdet targets and all cafdet targets into two tensors
                    # targets = tuple([torch.cat([targets[0][i] for i in range(len(targets[0])) if targets[0][i] is not None],dim=0),\  #cifdet targets
                    #                  torch.cat([targets[1][i] for i in range(len(targets[1])) if targets[1][i] is not None],dim=0)])  #cafdet targets
            else:
                with torch.autograd.profiler.record_function('to-device'):
                    data = data.to(self.device, non_blocking=True)
                    targets = [head.to(self.device, non_blocking=True)
                            if head is not None else None
                            for head in targets]

            # target_time = time.time() - target_formulation_start
            # print('target formulation time: {}'.format(target_time))

        # train encoder
        with torch.autograd.profiler.record_function('model'):

            # output_start = time.time()
            # print(type(targets),type(targets[0]),len(targets),len(targets[0]))
            if type(targets) == tuple and ((isinstance(self.model,nets.Shell) and self.model.neck_net is not None) or \
                                             (isinstance(self.model,torch.nn.parallel.DistributedDataParallel) and self.model.module.neck_net is not None)):
                outputs = self.model(data, head_mask=[t is not None for t in targets[0]])
            else:
                outputs = self.model(data, head_mask=[t is not None for t in targets])

            # output_time = time.time() - output_start
            # print('output time: {}'.format(output_time))

            if self.train_profile and self.device.type != 'cpu':
                torch.cuda.synchronize() #torch.cuda.synchronize() is a function in the PyTorch deep learning library that allows you to synchronize the CPU with the GPU.
        with torch.autograd.profiler.record_function('loss'):
            # loss_start = time.time()
            # print('targets type: {}'.format(type(targets)))
            if type(targets) == tuple and ((isinstance(self.model,nets.Shell) and self.model.neck_net is not None) or \
                                             (isinstance(self.model,torch.nn.parallel.DistributedDataParallel) and self.model.module.neck_net is not None)):
                assert type(outputs[0]) == tuple
                assert len(targets) == len(outputs)
                ###check the shape of outputs and targets:
                '''cif target shape: (batch_size, num of keypoints=17, cif field channel=5, field_h, field_w)
                   caf target shape: (batch_size, num of association=19, caf field channel=9, field_h, field_w)
                   cif output shape: (batch_size, num of keypoints=17, cif field channel=5, field_h, field_w)
                   caf output shape: (batch_size, num of association=19, caf field channel-1=8, field_h, field_w)'''
                # for i in range(len(targets)):
                    # print('target len: {}, output len: {}'.format(len(targets[i]),len(outputs[i])))
                    # print('target shape: {} {}, output shape: {} {}'.format(targets[i][0].size(),targets[i][1].size(),\
                    #                                                         outputs[i][0].size(),outputs[i][1].size()))
                    # print('target type: {}, output type: {}'.format(type(targets[i][2]),type(outputs[i][2])))

                multistage_loss, multistage_head_losses = multi_apply(self.loss,outputs,targets)
                # print(len(targets[0]),len(outputs[0]))
                # print(len(targets), len(outputs),len(multistage_loss))
            
                # print('multistage_loss: {}, multistage_head_losses: {}'.format(multistage_loss,multistage_head_losses))
                # average over the losses from different stage(could also do sum)
                assert len(multistage_loss) == len(multistage_head_losses)
                loss = sum(multistage_loss)/len(multistage_loss)
                # head_losses = [sum([head_loss[i] for head_loss in multistage_head_losses])/len(multistage_head_losses) for i in range(len(multistage_head_losses[0]))]
                # head_losses = [sum([head_loss[i] for head_loss in multistage_head_losses])/len(multistage_head_losses) for i in range(len(multistage_head_losses[0]))]
                head_losses = [None] * len(multistage_head_losses[0])
                for i in range(len(head_losses)):
                    if multistage_head_losses[0][i] is not None:
                        head_losses[i] = sum([head_loss[i] for head_loss in multistage_head_losses])/len(multistage_head_losses)
                # loss = sum(head_losses)
            elif type(targets) == tuple and ((isinstance(self.model,nets.Shell) and self.model.neck_net is None) or \
                                             (isinstance(self.model,torch.nn.parallel.DistributedDataParallel) and self.model.module.neck_net is None)):
                # deal with the case where we use cifcaf detection head

                # multihead_start = time.time()
                assert type(outputs[0]) == tuple
                # print(len(outputs),len(outputs[0]),outputs[0][0].shape,outputs[0][1].shape)
                # assert len(targets) == len(outputs)

                #concatenate all the cifdet outpus and all the cafdet outputs into two tensors at dim = 1
                outputs = tuple([torch.cat([outputs[i][0] for i in range(len(outputs)) if outputs[i][0] is not None],dim=1),\
                                    torch.cat([outputs[i][1] for i in range(len(outputs)) if outputs[i][1] is not None],dim=1)])    #cifdet outputs,cafdet outputs

                '''
                multiclass_loss, multiclass_head_losses = multi_apply(self.loss,outputs,targets)
                #try using single apply(for loop replacing multi_apply)
                # multiclass_loss = []
                # multiclass_head_losses = []
                # for output, target in zip(outputs,targets):
                #     class_loss, class_head_loss = self.loss(output,target)
                #     multiclass_loss.append(class_loss)
                #     multiclass_head_losses.append(class_head_loss)
                assert len(multiclass_loss) == len(multiclass_head_losses)

                # multihead_time = time.time() - multihead_start
                # print('multihead loss calculation time: {}'.format(multihead_time))
                # print(type(multiclass_loss),type(multiclass_loss[0]),multiclass_loss)
                # max_loss = max(class_loss.item() for class_loss in multiclass_loss)
                # min_loss = min(class_loss.item() for class_loss in multiclass_loss)
                # print("Maximum Loss:", max_loss)
                # print("Minimum Loss:", min_loss)

                # category_loss_start = time.time()

                loss = sum(multiclass_loss)
                head_losses = [None] * len(multiclass_head_losses[0])
                for i in range(len(head_losses)):
                    if multiclass_head_losses[0][i] is not None:
                        head_losses[i] = sum([head_loss[i] for head_loss in multiclass_head_losses])
                 
                # category_loss_time = time.time() - category_loss_start
                # print('category loss calculation time: {}'.format(category_loss_time))
                '''
                # multihead_start = time.time()
                loss, head_losses = self.loss(outputs, targets)
                # multihead_time = time.time() - multihead_start
                # print('multihead loss calculation time: {}'.format(multihead_time))
            else:
                # print('target shape: {} {}, output shape: {} {}'.format(targets[0].size(),targets[1].size(),
                #                                                         outputs[0].size(),outputs[1].size()))
                loss, head_losses = self.loss(outputs, targets)
                # print('loss: {}, head_losses: {}'.format(loss,head_losses))

            # loss_time = time.time() - loss_start
            # print('loss time: {}'.format(loss_time))

            if self.train_profile and self.device.type != 'cpu':
                torch.cuda.synchronize()
        if loss is not None:
            # backprop_start = time.time()

            with torch.autograd.profiler.record_function('backward'):
                loss.backward()
                if self.train_profile and self.device.type != 'cpu':
                    torch.cuda.synchronize()
            
            # backprop_time = time.time() - backprop_start
            # print('backprop time: {}'.format(backprop_time))
        if self.clip_grad_norm:
            with torch.autograd.profiler.record_function('clip-grad-norm'):
                max_norm = self.clip_grad_norm / self.lr()
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm, norm_type=float('inf'))
                self.max_norm = max(float(total_norm), self.max_norm)
                if total_norm > max_norm:
                    self.n_clipped_grad += 1
                    print('CLIPPED GRAD NORM: total norm before clip: {}, max norm: {}'
                          ''.format(total_norm, max_norm))
                if self.train_profile and self.device.type != 'cpu':
                    torch.cuda.synchronize()
        if self.clip_grad_value:
            with torch.autograd.profiler.record_function('clip-grad-value'):
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
                if self.train_profile and self.device.type != 'cpu':
                    torch.cuda.synchronize()
        if apply_gradients:
            with torch.autograd.profiler.record_function('step'):
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.train_profile and self.device.type != 'cpu':
                    torch.cuda.synchronize()
            with torch.autograd.profiler.record_function('ema'):
                self.step_ema()
                if self.train_profile and self.device.type != 'cpu':
                    torch.cuda.synchronize()

        with torch.inference_mode():
            with torch.autograd.profiler.record_function('reduce-losses'):
                loss = self.reduce_loss(loss)
                head_losses = self.reduce_loss(head_losses)
                if self.train_profile and self.device.type != 'cpu':
                    torch.cuda.synchronize()

        return (
            float(loss.item()) if loss is not None else None,
            [float(l.item()) if l is not None else None
             for l in head_losses],
        )

    @classmethod
    def reduce_loss(cls, loss):
        if not cls.distributed_reduce_loss:
            return loss
        if loss is None:
            return loss
        if not torch.distributed.is_initialized():
            return loss

        if isinstance(loss, (list, tuple)):
            return [cls.reduce_loss(l) for l in loss]

        # average loss from all processes
        torch.distributed.reduce(loss, 0)
        if torch.distributed.get_rank() == 0:
            loss = loss / torch.distributed.get_world_size()
        return loss

    def val_batch(self, data, targets):
        if self.device:
            #for FPN case
            if type(targets[0]) == list and ((isinstance(self.model,nets.Shell) and self.model.neck_net is not None) or \
                                             (isinstance(self.model,torch.nn.parallel.DistributedDataParallel) and self.model.module.neck_net is not None)):
                data = data.to(self.device, non_blocking=True)
                targets = tuple([head.to(self.device, non_blocking=True)
                        if head is not None else None
                        for head in target] for target in targets) 
            #for cifcafdet case
            elif type(targets[0]) == list and ((isinstance(self.model,nets.Shell) and self.model.neck_net is None) or \
                                             (isinstance(self.model,torch.nn.parallel.DistributedDataParallel) and self.model.module.neck_net is None)):
                data = data.to(self.device, non_blocking=True)
                assert (len(targets) == 2) and (len(targets[0]) == len(targets[1]))
                # targets = tuple([[targets[0][i].to(self.device, non_blocking=True) if targets[0][i] is not None else None,\
                #                     targets[1][i].to(self.device, non_blocking=True) if targets[1][i] is not None else None]\
                #                     for i in range(len(targets[0]))])
                targets = tuple([torch.cat([targets[0][i] for i in range(len(targets[0])) if targets[0][i] is not None],dim=1).to(self.device, non_blocking=True),\
                                     torch.cat([targets[1][i] for i in range(len(targets[1])) if targets[1][i] is not None],dim=1).to(self.device, non_blocking=True)])  #cifdet targets,cafdet targets
            else:
                data = data.to(self.device, non_blocking=True)
                targets = [head.to(self.device, non_blocking=True)
                        if head is not None else None
                        for head in targets]

        with torch.inference_mode():
            outputs = self.model(data)

            if type(targets) == tuple and ((isinstance(self.model,nets.Shell) and self.model.neck_net is not None) or \
                                             (isinstance(self.model,torch.nn.parallel.DistributedDataParallel) and self.model.module.neck_net is not None)):
                assert type(outputs[0]) == tuple
                assert len(targets) == len(outputs)
                multistage_loss, multistage_head_losses = multi_apply(self.loss,outputs,targets)
                # average over the losses from different stage(could also do sum)
                assert len(multistage_loss) == len(multistage_head_losses)
                loss = sum(multistage_loss)/len(multistage_loss)
                # head_losses = [sum([head_loss[i] for head_loss in multistage_head_losses])/len(multistage_head_losses) for i in range(len(multistage_head_losses[0]))]
                head_losses = [None] * len(multistage_head_losses[0])
                for i in range(len(head_losses)):
                    if multistage_head_losses[0][i] is not None:
                        head_losses[i] = sum([head_loss[i] for head_loss in multistage_head_losses])/len(multistage_head_losses)
            elif type(targets) == tuple and ((isinstance(self.model,nets.Shell) and self.model.neck_net is None) or \
                                             (isinstance(self.model,torch.nn.parallel.DistributedDataParallel) and self.model.module.neck_net is None)):
                # deal with the case where we use cifcaf detection head
                assert type(outputs[0]) == tuple
                # print(len(outputs),len(outputs[0]),outputs[0][0].shape,outputs[0][1].shape)
                outputs = tuple([torch.cat([outputs[i][0] for i in range(len(outputs)) if outputs[i][0] is not None],dim=1),\
                                    torch.cat([outputs[i][1] for i in range(len(outputs)) if outputs[i][1] is not None],dim=1)])    #cifdet outputs,cafdet outputs
                loss, head_losses = self.loss(outputs, targets)
                '''
                assert len(targets) == len(outputs)
                multiclass_loss, multiclass_head_losses = multi_apply(self.loss,outputs,targets)
                assert len(multiclass_loss) == len(multiclass_head_losses)
                loss = sum(multiclass_loss)
                head_losses = [None] * len(multiclass_head_losses[0])
                for i in range(len(head_losses)):
                    if multiclass_head_losses[0][i] is not None:
                        head_losses[i] = sum([head_loss[i] for head_loss in multiclass_head_losses])
                '''
            else:
                loss, head_losses = self.loss(outputs, targets)
            loss = self.reduce_loss(loss)
            head_losses = self.reduce_loss(head_losses)

        return (
            float(loss.item()) if loss is not None else None,
            [float(l.item()) if l is not None else None
             for l in head_losses],
        )

    # pylint: disable=too-many-branches
    def train(self, scenes, epoch):
        start_time = time.time()
        self.model.train()
        if self.fix_batch_norm is True \
           or (self.fix_batch_norm is not False and self.fix_batch_norm <= epoch):
            LOG.info('fix batchnorm')
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    LOG.debug('eval mode for: %s', m)
                    m.eval()

        self.ema_restore()
        self.ema = None

        epoch_loss = 0.0
        head_epoch_losses = None
        head_epoch_counts = None
        last_batch_end = time.time()
        self.optimizer.zero_grad()
        for batch_idx, (data, target, _) in enumerate(scenes):

            #check annotations
            # print(type(meta))
            # print(meta)
            
            '''
            print(data)
            print(type(data),data.size(),data[0].size())
            print(meta[0]['dataset_index'],meta[0]['image_id'])
            print(meta[0]['cif_detections'],meta[0]['cif_stride'])
            print("meta length: {}".format(len(meta)))
            print('cif_detections length: {}'.format(len(meta[0]['cif_detections'][0][1])))

            cif_detections = meta[0]['cif_detections']
            cif_stride = meta[0]['cif_stride']  #in this case is 8 (base_stride=16, upsampling=2)
            image = data[0].cpu().numpy().transpose(1,2,0).astype(np.uint8).copy()
            print(image.shape,type(image))

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
            for detection in cif_detections:
                center_point = [cor * cif_stride for cor in detection[1][:2]] 
                top_left_point = [cor *cif_stride for cor in detection[1][2:4]]
                bottom_right_point = [cor * cif_stride for cor in detection[1][4:6]]
                print('center point: {}, top left point: {}, bottom right point: {}'.format(center_point,top_left_point,bottom_right_point))
                #draw center point in image with red color using opencv
                image = cv2.circle(image, (int(center_point[0]), int(center_point[1])), 5, (0,0,255), -1)
                #draw top left point in image with green color using opencv
                image = cv2.circle(image, (int(top_left_point[0]), int(top_left_point[1])), 5, (0,255,0), -1)
                #draw bottom right point in image with blue color using opencv
                image = cv2.circle(image, (int(bottom_right_point[0]), int(bottom_right_point[1])), 5, (255,0,0), -1)
            cv2.imwrite('/home/jiguo/test_{}.jpg'.format(meta[0]['image_id']), image)
            '''
            

            # soft, _ = resource.getrlimit(resource.RLIMIT_AS)
            # print("Current shared memory soft limit: {} bytes".format(soft))
            # hard, _ = resource.getrlimit(resource.RLIMIT_AS)
            # print("Current shared memory hard limit: {} bytes".format(hard))
            # print("CPU memory usage before {}th batch: {:.2f} MB".format(batch_idx, psutil.Process().memory_info().rss / 1024 ** 2))
            # print("shared CPU memory before {}th batch: {:.2f} MB".format(batch_idx, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))

            preprocess_time = time.time() - last_batch_end

            batch_start = time.time()
            apply_gradients = batch_idx % self.stride_apply == 0
            loss, head_losses = self.train_batch(data, target, apply_gradients)

            # print("CPU memory usage after {}th batch: {:.2f} MB".format(batch_idx, psutil.Process().memory_info().rss / 1024 ** 2))
            # print("shared CPU memory usage after {}th batch: {:.2f} MB".format(batch_idx, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))

            

            # update epoch accumulates
            if loss is not None:
                epoch_loss += loss
            if head_epoch_losses is None:
                head_epoch_losses = [0.0 for _ in head_losses]
                head_epoch_counts = [0 for _ in head_losses]
            for i, head_loss in enumerate(head_losses):
                if head_loss is None:
                    continue
                head_epoch_losses[i] += head_loss
                head_epoch_counts[i] += 1

            batch_time = time.time() - batch_start

            # write training loss
            if batch_idx % self.log_interval == 0:
                batch_info = {
                    'type': 'train',
                    'epoch': epoch, 'batch': batch_idx, 'n_batches': len(scenes),
                    'time': round(batch_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': round(self.lr(), 8),
                    'loss': round(loss, 3) if loss is not None else None,
                    'head_losses': [round(l, 3) if l is not None else None
                                    for l in head_losses],
                }
                if hasattr(self.loss, 'batch_meta'):
                    batch_info.update(self.loss.batch_meta())
                LOG.info(batch_info)

            # initialize ema
            if self.ema is None and self.ema_decay:
                self.ema = copy.deepcopy([p.data for p in self.model.parameters()])

            # update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.n_train_batches and batch_idx + 1 >= self.n_train_batches:
                break

            last_batch_end = time.time()

        self.apply_ema()
        LOG.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / len(scenes), 5),
            'head_losses': [round(l / max(1, c), 5)
                            for l, c in zip(head_epoch_losses, head_epoch_counts)],
            'time': round(time.time() - start_time, 1),
            'n_clipped_grad': self.n_clipped_grad,
            'max_norm': self.max_norm,
        })
        self.n_clipped_grad = 0
        self.max_norm = 0.0

    def val(self, scenes, epoch):
        start_time = time.time()

        # Train mode implies outputs are for losses, so have to use it here.
        self.model.train()
        if self.fix_batch_norm is True \
           or (self.fix_batch_norm is not False and self.fix_batch_norm <= epoch - 1):
            LOG.info('fix batchnorm')
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    LOG.debug('eval mode for: %s', m)
                    m.eval()

        epoch_loss = 0.0
        head_epoch_losses = None
        head_epoch_counts = None
        for batch_idx, (data, target, _) in enumerate(scenes):
            loss, head_losses = self.val_batch(data, target)

            # update epoch accumulates
            if loss is not None:
                epoch_loss += loss
            if head_epoch_losses is None:
                head_epoch_losses = [0.0 for _ in head_losses]
                head_epoch_counts = [0 for _ in head_losses]
            for i, head_loss in enumerate(head_losses):
                if head_loss is None:
                    continue
                head_epoch_losses[i] += head_loss
                head_epoch_counts[i] += 1

            if self.n_val_batches and batch_idx + 1 >= self.n_val_batches:
                break


            #temp information for debugging val(will be removed after)
            # batch_info = {
            #         'type': 'val',
            #         'epoch': epoch, 'batch': batch_idx, 'n_batches': len(scenes),
            #         'lr': round(self.lr(), 8),
            #         'loss': round(loss, 3) if loss is not None else None,
            #         'head_losses': [round(l, 3) if l is not None else None
            #                         for l in head_losses],
            #     }
            # LOG.info(batch_info)

        eval_time = time.time() - start_time

        LOG.info({
            'type': 'val-epoch',
            'epoch': epoch,
            'loss': round(epoch_loss / len(scenes), 5),
            'head_losses': [round(l / max(1, c), 5)
                            for l, c in zip(head_epoch_losses, head_epoch_counts)],
            'time': round(eval_time, 1),
        })

    def write_model(self, epoch, final=True):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        model_to_save = self.model
        if self.checkpoint_shell is not None:
            model = self.model if not hasattr(self.model, 'module') else self.model.module
            self.checkpoint_shell.load_state_dict(model.state_dict())
            model_to_save = self.checkpoint_shell

        filename = '{}.epoch{:03d}'.format(self.out, epoch)
        LOG.debug('about to write model')
        torch.save({
            'model': model_to_save,
            'epoch': epoch,
            'meta': self.model_meta_data,
        }, filename)
        LOG.info('model written: %s', filename)

        optim_filename = '{}.optim.epoch{:03d}'.format(self.out, epoch)
        LOG.debug('about to write training state')
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss.state_dict(),
        }, optim_filename)
        LOG.info('training state written: %s', optim_filename)

        if final:
            sha256_hash = hashlib.sha256()
            with open(filename, 'rb') as f:
                for byte_block in iter(lambda: f.read(8192), b''):
                    sha256_hash.update(byte_block)
            file_hash = sha256_hash.hexdigest()
            outname, _, outext = self.out.rpartition('.')
            final_filename = '{}-{}.{}'.format(outname, file_hash[:8], outext)
            shutil.copyfile(filename, final_filename)
