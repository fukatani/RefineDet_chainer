import cupy
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import training

import numpy

import six


class RefineDetUpdater(training.StandardUpdater):

    def __init__(self, iterator, optimizer, coder, converter=convert.concat_examples,
                 device=None, arm_locs_func=None, loss_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)

        self.converter = converter
        self.arm_locs_func = arm_locs_func
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0
        self.coder = coder

    def update_core(self):
        batch = self._iterators['main'].next()

        for i, (_, bbox, label) in enumerate(batch):
            bbox = cupy.asarray(bbox)
            label = cupy.asarray(label)

            mb_loc, mb_label = self.coder.encode(bbox, label)
            batch[i] = list(batch[i])
            batch[i][1] = mb_loc
            batch[i][2] = mb_label
            batch[i] = tuple(batch[i])

        in_arrays = self.converter(batch, self.device)
        arm_locs = self.arm_locs_func(*in_arrays)

        for i, arm_loc in enumerate(arm_locs):
            _, mb_loc, mb_label = batch[i]
            soft_label = cupy.eye(21)[mb_label]
            decode_arm = self.coder.decode_simple(mb_loc, soft_label)
            refined_loc, _ = self.coder.encode(bbox, label, decode_arm=decode_arm)
            batch[i] = list(batch[i])
            batch[i].append(refined_loc)
            batch[i] = tuple(batch[i])

        in_arrays = self.converter(batch, self.device)
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        optimizer.update(loss_func, *in_arrays)
