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

        img_batch = list([img for img, bbox, label in batch])
        in_arrays = self.converter(img_batch, self.device)
        arm_locs = self.arm_locs_func(in_arrays)

        for i, arm_loc in enumerate(arm_locs):
            _, bbox, label = batch[i]
            bbox = cupy.asarray(bbox)
            label = cupy.asarray(label)
            mb_loc, mb_label = self.coder.encode(bbox, label)
            decode_arm = self.coder.decode_simple(mb_loc)
            refined_loc, _ = self.coder.encode(bbox, label, decode_arm=decode_arm)
            # refined_loc, _ = self.coder.encode(bbox, label, decode_arm=self.coder._default_bbox)
            batch[i] = list(batch[i])
            batch[i][1] = mb_loc
            batch[i][2] = mb_label
            batch[i].append(refined_loc)
            batch[i] = tuple(batch[i])

        in_arrays = self.converter(batch, self.device)
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        optimizer.update(loss_func, *in_arrays)
