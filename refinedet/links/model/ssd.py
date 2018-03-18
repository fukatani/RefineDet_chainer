from __future__ import division

import chainer

from chainercv.links.model.ssd import SSD
from chainercv import transforms

from refinedet.links.model.multibox_coder import RefineDetMultiboxCoder


class RefineDetSSD(SSD):

    def __init__(
            self, extractor, multibox,
            steps, sizes, variance=(0.1, 0.2),
            mean=0):
        self.mean = mean
        self.use_preset('visualize')

        super(SSD, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.multibox = multibox

        self.coder = RefineDetMultiboxCoder(
            extractor.grids, multibox.aspect_ratios, steps, sizes, variance)

    def predict(self, imgs):
        x = list()
        sizes = list()
        for img in imgs:
            _, H, W = img.shape
            img = self._prepare(img)
            x.append(self.xp.array(img))
            sizes.append((H, W))

        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            x = chainer.Variable(self.xp.stack(x))
            arm_locs, arm_confs, odm_locs, odm_confs = self(x)
        arm_locs, arm_confs = arm_locs.array, arm_confs.array
        odm_locs, odm_confs = odm_locs.array, odm_confs.array

        bboxes = list()
        labels = list()
        scores = list()
        for arm_loc, arm_conf, odm_loc, odm_conf, size in zip(
                arm_locs, arm_confs, odm_locs, odm_confs, sizes):
            bbox, label, score = self.coder.decode(arm_loc, arm_conf,
                odm_loc, odm_conf, self.nms_thresh, self.score_thresh)
            bbox = transforms.resize_bbox(
                bbox, (self.insize, self.insize), size)
            bboxes.append(chainer.cuda.to_cpu(bbox))
            labels.append(chainer.cuda.to_cpu(label))
            scores.append(chainer.cuda.to_cpu(score))

        return bboxes, labels, scores
