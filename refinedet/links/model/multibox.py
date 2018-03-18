import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class ResidualMultibox(chainer.Chain):
    """Multibox head of Deconvolutional Single Shot Detector.

    This is a head part of Deconvolutional Single Shot Detector, also usable
    as a head part of Single Shot Multibox Detector[#]_.

    This link computes :obj:`mb_locs` and :obj:`mb_confs` from feature maps.
    :obj:`mb_locs` contains information of the coordinates of bounding boxes
    and :obj:`mb_confs` contains confidence scores of each classes.

    .. [#] Cheng-Yang Fu, Wei Liu, Ananth Ranga, Ambrish Tyagi,
       Alexander C. Berg
       DSSD : Deconvolutional Single Shot Detector.
       https://arxiv.org/abs/1701.06659.

    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.LeCunUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    """

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(ResidualMultibox, self).__init__()
        with self.init_scope():
            self.res = chainer.ChainList()
            self.loc = chainer.ChainList()
            self.conf = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.res.add_link(Residual(**init))
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

    def __call__(self, xs):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        mb_locs = list()
        mb_confs = list()
        for i, x in enumerate(xs):
            x = self.res[i](x)
            mb_loc = self.loc[i](x)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.conf[i](x)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self.n_class))
            mb_confs.append(mb_conf)

        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs


class Residual(chainer.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.
    Args:
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        initial_bias (4-D array): Initial bias value used in
            the convolutional layers.
    """

    def __init__(self, initialW=None, initial_bias=None):
        super(Residual, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                256, 1, pad=0, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.bn1 = L.BatchNormalization(256)
            self.conv2 = L.Convolution2D(
                256, 1, pad=0, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.bn2 = L.BatchNormalization(256)
            self.conv3 = L.Convolution2D(
                1024, 1, pad=0,initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.bn3 = L.BatchNormalization(1024)
            self.conv4 = L.Convolution2D(
                1024, 1, pad=0, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.bn4 = L.BatchNormalization(1024)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class DeconvolutionalResidualMultibox(chainer.Chain):
    """Multibox head of Deconvolutional Single Shot Detector.

    This is a head part of Deconvolutional Single Shot Detector, also usable
    as a head part of Single Shot Multibox Detector[#]_.

    This link computes :obj:`mb_locs` and :obj:`mb_confs` from feature maps.
    :obj:`mb_locs` contains information of the coordinates of bounding boxes
    and :obj:`mb_confs` contains confidence scores of each classes.

    .. [#] Cheng-Yang Fu, Wei Liu, Ananth Ranga, Ambrish Tyagi,
       Alexander C. Berg
       DSSD : Deconvolutional Single Shot Detector.
       https://arxiv.org/abs/1701.06659.

    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.LeCunUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    """

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(DeconvolutionalResidualMultibox, self).__init__()
        with self.init_scope():
            self.res = chainer.ChainList()
            self.loc = chainer.ChainList()
            self.conf = chainer.ChainList()
            self.dec = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for ar in reversed(aspect_ratios):
            n = (len(ar) + 1) * 2
            self.res.add_link(Residual(**init))
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

        for i in range(len(aspect_ratios) - 1):
            if i == 0:
                dec_ksize, con_ksize, dec_stride = 2, 2, 2
            elif i == 1:
                dec_ksize, con_ksize, dec_stride = 2, 2, 1
            elif i == 2:
                dec_ksize, con_ksize, dec_stride = 2, 3, 2
            elif i == 3:
                dec_ksize, con_ksize, dec_stride = 1, 3, 2
            else:
                dec_ksize, con_ksize, dec_stride = 2, 3, 2

            out_channel = 512
            self.dec.add_link(DeconvolutionModule(out_channel, dec_ksize,
                                                  con_ksize, dec_stride, **init))

    def __call__(self, xs):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        mb_locs = list()
        mb_confs = list()
        for i, x in enumerate(reversed(xs)):
            if i == 0:
                y = x
            else:
                y = self.dec[i-1](x, y)
            y = self.res[i](y)
            mb_loc = self.loc[i](y)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.conf[i](y)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self.n_class))
            mb_confs.append(mb_conf)

        mb_locs = reversed(mb_locs)
        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = reversed(mb_confs)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs


class DeconvolutionModule(chainer.Chain):
    def __init__(self, out_channel, dec_ksize, con_ksize, dec_stride, **init):
        super(DeconvolutionModule, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(out_channel, 3, pad=1, **init)
            self.conv1_2 = L.Convolution2D(out_channel, 3, pad=1, **init)
            self.bn1_1 = L.BatchNormalization(out_channel)
            self.bn1_2 = L.BatchNormalization(out_channel)

            self.deconv2_1 = L.Deconvolution2D(out_channel, dec_ksize,
                                               stride=dec_stride, **init)
            self.bn2_1 = L.BatchNormalization(out_channel)
            self.conv2_1 = L.Convolution2D(out_channel, con_ksize, 3, pad=1,
                                           **init)

    def __call__(self, x1, x2):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        x1 = self.conv1_1(x1)
        x1 = self.bn1_1(x1)
        x1 = F.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.bn1_2(x1)

        x2 = self.deconv2_1(x2)
        x2 = self.conv2_1(x2)
        x2 = self.bn2_1(x2)

        return F.relu(x1 * x2)


class ExtendedResidualMultibox(chainer.Chain):

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(ExtendedResidualMultibox, self).__init__()
        with self.init_scope():
            self.res = chainer.ChainList()
            self.loc = chainer.ChainList()
            self.conf = chainer.ChainList()
            self.ext = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for i in range(3):
            self.ext.add_link(ExtensionModule(i==1, **init))

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.res.add_link(Residual(**init))
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

    def __call__(self, xs):

        mb_locs = list()
        mb_confs = list()

        for i in reversed(range(3)):
            xs[i] = self.ext[i](xs[i], xs[i + 1])

        for i, x in enumerate(xs):
            x = self.res[i](x)
            mb_loc = self.loc[i](x)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.conf[i](x)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self.n_class))
            mb_confs.append(mb_conf)

        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs


class ExtensionModule(chainer.Chain):
    def __init__(self, change_dec2=False, **init):
        super(ExtensionModule, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(512, 3, pad=1, **init)
            self.conv1_2 = L.Convolution2D(512, 3, pad=1, **init)
            self.bn1_1 = L.BatchNormalization(512)
            self.bn1_2 = L.BatchNormalization(512)

            if change_dec2:
                self.deconv2_1 = L.Deconvolution2D(512, 3,
                                                   stride=2, pad=1, **init)
            else:
                self.deconv2_1 = L.Deconvolution2D(512, 2,
                                                   stride=2, **init)
            self.bn2_1 = L.BatchNormalization(512)
            self.conv2_1 = L.Convolution2D(512, 3, pad=1, **init)

    def __call__(self, x1, x2):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        x1 = self.conv1_1(x1)
        x1 = self.bn1_1(x1)
        x1 = F.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.bn1_2(x1)
        x1 = F.relu(x1)

        x2 = self.deconv2_1(x2)
        x2 = self.conv2_1(x2)
        x2 = self.bn2_1(x2)
        x2 = F.relu(x2)

        return x1 + x2


class ExtendedConv(chainer.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.
    Args:
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        initial_bias (4-D array): Initial bias value used in
            the convolutional layers.
    """

    def __init__(self, initialW=None, initial_bias=None):
        super(ExtendedConv, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                512, 1, pad=0, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.conv2 = L.Convolution2D(
                512, 1, pad=0, initialW=initialW, initial_bias=initial_bias,
                nobias=True)

    def __call__(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(x)
        return h1, h2


class ExtendedMultibox(chainer.Chain):

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(ExtendedMultibox, self).__init__()
        with self.init_scope():
            self.extconv = chainer.ChainList()
            self.loc = chainer.ChainList()
            self.conf = chainer.ChainList()
            self.ext = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for i in range(3):
            self.ext.add_link(ExtensionModule(i==1, **init))

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.extconv.add_link(ExtendedConv(**init))
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

    def __call__(self, xs):

        mb_locs = list()
        mb_confs = list()

        for i in reversed(range(3)):
            xs[i] = self.ext[i](xs[i], xs[i + 1])

        for i, x in enumerate(xs):
            x1, x2 = self.extconv[i](x)
            mb_loc = self.loc[i](x1)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.conf[i](x2)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self.n_class))
            mb_confs.append(mb_conf)

        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs


class TransferConnectionEnd(chainer.Chain):

    def __init__(self, initialW=None, initial_bias=None):
        super(TransferConnectionEnd, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                256, 3, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.conv2 = L.Convolution2D(
                256, 3, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.conv3 = L.Convolution2D(
                256, 3, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)

    def __call__(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = self.conv2(h1)
        h3 = F.relu(h2)
        h4 = F.relu(self.conv3(h3))
        return h4


class TransferConnection(chainer.Chain):

    def __init__(self, initialW=None, initial_bias=None):
        super(TransferConnection, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                256, 3, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.conv2 = L.Convolution2D(
                256, 3, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.deconv = L.Deconvolution2D(
                256, 4, stride=2, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.conv3 = L.Convolution2D(
                256, 3, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)

    def __call__(self, x, y):
        h1 = F.relu(self.conv1(x))
        h2 = self.conv2(h1)
        d1 = self.deconv(y)
        h3 = F.relu(h2 + d1)
        h4 = F.relu(self.conv3(h3))
        return h4


class MultiboxWithTCB(chainer.Chain):

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(MultiboxWithTCB, self).__init__()
        with self.init_scope():
            self.arm_loc = chainer.ChainList()
            self.arm_conf = chainer.ChainList()
            self.tcb = chainer.ChainList()
            self.odm_loc = chainer.ChainList()
            self.odm_conf = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for i in range(3):
            self.tcb.add_link(TransferConnection(**init))
        self.tcb.add_link(TransferConnectionEnd(**init))

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2 - 1
            self.arm_loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.arm_conf.add_link(L.Convolution2D(n, 3, pad=1, **init))
            self.odm_loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.odm_conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

    def __call__(self, xs):

        arm_locs = list()
        arm_confs = list()
        odm_locs = list()
        odm_confs = list()

        ys = [None] * 4
        ys[3] = self.tcb[3](xs[3])
        for i in reversed(range(3)):
            ys[i] = self.tcb[i](xs[i], ys[i + 1])

        for i, x in enumerate(xs):
            arm_loc = self.arm_loc[i](x)
            arm_loc = F.transpose(arm_loc, (0, 2, 3, 1))
            arm_loc = F.reshape(arm_loc, (arm_loc.shape[0], -1, 4))
            arm_locs.append(arm_loc)

            arm_conf = self.arm_conf[i](x)
            arm_conf = F.transpose(arm_conf, (0, 2, 3, 1))
            arm_conf = F.reshape(
                arm_conf, (arm_conf.shape[0], -1, 1))
            arm_confs.append(arm_conf)

        arm_locs = F.concat(arm_locs, axis=1)
        arm_confs = F.concat(arm_confs, axis=1)

        for i, y in enumerate(ys):
            odm_loc = self.odm_loc[i](y)
            odm_loc = F.transpose(odm_loc, (0, 2, 3, 1))
            odm_loc = F.reshape(odm_loc, (odm_loc.shape[0], -1, 4))
            odm_locs.append(odm_loc)

            odm_conf = self.odm_conf[i](y)
            odm_conf = F.transpose(odm_conf, (0, 2, 3, 1))
            odm_conf = F.reshape(
                odm_conf, (odm_conf.shape[0], -1, self.n_class))
            odm_confs.append(odm_conf)

        odm_locs = F.concat(odm_locs, axis=1)
        odm_confs = F.concat(odm_confs, axis=1)

        return arm_locs, arm_confs, odm_locs, odm_confs
