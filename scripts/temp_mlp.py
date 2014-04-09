"""
Multilayer Perceptron
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import math
import sys
import warnings

import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d_c01b as conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.monitor import get_monitor_doc
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.models.mlp import Layer

class ConvRectifiedLinearC01B(Layer):
    """
    .. todo::

        WRITEME
    """
    def __init__(self,
                 output_channels,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 irange=None,
                 border_mode='valid',
                 sparse_init=None,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 left_slope=0.0,
                 max_kernel_norm=None,
                 pool_type='max',
                 detector_normalization=None,
                 output_normalization=None,
                 kernel_stride=(1, 1)):
        """
        .. todo::

            WRITEME properly

         output_channels: The number of output channels the layer should have.
         kernel_shape: The shape of the convolution kernel.
         pool_shape: The shape of the spatial max pooling. A two-tuple of ints.
         pool_stride: The stride of the spatial max pooling. Also must be
                      square.
         layer_name: A name for this layer that will be prepended to
                     monitoring channels related to this layer.
         irange: if specified, initializes each weight randomly in
                 U(-irange, irange)
         border_mode: A string indicating the size of the output:
            full - The output is the full discrete linear convolution of the
                   inputs.
            valid - The output consists only of those elements that do not rely
                    on the zero-padding.(Default)
         include_prob: probability of including a weight element in the set
                       of weights initialized to U(-irange, irange). If not
                       included it is initialized to 0.
         init_bias: All biases are initialized to this number
         W_lr_scale: The learning rate on the weights for this layer is
                     multiplied by this scaling factor
         b_lr_scale: The learning rate on the biases for this layer is
                     multiplied by this scaling factor
         left_slope: **TODO**
         max_kernel_norm: If specifed, each kernel is constrained to have at
                          most this norm.
         pool_type: The type of the pooling operation performed the the
                    convolution. Default pooling type is max-pooling.
         detector_normalization, output_normalization:
              if specified, should be a callable object. the state of the
              network is optionally replaced with normalization(state) at each
              of the 3 points in processing:
                  detector: the maxout units can be normalized prior to the
                            spatial pooling
                  output: the output of the layer, after sptial pooling, can
                          be normalized as well
         kernel_stride: The stride of the convolution kernel. A two-tuple of
                        ints.
        """

        super(ConvRectifiedLinearC01B, self).__init__()

        if (irange is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear.")
        elif (irange is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear and not both.")

        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError("ConvRectifiedLinear.set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0]-self.kernel_shape[0]) /
                            self.kernel_stride[0] + 1,
                            (self.input_space.shape[1]-self.kernel_shape[1]) /
                            self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0]+self.kernel_shape[0]) /
                            self.kernel_stride[0] - 1,
                            (self.input_space.shape[1]+self.kernel_shape[1]) /
                            self.kernel_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('c', 0, 1, 'b'))

        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                irange=self.irange,
                input_channels=self.input_space.num_channels,
                input_axes=self.input_space.axes,
                output_axes=self.detector_space.axes,
                output_channels=self.detector_space.num_channels,
                kernel_shape=self.kernel_shape,
                kernel_stride=self.kernel_stride,
                rng=rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                num_nonzero=self.sparse_init,
                input_space=self.input_space,
                output_space=self.detector_space,
                kernel_shape=self.kernel_shape,
                batch_size=self.mlp.batch_size,
                subsample=self.kernel_stride,
                border_mode=self.border_mode,
                rng=rng)
        W, = self.transformer.get_params()
        W.name = 'W'

        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
        self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape

        assert self.pool_type in ['max', 'mean']

        dummy_batch_size = self.mlp.batch_size
        if dummy_batch_size is None:
            dummy_batch_size = 2
        dummy_detector = sharedX(
            self.detector_space.get_origin_batch(dummy_batch_size))
        if self.pool_type == 'max':
            dummy_p = max_pool_c01b(c01b=dummy_detector,
                               pool_shape=self.pool_shape,
                               pool_stride=self.pool_stride,
                               image_shape=self.detector_space.shape)
        elif self.pool_type == 'mean':
            dummy_p = mean_pool(bc01=dummy_detector,
                                pool_shape=self.pool_shape,
                                pool_stride=self.pool_stride,
                                image_shape=self.detector_space.shape)
        dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                               dummy_p.shape[3]],
                                        num_channels=self.output_channels,
                                        axes=('c', 0, 1, 'b'))

        print 'Output space: ', self.output_space.shape

    @wraps(Layer.censor_updates)
    def censor_updates(self, updates):
        """
        .. todo::

            WRITEME
        """

        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1, 2, 3)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = updated_W * scales.dimshuffle(0, 'x', 'x', 'x')

    @wraps(Layer.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp, rows, cols, inp))

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):

        W, = self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1, 2, 3)))

        return OrderedDict([('kernel_norms_min',  row_norms.min()),
                            ('kernel_norms_mean', row_norms.mean()),
                            ('kernel_norms_max',  row_norms.max()), ])

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        d = z * (z > 0.) + self.left_slope * z * (z < 0.)

        self.detector_space.validate(d)

        if not hasattr(self, 'detector_normalization'):
            self.detector_normalization = None

        if self.detector_normalization:
            d = self.detector_normalization(d)

        assert self.pool_type in ['max', 'mean']
        if self.pool_type == 'max':
            p = max_pool_c01b(c01b=d,
                         pool_shape=self.pool_shape,
                         pool_stride=self.pool_stride,
                         image_shape=self.detector_space.shape)
        #elif self.pool_type == 'mean':
        #    p = mean_pool(bc01=d,
        #                  pool_shape=self.pool_shape,
        #                  pool_stride=self.pool_stride,
        #                  image_shape=self.detector_space.shape)
        
        self.output_space.validate(p)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p


def max_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    .. todo::

        WRITEME properly

    Theano's max pooling op only support pool_stride = pool_shape
    so here we have a graph that does max pooling with strides

    bc01: minibatch in format (batch size, channels, rows, cols)
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    assert pr <= r
    assert pc <= c

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    for bc01v in get_debug_values(bc01):
        assert not np.any(np.isinf(bc01v))
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    wide_infinity = T.alloc(T.constant(-np.inf, dtype=config.floatX),
                            bc01.shape[0],
                            bc01.shape[1],
                            required_r,
                            required_c)

    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = T.set_subtensor(wide_infinity[:, :, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,
                       :,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs]
            cur.name = ('max_pool_cur_' + bc01.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = ('max_pool_mx_' + bc01.name + '_' +
                           str(row_within_pool) + '_' + str(col_within_pool))

    mx.name = 'max_pool('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx
    
def max_pool_c01b(c01b, pool_shape, pool_stride, image_shape):
    """
    .. todo::

        WRITEME properly

    Like max_pool but with input using axes ('c', 0, 1, 'b')
      (Alex Krizhevsky format)
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride
    assert pr > 0
    assert pc > 0
    assert pr <= r
    assert pc <= c

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    for c01bv in get_debug_values(c01b):
        assert not np.any(np.isinf(c01bv))
        assert c01bv.shape[1] == r
        assert c01bv.shape[2] == c

    wide_infinity = T.alloc(-np.inf,
                            c01b.shape[0],
                            required_r,
                            required_c,
                            c01b.shape[3])

    name = c01b.name
    if name is None:
        name = 'anon_bc01'
    c01b = T.set_subtensor(wide_infinity[:, 0:r, 0:c, :], c01b)
    c01b.name = 'infinite_padded_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = c01b[:,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs,
                       :]
            cur.name = ('max_pool_cur_' + c01b.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = ('max_pool_mx_' + c01b.name + '_' +
                           str(row_within_pool)+'_'+str(col_within_pool))

    mx.name = 'max_pool('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx


def mean_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    .. todo::

        WRITEME properly

    bc01: minibatch in format (batch size, channels, rows, cols)
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    for bc01v in get_debug_values(bc01):
        assert not np.any(np.isinf(bc01v))
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    wide_infinity = T.alloc(-np.inf,
                            bc01.shape[0],
                            bc01.shape[1],
                            required_r,
                            required_c)

    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = T.set_subtensor(wide_infinity[:, :, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name

    # Create a 'mask' used to keep count of the number of elements summed for
    # each position
    wide_infinity_count = T.alloc(0, bc01.shape[0], bc01.shape[1], required_r,
                                  required_c)
    bc01_count = T.set_subtensor(wide_infinity_count[:, :, 0:r, 0:c], 1)

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,
                       :,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs]
            cur.name = ('mean_pool_cur_' + bc01.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            cur_count = bc01_count[:,
                                   :,
                                   row_within_pool:row_stop:rs,
                                   col_within_pool:col_stop:cs]
            if mx is None:
                mx = cur
                count = cur_count
            else:
                mx = mx + cur
                count = count + cur_count
                mx.name = ('mean_pool_mx_' + bc01.name + '_' +
                           str(row_within_pool) + '_' + str(col_within_pool))

    mx /= count
    mx.name = 'mean_pool('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx


def WeightDecay(*args, **kwargs):
    """
    .. todo::

        WRITEME
    """
    warnings.warn("pylearn2.models.mlp.WeightDecay has moved to "
                  "pylearn2.costs.mlp.WeightDecay")
    from pylearn2.costs.mlp import WeightDecay as WD
    return WD(*args, **kwargs)


def L1WeightDecay(*args, **kwargs):
    """
    .. todo::

        WRITEME
    """
    warnings.warn("pylearn2.models.mlp.L1WeightDecay has moved to "
                  "pylearn2.costs.mlp.WeightDecay")
    from pylearn2.costs.mlp import L1WeightDecay as L1WD
    return L1WD(*args, **kwargs)
