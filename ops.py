import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import sys
import math


def Conv(name, x, filter_size, in_filters, out_filters, strides, padding):
    with tf.compat.v1.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.compat.v1.get_variable(name=name + 'kernel',
                                           shape=[filter_size, filter_size, in_filters, out_filters],
                                           dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
        bias = tf.compat.v1.get_variable(name=name + 'bias', shape=[out_filters], dtype=tf.float32,
                                         initializer=tf.zeros_initializer())

        return tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding=padding) + bias
    


def Conv_transpose(name, x, filter_size, in_filters, out_filters, fraction = 2, padding = "SAME"):
    
    with tf.compat.v1.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.compat.v1.get_variable('filter', [filter_size, filter_size, out_filters, in_filters],
                                           tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)) )
        size = tf.shape(x)
        output_shape = tf.stack([size[0], size[1] * fraction, size[2] * fraction, out_filters])
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, fraction, fraction, 1], padding)
        
        return x



def dilated_conv_layer(name, x,  filter_size, in_filters, out_filters, dilation):
    with tf.compat.v1.variable_scope(name):
        dilated_kernel = tf.compat.v1.get_variable(name=name+'dilatedconv', shape=[filter_size,  filter_size, in_filters, out_filters],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(stddev = 0.01), trainable=True)


        return tf.nn.atrous_conv2d(x, filters=dilated_kernel, rate=dilation, padding='SAME')



def instance_norm(name, x, dim, affine = False, BN_decay = 0.999, BN_epsilon = 1e-3):

    mean, variance = tf.nn.moments(x, axes = [1, 2])
    x = (x - mean) / ((variance + BN_epsilon) ** 0.5)
    
    if affine :
        beta = tf.compat.v1.get_variable(name = name + "beta", shape = dim, dtype = tf.float32,
                               initializer = tf.constant_initializer(0.0, tf.float32))
        gamma = tf.compat.v1.get_variable(name + "gamma", dim, tf.float32,
                                initializer = tf.constant_initializer(1.0, tf.float32))
        x = gamma * x + beta 
    
    return x

def prelu(name, _x):
    alphas = tf.compat.v1.get_variable(name= name+'alpha', shape=_x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32, trainable=True)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - tf.abs(_x)) * 0.5
    return pos + neg
def group_Norm(name, x, dim, affine = False, num_groups=32, esp=1e-5, scope=None):
    N, H, W, C = x.shape
    G=num_groups
    #G = C // G

    out = []
    for t in tf.split(x, G, axis = 3,):
        x = t
        mean, var = tf.nn.moments(x, axes = [1, 2, 3], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        if affine:
            beta = tf.compat.v1.get_variable(name=name + "beta", shape=dim, dtype=tf.float32,
                                             initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.compat.v1.get_variable(name + "gamma", dim, tf.float32,
                                      initializer=tf.constant_initializer(1.0, tf.float32))
            x = gamma * x + beta
        out.append(x)
    x = tf.concat(out, axis = -1)
    return x

def l2_loss(a, b, weights=1.0, scope=None):
    with tf.name_scope(scope, 'l2_loss', [a, b, weights]):
        loss = tf.reduce_mean((a - b) ** 2) * weights
        return loss


def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:, :, :channel], x[:, :, channel:])  # 将两实数转化成复数形式
    elif x.shape.ndims == 4:
        return tf.complex(x[:, :, :, :channel], x[:, :, :, channel:])

def FFT(inputx, ndims=2):
    if not inputx.dtype in [tf.complex64, tf.complex128]:
        print('Warning: inputx is not complex. Converting.', file=sys.stderr)
        # if inputx is float, this will assume 0 imag channel
        inputx = tf.cast(inputx, tf.complex64)
    # get the right fft
    if ndims == 1:
        fft = tf.fft
    elif ndims == 2:
        fft = tf.fft2d
    else:
        fft = tf.fft3d
    perm_dims = [0, ndims + 1] + list(range(1, ndims + 1))
    invert_perm_ndims = [0] + list(range(2, ndims + 2)) + [1]
    perm_inputx = K.permute_dimensions(inputx, perm_dims)  # [batch_size, nb_features, *vol_size]
    fft_inputx = fft(perm_inputx)
    # fft_inputx = tf.angle(fft_inputx)
    output = K.permute_dimensions(fft_inputx, invert_perm_ndims)
    output = tf.cast(output, tf.float32)
    output = tf.nn.l2_normalize(output, dim=[1, 2])
    output = tf.abs(output)
    return output


# %%

# %%
# 模仿matlab的fspecial函数，创建滤波算子（计算SSIM用）
# 模仿matlab的fspecial函数，创建滤波算子（计算SSIM用）
def _tf_fspecial_gauss(size, sigma, channels=1):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

    window = g / tf.reduce_sum(g)
    return tf.tile(window, (1, 1, channels, channels))


# 计算ssim
def tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=11, filter_sigma=1.5):
    _, height, width, ch = img1.get_shape().as_list()
    # size = min(filter_size, height, width)
    size = filter_size
    sigma = size * filter_sigma / filter_size if filter_size else 0

    window = _tf_fspecial_gauss(size, sigma, ch)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # 求取滑块内均值Ux Uy，均方值Ux_sq
    padded_img1 = tf.pad(img1, [[0, 0], [size // 2, size // 2], [size // 2, size // 2], [0, 0]],
                         mode="CONSTANT")  # img1 上下左右补零
    padded_img2 = tf.pad(img2, [[0, 0], [size // 2, size // 2], [size // 2, size // 2], [0, 0]],
                         mode="CONSTANT")  # img2 上下左右补零
    mu1 = tf.nn.conv2d(padded_img1, window, strides=[1, 1, 1, 1], padding='VALID')  # 利用滑动窗口，求取窗口内图像的的加权平均
    mu2 = tf.nn.conv2d(padded_img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1  # img(x,y) Ux*Ux 均方
    mu2_sq = mu2 * mu2  # img(x,y) Uy*Uy
    mu1_mu2 = mu1 * mu2  # img(x,y) Ux*Uy

    # 求取方差，方差等于平方的期望减去期望的平方，平方的均值减去均值的平方
    paddedimg11 = padded_img1 * padded_img1
    paddedimg22 = padded_img2 * padded_img2
    paddedimg12 = padded_img1 * padded_img2

    sigma1_sq = tf.nn.conv2d(paddedimg11, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq  # sigma1方差
    sigma2_sq = tf.nn.conv2d(paddedimg22, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq  # sigma2方差
    sigma12 = tf.nn.conv2d(paddedimg12, window, strides=[1, 1, 1, 1],
                           padding='VALID') - mu1_mu2  # sigma12协方差，乘积的均值减去均值的乘积

    ssim_value = tf.clip_by_value(
        ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)), 1e-7, 1)
    if cs_map:  # 只考虑contrast对比度，structure结构，不考虑light亮度
        cs_map_value = tf.clip_by_value((2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2), 1e-7, 1)  # 对比度结构map
        value = (ssim_value, cs_map_value)
    else:
        value = ssim_value
    if mean_metric:  # 求取矩阵的均值，否则返回ssim矩阵
        value = tf.reduce_mean(value)
    return value


# 计算跨尺度结构相似度指数（通过缩放原始图像方式）
def tf_ms_ssim_resize(img1, img2, weights=None, return_ssim_map=None, filter_size=11, filter_sigma=1.5):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # 论文中提到的几个参数
    level = len(weights)
    assert return_ssim_map is None or return_ssim_map < level
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    _, h, w, _ = img1.get_shape().as_list()
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size,
                                   filter_sigma=filter_sigma)
        if return_ssim_map == l:
            return_ssim_map = tf.image.resize_images(ssim_map, size=(h, w),
                                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        img1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        img2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    # ms-ssim公式
    value = tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) * (mssim[level - 1] ** weight[level - 1])
    if return_ssim_map is not None:
        return value, return_ssim_map
    else:
        return value


# 计算跨尺度结构相似度指数（通过扩大感受野方式）
def tf_ms_ssim(img1, img2, weights=None, mean_metric=False):
    if weights is None:
        weights = [1, 1, 1, 1, 1]  # [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] #[1, 1, 1, 1, 1] #

    level = len(weights)
    sigmas = [0.5]
    for i in range(level - 1):
        sigmas.append(sigmas[-1] * 2)
    weight = tf.constant(weights, dtype=tf.float32)

    mssim = []
    mcs = []
    for l, sigma in enumerate(sigmas):
        filter_size = int(max(sigma * 4 + 1, 11))
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size,
                                   filter_sigma=sigma)
        mssim.append(ssim_map)
        mcs.append(cs_map)

    # list to tensor of dim D+1
    value = mssim[level - 1] ** weight[level - 1]
    for l in range(level):
        value = value * (mcs[l] ** weight[l])
    if mean_metric:
        return tf.reduce_mean(value)
    else:
        return value





