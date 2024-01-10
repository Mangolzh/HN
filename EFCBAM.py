import tensorflow as tf
import sys
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization

# 提取边缘
def getEdges(input):
    shape = tf.shape(input)
    pad = tf.abs(tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC"))
    input = tf.abs(input)
    pad1 = tf.abs(input - pad[:, :-2, 1:-1, :]) # H：从头取，除了最后两个元素 W：从下标为1的元素开始取，除了最后一个
    pad2 = tf.abs(input - pad[:, 2:, 1:-1, :])
    pad3 = tf.abs(input - pad[:, 1:-1, :-2, :])
    pad4 = tf.abs(input - pad[:, 1:-1, 2:, :])
    output = K.max([pad1, pad2, pad3, pad4], axis=0, keepdims=True)
    output = tf.reshape(output, shape=shape)
    return output
# 傅里叶变换函数
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



#CABM

# 判断输入数据格式，是channels_first还是channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3

# CAM 通道注意力
def channel_attention(input_xs, reduction_ratio=0.125): #0.125=1/8
    # get channel

    channel = int(input_xs.shape[channel_axis])
    # input_xs_fft = Lambda(FFT)(input_xs)
    input_xs_fft = input_xs
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs_fft)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs_fft)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    output = KL.Multiply()([channel_attention_feature, input_xs])
    return output

# SAM 空间注意力
def spatial_attention(channel_refined_feature):
    # channel_refined_feature_edge = channel_refined_feature
    channel_refined_feature_edge = Lambda(getEdges)(channel_refined_feature)
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature_edge)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature_edge)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    output = KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)
    return output


def cbam_module(input_xs, reduction_ratio=0.5):
    # channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    channel_refined_feature = input_xs
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    output = KL.Add()([refined_feature, input_xs])
    return output



if __name__ == '__main__':

    image_shape = (256, 256, 3)  # h*w*c
    inputs = Input(shape=image_shape)
    x_cbam = cbam_module(inputs)
    model = Model(inputs=inputs, outputs=x_cbam, name='cbam')

    d_opt = RMSprop(lr=0.0001, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=d_opt, loss="mse")
    model.summary()

    # # 使用numpy模拟一个真实图片的尺寸
    # input_xs = np.ones([2, 256, 256, 3], dtype='float32') * 0.5
    # # numpy转Tensor
    # input_xs = tf.convert_to_tensor(input_xs)
    # print(input_xs.shape)  # output： (2, 256, 256, 3)
    # outputs = cbam_module(input_xs)
    # print(outputs.shape)  # output： (2, 256, 256, 3)



