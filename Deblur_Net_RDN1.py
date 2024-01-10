from ops import *
from data_loader import dataloader
from vgg19 import Vgg19
from tflearn.layers.conv import global_avg_pool
from DCT import MultiSpectralAttentionLayer
import keras.layers as KL
from EFCBAM import cbam_module, FFT, getEdges, spatial_attention
from tensorflow.keras.layers import Lambda, UpSampling2D


class Deblur_Net():

    def __init__(self, args):
        self.data_loader = dataloader(args)

        self.channel = args.channel
        self.n_feats = args.n_feats
        self.in_memory = args.in_memory
        self.mode = args.mode
        self.batch_size = args.batch_size
        self.num_of_down_scale = args.num_of_down_scale
        self.num_res = args.num_res
        self.num_dilated = args.num_dilated
        self.gen_resblocks = args.gen_resblocks
        self.discrim_blocks = args.discrim_blocks
        self.vgg_path = args.vgg_path
        self.ratio = args.ratio
        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step

    def senet(self, x, out_dim, ratio):
        squeeze = global_avg_pool(x, name="GlobalAvgPool")
        excitation = tf.layers.dense(inputs=squeeze, units=out_dim / ratio, use_bias=True)
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(inputs=excitation, units=out_dim, use_bias=True)
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        scale = x * excitation

        return scale

    def spatial_attention(self, name, x):
        kernel_size = 7
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name):
            avg_pool = tf.reduce_mean(x, axis=[3], keepdims=True)
            assert avg_pool.get_shape()[-1] == 1
            max_pool = tf.reduce_max(x, axis=[3], keepdims=True)
            assert max_pool.get_shape()[-1] == 1
            concat = tf.concat([avg_pool, max_pool], 3)
            assert concat.get_shape()[-1] == 2

            concat = tf.layers.conv2d(concat, filters=1, kernel_size=[kernel_size, kernel_size], strides=[1, 1],
                                      padding="same",
                                      activation=None, kernel_initializer=kernel_initializer, use_bias=False,
                                      name=name + 'conv')
            assert concat.get_shape()[-1] == 1
            concat = tf.sigmoid(concat, 'sigmoid')

        return concat

    def channel_attention(self, name, input_feature, ratio=8):
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)
        with tf.variable_scope(name):
            channel = input_feature.get_shape()[-1]
            avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

            assert avg_pool.get_shape()[1:] == (1, 1, channel)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                       units=channel // ratio,
                                       activation=tf.nn.relu,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name='mlp_0',
                                       reuse=None)
            avg_pool_prelu = tf.layers.dense(inputs=avg_pool,
                                             units=channel // ratio,
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer,
                                             name='mlp_1',
                                             reuse=None)
            assert (avg_pool + avg_pool_prelu).get_shape()[1:] == (1, 1, channel // ratio)
            avg_pool = tf.layers.dense(inputs=(avg_pool + avg_pool_prelu),
                                       units=channel,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name='mlp_2',
                                       reuse=None)
            assert avg_pool.get_shape()[1:] == (1, 1, channel)

            max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel)
            max_pool = tf.layers.dense(inputs=max_pool,
                                       units=channel // ratio,
                                       activation=tf.nn.relu,
                                       name='mlp_0',
                                       reuse=True)
            max_pool_prelu = tf.layers.dense(inputs=max_pool,
                                             units=channel // ratio,
                                             activation=tf.nn.leaky_relu,
                                             name='mlp_1',
                                             reuse=True)
            assert (max_pool + max_pool_prelu).get_shape()[1:] == (1, 1, channel // ratio)
            max_pool = tf.layers.dense(inputs=(max_pool + max_pool_prelu),
                                       units=channel,
                                       name='mlp_2',
                                       reuse=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel)

            scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

        return input_feature * scale

    def res_block(self, name, x, n_feats):

        _res = x

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = Conv(name=name + 'conv1', x=x, filter_size=3, in_filters=n_feats, out_filters=n_feats, strides=1,
                 padding='VALID')
        x = instance_norm(name=name + 'instance_norm1', x=x, dim=n_feats)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x = Conv(name=name + 'conv2', x=x, filter_size=3, in_filters=n_feats, out_filters=n_feats, strides=1,
                 padding='VALID')
        x = instance_norm(name=name + 'instance_norm2', x=x, dim=n_feats)

        x = self.senet(x, out_dim=n_feats, ratio=self.ratio)

        x = x + _res

        return x

    def DenseBlock(self, name, x, n_feats, ndenselayers, growthrate):
        res = x
        for i in range(ndenselayers):
            x_res = x
            x = Conv(name=name + 'conv' + str(i), x=x, filter_size=3, in_filters=n_feats + growthrate * i,
                     out_filters=growthrate, strides=1, padding='SAME')
            x = tf.nn.relu(x)
            x = tf.concat((x, x_res), axis=-1)
        x = Conv(name=name + 'Conv', x=x, filter_size=1, in_filters=growthrate * ndenselayers + n_feats,
                 out_filters=n_feats, strides=1, padding='SAME')
        x = x + res
        return x

    def bottleNeck(self, name, x, n_feats):
        res = x
        x_7 = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        x_7 = Conv(name=name + 'conv_7x7_1', x=x_7, filter_size=3, in_filters=n_feats, out_filters=n_feats // 2,
                   strides=1, padding='VALID')
        x_7 = instance_norm(name + 'ins_7x7_1', x=x_7, dim=n_feats // 2)
        x_7 = tf.nn.relu(x_7)

        x_5 = Conv(name=name + 'conv_5x5_1', x=x_7, filter_size=5, in_filters=n_feats // 2, out_filters=n_feats // 2,
                   strides=1, padding='SAME')
        x_5 = instance_norm(name + 'ins_5x5_1', x=x_5, dim=n_feats // 2)
        x_5 = tf.nn.relu(x_5)

        add1 = x_5 + x_7
        x_5_2 = Conv(name=name + 'conv_5x5_2', x=add1, filter_size=5, in_filters=n_feats // 2,
                     out_filters=n_feats // 2,
                     strides=1, padding='SAME')
        x_5_2 = instance_norm(name + 'ins_5x5_2', x=x_5_2, dim=n_feats // 2)
        x_5_prelu = prelu(name=name + 'prelu_5x5_2', _x=x_5_2)

        add2 = x_5_prelu + x_7
        x_3_1 = Conv(name=name + 'conv_3x3_1', x=add2, filter_size=3, in_filters=n_feats // 2, out_filters=n_feats // 2,
                     strides=1,
                     padding='SAME')
        x_3_1 = instance_norm(name + 'ins_3x3_1', x=x_3_1, dim=n_feats // 2)
        x_3_1 = tf.nn.relu(x_3_1)

        add3 = x_3_1 + x_7
        x_3_1 = Conv(name=name + 'conv_3x3_2', x=add3, filter_size=3, in_filters=n_feats // 2, out_filters=n_feats // 2,
                     strides=1,
                     padding='SAME')
        x_3_1 = instance_norm(name + 'ins_3x3_2', x=x_3_1, dim=n_feats // 2)
        x_3_1_prelu = prelu(name=name + 'prelu_3x3_2', _x=x_3_1)

        cat = tf.concat((x_5, x_5_prelu, x_3_1, x_3_1_prelu), axis=-1)
        cat = Conv(name=name + 'conv_1x1', x=cat, filter_size=1, in_filters=n_feats * 2, out_filters=n_feats,
                   strides=1,
                   padding='SAME')
        cat = instance_norm(name + 'ins_1x1', x=cat, dim=n_feats // 2)
        cat = tf.nn.relu(cat)

        x = self.channel_attention(name=name + 'ca', input_feature=cat)
        x1 = self.spatial_attention(name=name + 'sa', x=x)
        x = x * x1
        x = x + res
        return x

    def gated(self, name, x, y, x_in_ch, n_feats):
        x = Conv(name=name + 'conv1', x=x, filter_size=1, in_filters=x_in_ch, out_filters=n_feats, strides=1,
                 padding='SAME')
        x = instance_norm(name=name + 'ins', x=x, dim=n_feats)
        x = tf.nn.relu(x)
        F = tf.concat((x, y), axis=-1)
        F = Conv(name=name + 'gate1', x=F, filter_size=3, in_filters=n_feats * 2, out_filters=n_feats,
                 strides=1, padding='SAME')
        F = tf.nn.leaky_relu(F)
        F = Conv(name=name + 'gate2', x=F, filter_size=1, in_filters=n_feats, out_filters=n_feats,
                 strides=1, padding='SAME')
        G = tf.nn.sigmoid(F)
        y1 = y * (1 - G)
        x1 = x * G
        x1 = x1 + y1
        return x1

    def gated3(self, name, x, y, z, x_in_ch, n_feats):
        x = Conv(name=name + 'conv1', x=x, filter_size=1, in_filters=x_in_ch, out_filters=n_feats, strides=1,
                 padding='SAME')
        x = instance_norm(name=name + 'ins', x=x, dim=n_feats)
        x = tf.nn.relu(x)
        F = tf.concat((x,y,z),axis=-1)
        F = Conv(name=name + 'gate1', x=F, filter_size=3, in_filters=n_feats * 3, out_filters=n_feats,
                 strides=1, padding='SAME')
        F = tf.nn.leaky_relu(F)
        F = Conv(name=name + 'gate2', x=F, filter_size=1, in_filters=n_feats, out_filters=n_feats,
                 strides=1, padding='SAME')
        G = tf.nn.sigmoid(F)
        z1 = z * G
        x1 = x * (1-G)
        x1 = x1 + z1
        return x1

    def resBlock(self, name, x, n_feats):
        # res = x
        x_3_1 = instance_norm(name + 'ins_3x3_1', x=x, dim=n_feats)
        x_3_1 = tf.nn.relu(x_3_1)
        x_3_1 = Conv(name=name + 'conv_3x3_1', x=x_3_1, filter_size=3, in_filters=n_feats, out_filters=n_feats,
                     strides=1,
                     padding='SAME')
        x_3_2 = instance_norm(name + 'ins_3x3_2', x=x_3_1, dim=n_feats)
        x_3_2 = tf.nn.relu(x_3_2)
        x_3_2 = Conv(name=name + 'conv_3x3_2', x=x_3_2, filter_size=3, in_filters=n_feats, out_filters=n_feats,
                     strides=1,
                     padding='SAME')

        x = self.channel_attention(name=name + 'ca', input_feature=x_3_2)
        x1 = self.spatial_attention(name=name + 'sa', x=x)
        x = x * x1
        # x = x + res
        return x

    def mainBranchResblock(self, name, x, edge, n_feats):
        ex = self.gated(name=name + 'edge_concat_with_x', x=x, y=edge, x_in_ch=n_feats, n_feats=n_feats)
        x = self.resBlock(name=name + 'main_res', x=ex, n_feats=n_feats)
        # res = ex
        # cat_2 = self.DenseBlock(name=name + "dense", x=ex, n_feats=self.n_feats * 4, ndenselayers=3, growthrate=16)
        # x1 = self.spatial_attention(name=name + 'sa', x=cat_2)
        # x = res * x1
        # x = self.senet(x=x, out_dim=n_feats, ratio=16)
        # x = x + res
        return x

    def edgeBranchResblock(self, name, x, edge, n_feats):
        x = instance_norm(name + 'ins_image', x=x, dim=n_feats)
        edge = instance_norm(name + 'ins_edge', x=edge, dim=n_feats)
        ex = self.gated(name=name + 'edge_concat_with_x', x=x, y=edge, x_in_ch=n_feats, n_feats=n_feats)
        ex_edge = Lambda(getEdges)(ex)
        return ex_edge

    def generator(self, x, edge, reuse=False, name='generator'):

        with tf.compat.v1.variable_scope(name_or_scope=name, reuse=reuse):
            _res = x
            l1 = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
            l1 = Conv(name='conv1', x=l1, filter_size=7, in_filters=self.channel, out_filters=self.n_feats, strides=1,
                      padding='VALID')
            l1 = instance_norm(name='inst_norm1', x=l1, dim=self.n_feats)
            l1 = tf.nn.relu(l1)
            l1 = self.DenseBlock(name='dense1', x=l1, n_feats=self.n_feats, ndenselayers=3, growthrate=16)

            l2 = Conv(name='conv2', x=l1, filter_size=3, in_filters=self.n_feats, out_filters=self.n_feats * 2,
                      strides=2, padding='SAME')
            l2 = instance_norm(name='inst_norm2', x=l2, dim=self.n_feats * 2)
            l2 = tf.nn.relu(l2)
            l2 = self.DenseBlock(name='dense2', x=l2, n_feats=self.n_feats * 2, ndenselayers=3, growthrate=16)

            l3 = Conv(name='conv3', x=l2, filter_size=3, in_filters=self.n_feats * 2, out_filters=self.n_feats * 4,
                      strides=2, padding='SAME')
            l3 = instance_norm(name='inst_norm3', x=l3, dim=self.n_feats * 4)
            l3 = tf.nn.relu(l3)
            l3 = self.DenseBlock(name='dense3', x=l3, n_feats=self.n_feats * 4, ndenselayers=3, growthrate=16)
            y = l3

            for i in range(self.gen_resblocks):
                y = self.bottleNeck(name='res_%02d' % i, x=y, n_feats=self.n_feats * 4)

            y = self.DenseBlock(name='dense4', x=y, n_feats=self.n_feats * 4, ndenselayers=3, growthrate=16)
            d1 = tf.compat.v1.depth_to_space(input=y, block_size=2, name=None)
            d1 = instance_norm(name='deconv_instance_norm1', x=d1, dim=self.n_feats)
            d1 = tf.nn.relu(d1)
            d1 = self.gated(name='d1_concat', x=d1, y=l2, x_in_ch=self.n_feats, n_feats=self.n_feats * 2)

            d1 = self.DenseBlock(name='dense5', x=d1, n_feats=self.n_feats * 2, ndenselayers=3, growthrate=16)
            d2 = tf.compat.v1.depth_to_space(input=d1, block_size=2, name=None)
            d2 = instance_norm(name='deconv_instance_norm2', x=d2, dim=self.n_feats // 2)
            d2 = tf.nn.relu(d2)
            d2 = self.gated(name='d2_concat', x=d2, y=l1, x_in_ch=self.n_feats // 2, n_feats=self.n_feats)

            o = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
            o = Conv(name='conv_last', x=o, filter_size=7, in_filters=self.n_feats, out_filters=self.n_feats // 2,
                     strides=1, padding='VALID')
            o = tf.nn.relu(o)
            o = Conv(name='conv_last_2', x=o, filter_size=3, in_filters=self.n_feats // 2, out_filters=self.channel,
                     strides=1, padding='SAME')
            o = tf.nn.tanh(o)
            o = o + _res
            o = tf.clip_by_value(o, -1.0, 1.0)

        with tf.compat.v1.variable_scope(name_or_scope=name + '_deblur', reuse=reuse):
            m1 = tf.concat((o, edge), axis=-1)
            m1 = tf.pad(m1, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
            m1 = Conv(name='conv_with_1', x=m1, filter_size=7, in_filters=4, out_filters=self.n_feats, strides=1,
                      padding='VALID')
            m1 = instance_norm(name='inst_norm1', x=m1, dim=self.n_feats)
            m1 = tf.nn.relu(m1)

            m2 = Conv(name='conv_with_2', x=m1, filter_size=3, in_filters=self.n_feats, out_filters=self.n_feats * 2,
                      strides=2, padding='SAME')
            m2 = instance_norm(name='inst_norm2', x=m2, dim=self.n_feats * 2)
            m2 = tf.nn.relu(m2)
            m2 = self.gated(name='o_with_edge_2', x=l2, y=m2, x_in_ch=self.n_feats * 2, n_feats=self.n_feats * 2)

            m3 = Conv(name='conv_with_3', x=m2, filter_size=3, in_filters=self.n_feats * 2, out_filters=self.n_feats * 4,
                      strides=2, padding='SAME')
            m3 = instance_norm(name='inst_norm3', x=m3, dim=self.n_feats * 4)
            m3 = tf.nn.relu(m3)
            m3 = self.gated(name='o_with_edge_3', x=l3, y=m3, x_in_ch=self.n_feats * 4, n_feats=self.n_feats * 4)

            brm1 = self.mainBranchResblock(name='first_brm', x=m3, edge=y, n_feats=self.n_feats * 4)
            brm_e_1 = self.edgeBranchResblock(name='first_brm_edge', x=y, edge=brm1, n_feats=self.n_feats * 4)
            x = brm1
            y = brm_e_1
            for i in range(self.num_res):
                x = self.mainBranchResblock(name='brm_%02d' % i, x=x, edge=y, n_feats=self.n_feats * 4)
                y = self.edgeBranchResblock(name='brm_edge_%02d' % i, x=x, edge=y, n_feats=self.n_feats * 4)
            
            p1 = tf.compat.v1.depth_to_space(input=x, block_size=2, name=None)
            p1 = instance_norm(name='m_instance_norm1', x=p1, dim=self.n_feats)
            p1 = tf.nn.relu(p1)
            p1 = self.gated3(name='o_with_edge_4', x=p1, y=d1, z=m2, x_in_ch=self.n_feats, n_feats=self.n_feats * 2)

            p2 = tf.compat.v1.depth_to_space(input=p1, block_size=2, name=None)
            p2 = instance_norm(name='m_instance_norm3', x=p2, dim=self.n_feats // 2)
            p2 = tf.nn.relu(p2)
            p2 = self.gated3(name='o_with_edge_5', x=p2, y=d2, z=m1, x_in_ch=self.n_feats // 2, n_feats=self.n_feats)

            p4 = tf.pad(p2, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
            p4 = Conv(name='conv_with_last', x=p4, filter_size=7, in_filters=self.n_feats, out_filters=self.channel, strides=1,
                      padding='VALID')
            p4 = tf.nn.tanh(p4)
            p4 = p4 + o
            p4 = tf.clip_by_value(p4, -1.0, 1.0)

            return p4

    def discriminator(self, x, reuse=False, name='discriminator'):

        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            x = Conv(name='conv1', x=x, filter_size=4, in_filters=self.channel, out_filters=self.n_feats, strides=2,
                     padding="SAME")
            x = instance_norm(name='inst_norm1', x=x, dim=self.n_feats)
            x = tf.nn.leaky_relu(x)

            prev = 1
            n = 1

            for i in range(self.discrim_blocks):
                prev = n
                n = min(2 ** (i + 1), 8)
                x = Conv(name='conv%02d' % i, x=x, filter_size=4, in_filters=self.n_feats * prev,
                         out_filters=self.n_feats * n, strides=2, padding="SAME")
                x = instance_norm(name='instance_norm%02d' % i, x=x, dim=self.n_feats * n)
                x = tf.nn.leaky_relu(x)

            prev = n
            n = min(2 ** self.discrim_blocks, 8)
            x = Conv(name='conv_d1', x=x, filter_size=4, in_filters=self.n_feats * prev, out_filters=self.n_feats * n,
                     strides=1, padding="SAME")
            x = instance_norm(name='instance_norm_d1', x=x, dim=self.n_feats * n)
            x = tf.nn.leaky_relu(x)

            x = Conv(name='conv_d2', x=x, filter_size=4, in_filters=self.n_feats * n, out_filters=1, strides=1,
                     padding="SAME")
            x = tf.nn.sigmoid(x)

            return x

    def build_graph(self):

        if self.in_memory:
            self.blur = tf.compat.v1.placeholder(name="blur", shape=[None, None, None, self.channel], dtype=tf.float32)
            self.sharp = tf.compat.v1.placeholder(name="sharp", shape=[None, None, None, self.channel], dtype=tf.float32)

            x = self.blur
            label = self.sharp

        else:
            self.data_loader.build_loader()

            if self.mode == 'test_only':
                x = self.data_loader.next_batch
                label = tf.placeholder(name='dummy', shape=[None, None, None, self.channel], dtype=tf.float32)

            elif self.mode == 'train' or self.mode == 'test':
                x = self.data_loader.next_batch[0]
                label = self.data_loader.next_batch[1]

        self.epoch = tf.compat.v1.placeholder(name='train_step', shape=None, dtype=tf.int32)
        self.edge = tf.placeholder(name="edge", shape=[None, None, None, 1], dtype=tf.float32)

        x = (2.0 * x / 255.0) - 1.0
        edge = (2.0 * self.edge / 255.0) - 1.0
        label = (2.0 * label / 255.0) - 1.0

        self.gene_img = self.generator(x, edge, reuse=False)
        self.real_prob = self.discriminator(label, reuse=False)
        self.fake_prob = self.discriminator(self.gene_img, reuse=True)

        epsilon = tf.random.uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)

        interpolated_input = epsilon * label + (1 - epsilon) * self.gene_img
        gradient = tf.gradients(self.discriminator(interpolated_input, reuse=True), [interpolated_input])[0]
        GP_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gradient), axis=[1, 2, 3])) - 1))

        d_loss_real = - tf.reduce_mean(self.real_prob)
        d_loss_fake = tf.reduce_mean(self.fake_prob)

        sobel_g = tf.image.sobel_edges(self.gene_img)
        sobel_l = tf.image.sobel_edges(label)

        sobel_loss = tf.reduce_mean(tf.reduce_sum(tf.square(sobel_l - sobel_g)))

        ms_ssim = 1 - tf_ms_ssim(self.gene_img, label, mean_metric=True)

        if self.mode == 'train':
            self.vgg_net = Vgg19(self.vgg_path)
            self.vgg_net.build(tf.concat([label, self.gene_img], axis=0))
            self.content_loss = tf.reduce_mean(tf.reduce_sum(
                tf.square(self.vgg_net.relu3_3[self.batch_size:] - self.vgg_net.relu3_3[:self.batch_size]),
                axis=[1, 2]))

            self.D_loss = d_loss_real + d_loss_fake + 10.0 * GP_loss
            self.G_loss = - d_loss_fake + 15 * self.content_loss + 20 * sobel_loss

            t_vars = tf.compat.v1.trainable_variables()
            # 选择generator_deblur部分的参数 
            G_vars = [var for var in t_vars if 'generator_deblur' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            # 优化器只优化选中的参数 list with tf.control_dependencies():


            lr = tf.minimum(self.learning_rate, tf.abs(
                2 * self.learning_rate - (self.learning_rate * tf.cast(self.epoch, tf.float32) / self.decay_step)))

            self.D_train = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(self.D_loss,
                                                                                                      var_list=D_vars)
            self.G_train = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(self.G_loss,
                                                                                                      var_list=G_vars)

            logging_D_loss = tf.compat.v1.summary.scalar(name='D_loss', tensor=self.D_loss)
            logging_G_loss = tf.compat.v1.summary.scalar(name='G_loss', tensor=self.G_loss)

        self.PSNR = tf.reduce_mean(tf.image.psnr(((self.gene_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val=1.0))
        self.ssim = tf.reduce_mean(tf.image.ssim(((self.gene_img + 1.0) / 2.0), ((label + 1.0) / 2.0), max_val=1.0))

        logging_PSNR = tf.compat.v1.summary.scalar(name='PSNR', tensor=self.PSNR)
        logging_ssim = tf.compat.v1.summary.scalar(name='ssim', tensor=self.ssim)

        self.output = (self.gene_img + 1.0) * 255.0 / 2.0
        self.output = tf.round(self.output)
        self.output = tf.cast(self.output, tf.uint8)
