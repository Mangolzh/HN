import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import util
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

from skimage.measure import compare_ssim as ssim


def train(args, model, sess):
    if args.fine_tuning:
        # 选择encode部分参数
        train_var = [var for var in tf.global_variables() if 'generator' in var.name]
        no_train_var = [var for var in train_var if not 'generator_deblur' in var.name]
        saver = tf.train.Saver(no_train_var)
        # 参数覆盖,边缘网络参数模型
        print('[*]Reading checkpoints...')
        ckpt = tf.train.get_checkpoint_state(args.pre_trained_edge_model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(args.pre_trained_edge_model, ckpt_name))
        print("saved edge model is loaded for fine-tuning!")
        print("edge model path is %s" % args.pre_trained_edge_model)

    num_imgs = len(os.listdir(args.train_Sharp_path))

    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter('./logs', sess.graph)
    if args.test_with_train:
        f = open("valid_logs.txt", 'w')

    epoch = 0

    step = num_imgs // args.batch_size

    saver = tf.compat.v1.train.Saver(max_to_keep=None)

    if args.in_memory:

        blur_imgs = util.image_loader(args.train_Blur_path, args.load_X, args.load_Y)
        sharp_imgs = util.image_loader(args.train_Sharp_path, args.load_X, args.load_Y)
        edge_imgs = util.image_loader(args.train_edge_path, args.load_X, args.load_Y)
        avg_D_loss = []
        avg_G_loss = []
        PSNR = []
        SSIM = []

        while epoch < args.max_epoch:
            random_index = np.random.permutation(len(blur_imgs))
            d_loss = []
            g_loss = []
            for k in range(step):
                s_time = time.time()
                blur_batch, sharp_batch, edge_batch = util.batch_gen(blur_imgs, sharp_imgs, edge_imgs, args.patch_size, args.batch_size, random_index, k, args.augmentation)
                edge = edge_batch.reshape(1, 256, 256, 1)

                for t in range(args.critic_updates):
                    _, D_loss = sess.run([model.D_train, model.D_loss],
                                         feed_dict={model.blur: blur_batch, model.sharp: sharp_batch, model.edge: edge,
                                                    model.epoch: epoch})
                    ##
                    d_loss.append(D_loss)

                _, G_loss = sess.run([model.G_train, model.G_loss],
                                     feed_dict={model.blur: blur_batch, model.sharp: sharp_batch, model.edge: edge,
                                                model.epoch: epoch})
                ##
                g_loss.append(G_loss)
                e_time = time.time()

                mean_dloss = np.mean(d_loss)
                mean_gloss = np.mean(g_loss)
                # print("D_loss : %0.4f, \t G_loss : %0.4f" % (D_loss, G_loss))

            if epoch % args.log_freq == 0:
                summary = sess.run(merged,
                                   feed_dict={model.blur: blur_batch, model.sharp: sharp_batch, model.edge: edge})
                train_writer.add_summary(summary, epoch)
                if args.test_with_train:
                    mean_PSNR, mean_ssim = test(args, model, sess, saver, f, epoch, loading=False)
                    PSNR.append(mean_PSNR)
                    SSIM.append(mean_ssim)
                    fig3, ax3 = plt.subplots(figsize=(11, 8))
                    ax3.plot(range(epoch + 1), PSNR)
                    ax3.set_title("Val PSNR vs epochs")
                    ax3.set_xlabel("Epoch")
                    ax3.set_ylabel("Current PSNR")
                    plt.savefig('val_psnr_vs_epochs.png')
                    plt.clf()

                    fig4, ax4 = plt.subplots(figsize=(11, 8))
                    ax4.plot(range(epoch + 1), SSIM)
                    ax4.set_title("Val SSIM vs epochs")
                    ax4.set_xlabel("Epoch")
                    ax4.set_ylabel("Current SSIM")
                    plt.savefig('val_ssim_vs_epochs.png')

                    plt.clf()

                # print("D_loss : %0.4f, \t G_loss : %0.4f"%(D_loss, G_loss))
                # print("Elpased time : %0.4f"%(e_time - s_time))
            if epoch % args.model_save_freq == 0:
                saver.save(sess, './model/DeblurrGAN', global_step=epoch, write_meta_graph=False)

            ##
            mean_dloss = np.mean(d_loss)
            mean_gloss = np.mean(g_loss)

            print("%d training epoch completed" % epoch)
            print("mean_dloss : %0.4f, \t mean_gloss : %0.4f" % (mean_dloss, mean_gloss))
            print("Elpased time : %0.4f" % (e_time - s_time))
            with open("trainlog.txt", 'a+') as flog:  # -----改------#
                flog.write("%d-epoch step, \n mean_dloss : %0.4f, \t mean_gloss : %0.4f \n Elpased time : %0.4f \n" % (
                    epoch, mean_dloss, mean_gloss, e_time - s_time))

            avg_D_loss.append(mean_dloss)
            avg_G_loss.append(mean_gloss)
            fig1, ax1 = plt.subplots(figsize=(11, 8))
            ax1.plot(range(epoch + 1), avg_G_loss)
            ax1.set_title("Average G_loss vs epochs")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Current G_loss")
            plt.savefig('G_loss_vs_epochs.png')
            plt.clf()

            fig2, ax2 = plt.subplots(figsize=(11, 8))
            ax2.plot(range(epoch + 1), avg_D_loss)
            ax2.set_title("Average D_loss vs epochs")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Current D_loss")
            plt.savefig('D_loss_vs_epochs.png')
            plt.clf()

            epoch += 1

        saver.save(sess, './model/DeblurrGAN_last', write_meta_graph=False)

    else:
        while epoch < args.max_epoch:

            sess.run(model.data_loader.init_op['tr_init'])

            for k in range(step):
                s_time = time.time()

                for t in range(args.critic_updates):
                    _, D_loss = sess.run([model.D_train, model.D_loss], feed_dict={model.epoch: epoch})

                _, G_loss = sess.run([model.G_train, model.G_loss], feed_dict={model.epoch: epoch})

                e_time = time.time()

            if epoch % args.log_freq == 0:
                summary = sess.run(merged)
                train_writer.add_summary(summary, epoch)
                if args.test_with_train:
                    test(args, model, sess, saver, f, epoch, loading=False)
                print("%d training epoch completed" % epoch)
                print("D_loss : %0.4f, \t G_loss : %0.4f" % (D_loss, G_loss))
                print("Elpased time : %0.4f" % (e_time - s_time))
            if epoch % args.model_save_freq == 0:
                saver.save(sess, './model/DeblurrGAN', global_step=epoch, write_meta_graph=False)

            epoch += 1

        saver.save(sess, './model/DeblurrGAN_last', global_step=epoch, write_meta_graph=False)

    if args.test_with_train:
        f.close()


def test(args, model, sess, saver, file, step=-1, loading=False):
    if loading:
        print('[*]Reading checkpoints...')
        ckpt = tf.train.get_checkpoint_state(args.pre_trained_model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # ckpt_name = 'DeblurrGAN-200'
            # 参数覆盖
            saver.restore(sess, os.path.join(args.pre_trained_model, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")

    blur_img_name = sorted(os.listdir(args.test_Blur_path))
    sharp_img_name = sorted(os.listdir(args.test_Sharp_path))

    PSNR_list = []
    ssim_list = []

    # blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y)
    # sharp_imgs = util.image_loader(args.test_Sharp_path, args.load_X, args.load_Y)
    blur_imgs = util.image_loader(args.test_Blur_path, args.load_X, args.load_Y, is_train=False)
    sharp_imgs = util.image_loader(args.test_Sharp_path, args.load_X, args.load_Y, is_train=False)
    edge_imgs = util.image_loader(args.test_edge_path, args.load_X, args.load_Y, is_train=False)

    if not os.path.exists('./images_kohler/result/'):
        os.makedirs('./images_kohler/result/')

    for i, ele in enumerate(blur_imgs):
        s_time = time.time()
        blur = np.expand_dims(ele, axis=0)
        sharp = np.expand_dims(sharp_imgs[i], axis=0)
        edge = edge_imgs[i].reshape(1, 720, 1280, 1)
        # edge = edge_imgs[i].reshape(1,800,800,1)
        output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim],
                                      feed_dict={model.blur: blur, model.sharp: sharp, model.edge: edge})
        if args.save_test_result:
            output = Image.fromarray(output[0])
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.result_path, '%s.png' % (''.join(map(str, split_name[:-1])))))

        PSNR_list.append(psnr)

        ssim_list.append(ssim)

        e_time = time.time()

        print('_%02s completed' % blur_img_name[i])
        print('psnr: %0.4f  ssim: %0.4f' % (psnr, ssim))
        print("Elpased time : %0.4f" % (e_time - s_time))
        file.write('_%02s completed \n PSNR : %0.4f SSIM : %0.4f \n Elpased time : %0.4f\n' % (
            blur_img_name[i], psnr, ssim, e_time - s_time))

    length = len(PSNR_list)

    mean_PSNR = sum(PSNR_list) / length
    mean_ssim = sum(ssim_list) / length

    print('mean_PSNR: %0.4f  mean_ssim: %0.4f' % (mean_PSNR, mean_ssim))
    # else:
    #
    #     sess.run(model.data_loader.init_op['val_init'])
    #
    #     for i in range(len(blur_img_name)):
    #
    #         output, psnr, ssim = sess.run([model.output, model.PSNR, model.ssim])
    #
    #         if args.save_test_result:
    #             output = Image.fromarray(output[0])
    #             split_name = blur_img_name[i].split('.')
    #             output.save(os.path.join(args.result_path, '%s_sharp.png'%(''.join(map(str, split_name[:-1])))))
    #
    #         PSNR_list.append(psnr)
    #         ssim_list.append(ssim)
    #
    # length = len(PSNR_list)
    #
    # mean_PSNR = sum(PSNR_list) / length
    # mean_ssim = sum(ssim_list) / length

    if step == -1:
        file.write('PSNR : %0.4f SSIM : %0.4f' % (mean_PSNR, mean_ssim))
        file.close()

    else:
        file.write("%d-epoch step PSNR : %0.4f SSIM : %0.4f \n" % (step, mean_PSNR, mean_ssim))
    return mean_PSNR, mean_ssim

    # fig3, ax3 = plt.subplots(figsize=(11, 8))
    # ax3.plot(range(step + 1), PSNR)
    # ax3.set_title("Val PSNR vs epochs")
    # ax3.set_xlabel("Epoch")
    # ax3.set_ylabel("Current PSNR")
    # plt.savefig('val_psnr_vs_epochs.png')
    #
    # fig4, ax4 = plt.subplots(figsize=(11, 8))
    # ax4.plot(range(step + 1), SSIM)
    # ax4.set_title("Val SSIM vs epochs")
    # ax4.set_xlabel("Epoch")
    # ax4.set_ylabel("Current SSIM")
    # plt.savefig('val_ssim_vs_epochs.png')
    #
    # plt.clf()


def test_only(args, model, sess, saver):
    print('[*]Reading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(args.pre_trained_model)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        # ckpt_name = 'DeblurrGAN-30'
        saver.restore(sess, os.path.join(args.pre_trained_model, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
    else:
        print(" [*] Failed to find a checkpoint")

    blur_img_name = sorted(os.listdir(args.test_only_Blur_path))

    if args.in_memory:

        blur_imgs = util.image_loader(args.test_only_Blur_path, args.load_X, args.load_Y, is_train=False)

        for i, ele in enumerate(blur_imgs):
            blur = np.expand_dims(ele, axis=0)

            if args.chop_forward:
                output = util.recursive_forwarding(blur, args.chop_size, sess, model, args.chop_shave)
                output = Image.fromarray(output[0])

            else:
                output = sess.run(model.output, feed_dict={model.blur: blur})
                output = Image.fromarray(output[0])

            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.test_only_result_path, '%s_sharp.png' % (''.join(map(str, split_name[:-1])))))

    else:

        sess.run(model.data_loader.init_op['te_init'])

        for i in range(len(blur_img_name)):
            output = sess.run(model.output)
            output = Image.fromarray(output[0])
            split_name = blur_img_name[i].split('.')
            output.save(os.path.join(args.test_only_result_path, '%s_sharp.png' % (''.join(map(str, split_name[:-1])))))
