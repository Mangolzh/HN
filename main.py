import tensorflow as tf
from Deblur_Net_RDN1 import Deblur_Net
from mode import *
import argparse
from PIL import ImageFile
from tensorflow.python import debug as tfdbg

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ('true')


## Model specification
parser.add_argument("--channel", type=int, default=3)
parser.add_argument("--n_feats", type=int, default=64)
parser.add_argument("--num_of_down_scale", type=int, default=2)
parser.add_argument("--num_res", type=int, default=8)
parser.add_argument("--num_dilated", type=int, default=3)
parser.add_argument("--gen_resblocks", type = int, default=11)
parser.add_argument("--discrim_blocks", type=int, default=3)
parser.add_argument("--num_RDN", type=int, default=1)
parser.add_argument("--cardinality", type=int, default=32)

## Data specification
parser.add_argument("--train_Sharp_path", type=str, default="E:/00lzh/code/datasets/data-Gopro/test/test/sharp/")
parser.add_argument("--train_Blur_path", type=str, default="E:/00lzh/code/datasets/data-Gopro/test/test/blur/")
parser.add_argument("--train_edge_path", type=str, default="E:/00lzh/code/datasets/data-Gopro/test/test/edge_s/")

parser.add_argument("--test_Sharp_path", type=str, default="E:/00lzh/code/datasets/data-Gopro/test/group1/sharp/")
parser.add_argument("--test_Blur_path", type=str, default="E:/00lzh/code/datasets/data-Gopro/test/group1/blur/")
parser.add_argument("--test_edge_path", type=str, default="E:/00lzh/code/datasets/data-Gopro/test/group1/edge_s/")
parser.add_argument("--test_only_Blur_path", type=str, default="./data/test_only/")
parser.add_argument("--vgg_path", type=str, default="E:/00lzh/code2/our/vgg19/vgg19.npy")
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--result_path", type=str, default="./images_kohler/result")
parser.add_argument("--test_only_result_path", type=str, default="./images/test_only_result")
parser.add_argument("--model_path", type=str, default="./model")
parser.add_argument("--in_memory", type=str2bool, default=True)

## Optimization
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_epoch", type=int, default=300)
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--decay_step", type=int, default=150)
parser.add_argument("--test_with_train", type=str2bool, default=False)
parser.add_argument("--save_test_result", type=str2bool, default=True)

## Training or test specification
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--ratio", type=int, default=16)
parser.add_argument("--critic_updates", type=int, default=5)
parser.add_argument("--augmentation", type=str2bool, default=False)
parser.add_argument("--load_X", type=int, default=640)
parser.add_argument("--load_Y", type=int, default=360)
parser.add_argument("--fine_tuning", type=str2bool, default=True)
parser.add_argument("--log_freq", type=int, default=1)
parser.add_argument("--model_save_freq", type=int, default=50)
parser.add_argument("--test_batch", type=int, default=1)
parser.add_argument("--pre_trained_model", type=str, default="./model/")
parser.add_argument("--pre_trained_edge_model", type=str, default="E:/00lzh/code2/our/bigPaper/initialDeblurGenerator-005/DeblurGAN-tf-master/model/")
parser.add_argument("--chop_forward", type=str2bool, default=False)
parser.add_argument("--chop_size", type=int, default=8e4)
parser.add_argument("--chop_shave", type=int, default=16)

args = parser.parse_args()
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
model = Deblur_Net(args)
model.build_graph()

print("Build model!")

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)
# sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
# sess = tfdbg.LocalCLIDebugWrapperSession(sess, ui_type="readline", dump_root="E:/00lzh/debug")   # 调试步骤b
#
# sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
sess.run(tf.compat.v1.global_variables_initializer())
saver = tf.compat.v1.train.Saver(max_to_keep=None)

if args.mode == 'train':
    train(args, model, sess)

elif args.mode == 'test':
    f = open("test_kohler.txt", 'w')
    test(args, model, sess, saver, f, step=-1, loading=True)
    f.close()

elif args.mode == 'test_only':
    test_only(args, model, sess, saver)
