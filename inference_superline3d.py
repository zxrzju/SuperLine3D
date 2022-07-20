import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import sys
import open3d as o3d
from os.path import join

from tensorflow.python.framework.ops import prepend_name_scope

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from model import *

# from visualizer import Visualizer
from collections import OrderedDict

class opt:
    display_id = 1
    display_winsize = 256
    name = 'vis'


parser = argparse.ArgumentParser()
parser.add_argument('--model_idx', type=int, default=40, help='the number of GPUs to use [default: 2]')
parser.add_argument('--num_gpu', type=int, default=1, help='the number of GPUs to use [default: 2]')
parser.add_argument('--gpu_idx', type=int, default=0, help='GPU idx to use [default: 2]')
parser.add_argument('--log_dir', default='log_line', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=20000, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=101, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training for each GPU [default: 24]')
parser.add_argument('--stride', type=int, default=4, help='Batch Size during training for each GPU [default: 24]')
parser.add_argument('--desp', type=int, default=64, help='Batch Size during training for each GPU [default: 24]')

parser.add_argument('--knn', type=int, default=20, help='Batch Size during training for each GPU [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=30000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')

# parser.add_argument('--load_folder', type=str, default='/home/miyun/dataset/lpn_5k_small/', help='dataset folder')
# parser.add_argument('--pred_path', type=str, default='/home/miyun/dataset4t/dataset2/dgcnn_pred/TriFace_desp/', help='dataset folder')

# parser.add_argument('--load_folder', type=str, default='/home/miyun/dataset4t/dataset2/07_single_5k_ins_big_rot3/', help='dataset folder')
parser.add_argument('--load_folder', type=str, default='/home/miyun/dataset4t/dataset2/apollo_test_sjd_diff5/', help='dataset folder')
# parser.add_argument('--load_folder', type=str, default='/home/miyun/dataset4t/dataset2/kitti_test_v25/', help='dataset folder')
# parser.add_argument('--load_folder', type=str, default='/home/miyun/dataset4t/dataset2/kitti/preprocess/', help='dataset folder')
parser.add_argument('--pred_path', type=str, default='/home/miyun/dataset4t/dataset2/dgcnn_pred/apollo_test_2w_rand_k20_120_309011/', help='dataset folder')
parser.add_argument('--best_model', default='log_models/bak/epoch_120.ckpt', help='model checkpoint file path [default: log/model.ckpt]')

FLAGS = parser.parse_args()

TOWER_NAME = 'tower'

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
# MODEL_PATH = FLAGS.best_model % (FLAGS.model_idx)
MODEL_PATH = FLAGS.best_model
STRIDE = FLAGS.stride
KNN = FLAGS.knn
# PRED_PATH = FLAGS.pred_path % (FLAGS.model_idx)
PRED_PATH = FLAGS.pred_path
GPU_IDX = FLAGS.gpu_idx
DESP = FLAGS.desp

# visualizer = Visualizer(opt)

PRED_NP = PRED_PATH + 'np/'
PRED_PCD = PRED_PATH + 'pcd/'
PRED_DESP = PRED_PATH + 'desc/'
PRED_L_NP = PRED_PATH + 'l_np/'
PRED_L_PCD = PRED_PATH + 'l_pcd/'
PRED_L_DESP = PRED_PATH + 'l_desp/'

os.makedirs(PRED_NP) if not os.path.isdir(PRED_NP) else None
os.makedirs(PRED_PCD) if not os.path.isdir(PRED_PCD) else None
os.makedirs(PRED_DESP) if not os.path.isdir(PRED_DESP) else None
os.makedirs(PRED_L_NP) if not os.path.isdir(PRED_L_NP) else None
os.makedirs(PRED_L_PCD) if not os.path.isdir(PRED_L_PCD) else None
os.makedirs(PRED_L_DESP) if not os.path.isdir(PRED_L_DESP) else None


load_folder = FLAGS.load_folder

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) 
os.system('cp train.py %s' % (LOG_DIR)) 
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# MAX_NUM_POINT = 4096
# NUM_CLASSES = 13

MAX_NUM_POINT = NUM_POINT
NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

class_name = sorted(os.listdir(load_folder))

train_all_files = []
test_all_files = []

if os.path.isdir(load_folder):
    test_all_path = join(load_folder, 'test', 'npy')
    l_test_all_path = join(load_folder, 'test', 'npy')

    cls_test_files = sorted(os.listdir(test_all_path))
    l_cls_test_files = sorted(os.listdir(l_test_all_path))

    for i in range(0, len(cls_test_files), 1):
        if '.npy' in cls_test_files[i]:
            test_all_files.append(join(test_all_path, cls_test_files[i]))
            test_all_files.append(join(l_test_all_path, l_cls_test_files[i]))

test_num = len(test_all_files)


test_all_files = test_all_files[:200]

test_data, test_label = [], []
for i in range(len(test_all_files)):
    dat = np.load(test_all_files[i])
    test_data.append(dat[:, :3])
    test_label.append(dat[:, 3])


# # class weight
num_per_class = np.array([2, 1])
weight = num_per_class / float(sum(num_per_class))
ce_label_weight = 1 / (weight + 0.0001)
class_weight = np.expand_dims(ce_label_weight, axis=0)

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)


def get_learning_rate(batch):
  learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            DECAY_STEP,          # Decay step.
            DECAY_RATE,          # Decay rate.
            staircase=True)
  learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
  return learning_rate        

def get_bn_decay(batch):
  bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch*BATCH_SIZE,
            BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True)
  bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
  return bn_decay

def average_gradients(tower_grads):
  """Calculate average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been 
     averaged across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads



def evaluate():
    is_training = False
    GPU_INDEX = GPU_IDX
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, desp, nn_idx0 = get_model(pointclouds_pl, is_training_pl, STRIDE, KNN, DESP)

        # loss = get_loss(pred, labels_pl)
        loss = tf.constant(0)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'desp': desp,
           'loss': loss}

    eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False
    test_size = len(test_data)
    current_data = test_data
    current_label = test_label

    num_batches = test_size // BATCH_SIZE

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        # print(start_idx_1, end_idx_1)
        feed_dict = {
          # ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
          ops['pointclouds_pl']: current_data[start_idx:end_idx],
              ops['labels_pl']: current_label[start_idx:end_idx],
              ops['is_training_pl']: is_training,
              }
        loss_val, pred_val, desp_val = sess.run([ops['loss'], ops['pred'], ops['desp']], feed_dict=feed_dict)

        for j in range(BATCH_SIZE):
            file_name0 = test_all_files[BATCH_SIZE*batch_idx + j]
            # print(file_name0)
            file_name = file_name0[file_name0.rfind('/'):-4]
            if not 'l_' in file_name0:
                seg_file = PRED_NP + file_name + '.npy'
                desc_file = PRED_DESP + file_name + '.npz'
                pcd_file = PRED_PCD+ file_name + '.pcd'
            else:
                seg_file = PRED_L_NP + file_name + '.npy'
                desc_file = PRED_L_DESP + file_name + '.npz'
                pcd_file = PRED_L_PCD+ file_name + '.pcd'

            pred = pred_val[j,...]
            np.save(seg_file , pred)

            desp = desp_val[j, ...]
            # np.save(desc_file, desp)
            np.savez_compressed(desc_file, desp)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(current_data[start_idx:end_idx][j]))
            pcd.paint_uniform_color([0.8, 0.8, 0.8])
            colors = np.asarray(pcd.colors)
            # print(pred.shape)
            pred = np.argmax(pred, 1)
            colors[np.where(pred > 0)[0], :] = [1, 1, 0]
            o3d.io.write_point_cloud(pcd_file, pcd)


if __name__ == "__main__":
#   train()
  with tf.Graph().as_default():
    evaluate()
  LOG_FOUT.close()
