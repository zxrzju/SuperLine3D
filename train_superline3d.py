import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import sys

from os.path import join

from collections import OrderedDict
from liegroups.numpy import SO3

def load_h5_files(h5_path):
    files = sorted(os.listdir(h5_path))
    data_batchlist, label_batchlist, mask_batchlist = [], [], []
    for f in files[:1]:
        file = h5py.File(os.path.join(h5_path, f), 'r')
        data = file["data"][:]
        label = file["label"][:]
        mask = file["mask"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
        mask_batchlist.append(mask)
    # data_batches = np.asarray(data_batchlist)
    # seg_batches = np.asarray(label_batchlist)
    # mask_batches = np.asarray(mask_batchlist)

    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    mask_batches = np.concatenate(mask_batchlist, 0)

    return data_batches, seg_batches, mask_batches

def load_h5_poses(h5_path):
    files = sorted(os.listdir(h5_path))
    data_batchlist = []
    for f in files:
        file = h5py.File(os.path.join(h5_path, f), 'r')
        data = file["poses"][:]
        data_batchlist.append(data)
    data_batches = np.concatenate(data_batchlist, 0)
    return data_batches

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return ['/'+x.name[-5:] for x in local_device_protos if x.device_type == 'GPU']

class opt:
    display_id = 700
    display_winsize = 256
    name = 'vis'


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# import provider
import tf_util
from model import *

# print(os.environ.get('HYPER_PARAMETERS'))

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=2, help='the number of GPUs to use [default: 1]')
parser.add_argument('--vis', type=bool, default=True, help='enable visualization [default: False]')
parser.add_argument('--log_dir', default='./summary/', help='Log dir [default: log]')
parser.add_argument('--model_dir', default='./model/', help='model dir [default: /model/]')
parser.add_argument('--stride', type=int, default=1, help='stride in knn [default: 2]')
parser.add_argument('--trans_noise', type=int, default=20, help='translation noise [default: 10]')
parser.add_argument('--knn', type=int, default=20, help='k in knn [default: 20]')


parser.add_argument('--num_point', type=int, default=15000, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=2, help='4 12 10 Batch Size during training for each GPU [default: 24 14]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=40000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
# parser.add_argument('--load_folder', type=str, default='/public/home/zxr/dataset/TriFaceOneCornerRotLine5k_voxel_so3/', help='dataset folder')
parser.add_argument('--load_folder', type=str, default='/home/miyun/dataset4t/dataset2/kitti_reg_diff35/', help='dataset folder')

# train_args = os.environ.get('HYPER_PARAMETERS').split(' ')
FLAGS = parser.parse_args()
print(FLAGS)
# FLAGS = parser.parse_args()

gpu_name = get_available_gpus()
print(gpu_name)
NUM_GPU = len(gpu_name)
TOWER_NAME = 'tower'

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
VIS = FLAGS.vis
STRIDE = FLAGS.stride
TRANS_NOISE = FLAGS.trans_noise
KNN = FLAGS.knn
print('STRIDE: ', STRIDE)
print('TRANS_NOISE: ', TRANS_NOISE)

load_folder = FLAGS.load_folder

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) 
os.system('cp train_superline3d.py %s' % (LOG_DIR)) 
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


MAX_NUM_POINT = NUM_POINT
NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

if VIS:
  from visualizer import Visualizer
  visualizer = Visualizer(opt)

class_name = sorted(os.listdir(load_folder))

r_train_data0, r_train_label0, r_train_mask0 = load_h5_files(join(load_folder, 'r_train_h5'))
l_train_data0, l_train_label0, l_train_mask0 = load_h5_files(join(load_folder, 'l_train_h5'))

r_train_data0, r_train_label0, r_train_mask0 = r_train_data0[:100], r_train_label0[:100], r_train_mask0[:100]
l_train_data0, l_train_label0, l_train_mask0 = l_train_data0[:100], l_train_label0[:100], l_train_mask0[:100]


# # class weight
num_per_class = np.array([40,1])
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
  learning_rate = tf.maximum(learning_rate, 0.000001) # CLIP THE LEARNING RATE!!
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

def build_pc_node_keypoint_visual(pc_np, seg_gt, seg_pred, kp_gt=None, kp_pred=None, keypoint_other_np=None,  kp_other_gt=None, sigmas_np=None, sigmas_other_np=None):
    pc_color_np = np.repeat(np.expand_dims(np.array([255, 255, 255], dtype=np.int64), axis=0),
                            pc_np.shape[0],
                            axis=0)  # 1x3 -> Nx3
    seg_gt_color = np.repeat(np.expand_dims(np.array([0, 0, 255], dtype=np.int64), axis=0),
                              seg_gt.shape[0],
                              axis=0)  # 1x3 -> Mx3
    seg_pred_color = np.repeat(np.expand_dims(np.array([0, 255, 0], dtype=np.int64), axis=0),
                                seg_pred.shape[0],
                                axis=0)  # 1x3 -> Mx3
    if kp_pred is not None:
        keypoint_color_np = np.repeat(np.expand_dims(np.array([125, 0, 0], dtype=np.int64), axis=0),
                                      kp_pred.shape[0],
                                      axis=0)  # 1x3 -> Kx3
        # # consider the sigma
        # if sigmas_np is not None:
        #     sigmas_normalized_np = (1.0 / sigmas_np) / np.max(1.0 / sigmas_np)  # K
        #     keypoint_color_np = keypoint_color_np * np.expand_dims(sigmas_normalized_np, axis=1)  # Kx3
        #     keypoint_color_np = keypoint_color_np.astype(np.int32)
    if keypoint_other_np is not None:
        keypoint_other_color_np = np.repeat(np.expand_dims(np.array([0, 0, 255], dtype=np.int64), axis=0),
                                            keypoint_other_np.shape[0],
                                            axis=0)  # 1x3 -> Kx3
        # consider the sigma
        if sigmas_other_np is not None:
            sigmas_other_normalized_np = (
                1.0 / sigmas_other_np) / np.max(1.0 / sigmas_other_np)  # K
            keypoint_other_color_np = keypoint_other_color_np * np.expand_dims(sigmas_other_normalized_np,
                                                                                axis=1)  # Kx3
            keypoint_other_color_np = keypoint_other_color_np.astype(
                np.int32)

    if kp_gt is not None:
        gt_color_np = np.repeat(np.expand_dims(np.array([255, 0, 0], dtype=np.int64), axis=0),
                                kp_gt.shape[0],
                                axis=0)  # 1x3 -> Kx3

    pc_vis_np = np.concatenate((pc_np, seg_gt, seg_pred), axis=0)
    pc_vis_color_np = np.concatenate(
        (pc_color_np, seg_gt_color, seg_pred_color), axis=0)
    if kp_pred is not None:
        pc_vis_np = np.concatenate((pc_vis_np, kp_pred), axis=0)
        pc_vis_color_np = np.concatenate(
            (pc_vis_color_np, keypoint_color_np), axis=0)
    if keypoint_other_np is not None:
        pc_vis_np = np.concatenate((pc_vis_np, keypoint_other_np), axis=0)
        pc_vis_color_np = np.concatenate(
            (pc_vis_color_np, keypoint_other_color_np), axis=0)
    if kp_gt is not None:
        pc_vis_np = np.concatenate((pc_vis_np, kp_gt), axis=0)
        pc_vis_color_np = np.concatenate(
            (pc_vis_color_np, gt_color_np), axis=0)
    return pc_vis_np, pc_vis_color_np

def train():
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    batch = tf.Variable(0, trainable=False)
    
    bn_decay = get_bn_decay(batch)
    tf.summary.scalar('bn_decay', bn_decay)

    learning_rate = get_learning_rate(batch)
    tf.summary.scalar('learning_rate', learning_rate)
    
    trainer = tf.train.AdamOptimizer(learning_rate)
    
    num_batches = 2*r_train_data0.shape[0] // (NUM_GPU * BATCH_SIZE)

    loss_weights = tf.train.piecewise_constant(batch, [40*num_batches, 60*num_batches],[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    pred_weights = tf.train.piecewise_constant(batch, [60*num_batches],[[0.0, 1.0], [0.0, 1.0]])
    desp_loss_weights = tf.train.piecewise_constant(batch, [1*num_batches, 4*num_batches],[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    tower_grads = []
    pointclouds_phs = []
    labels_phs = []
    is_training_phs =[]
    sparse_mask_phs = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(NUM_GPU):
        # i += 1
        # with tf.device('/gpu:%d' % i):
        with tf.device(gpu_name[i]):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
      
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # sparse_mask_pl = tf.sparse_placeholder(tf.float32)
            sparse_mask_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_POINT))
            pointclouds_phs.append(pointclouds_pl)
            labels_phs.append(labels_pl)
            is_training_phs.append(is_training_pl)
            sparse_mask_phs.append(sparse_mask_pl)

            pred, desp = get_superline3d_model(pointclouds_phs[-1], is_training_phs[-1], KNN, STRIDE, bn_decay=bn_decay)
            # pred= get_model(pointclouds_phs[-1], is_training_phs[-1], bn_decay=bn_decay)
            # loss = get_loss(pred, labels_phs[-1])
            seg_loss = get_seg_loss(pred, labels_phs[-1], class_weight)

            labels_gt = tf.cast(labels_phs[-1], tf.float32)
            # labels_for_loss = pred_weights[0]*labels_pred + pred_weights[1]*labels_gt
            labels_for_loss = labels_gt

            disc_loss, l_var, l_dist, l_reg, disc_loss0, l_var0, l_dist0, l_reg0 = get_desc_loss(desp, sparse_mask_phs[-1])
            # loss = desp_loss_weights[0] * seg_loss + desp_loss_weights[1]*(l_var0 + l_dist0 + l_dist + l_var)
            loss = seg_loss + l_var0 + l_dist0 + l_dist + l_var
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('seg_loss', seg_loss)
            tf.summary.scalar('desp_loss', disc_loss)
            tf.summary.scalar('desp_loss0', disc_loss0)
            tf.summary.scalar('weight0', desp_loss_weights[0])
            tf.summary.scalar('weight1', desp_loss_weights[1])

            # tf.summary.scalar('nd_num', nd_num)
            tf.summary.scalar('l_dist', l_dist)
            tf.summary.scalar('l_var', l_var)
            tf.summary.scalar('l_reg', l_reg)

            tf.summary.scalar('l_dist0', l_dist0)
            tf.summary.scalar('l_var0', l_var0)
            tf.summary.scalar('l_reg0', l_reg0)
            # tf.summary.scalar('positive', pd)
            # tf.summary.scalar('negative', nd)

            # tf.summary.scalar('feat_positive', feat_pd)
            # tf.summary.scalar('feat_negative', feat_nd)

            # tf.summary.scalar('positive_diff', pd_diff)
            # tf.summary.scalar('negative_diff', nd_diff)

            # tf.summary.scalar('positive_transpose_diff', pd_transpose_diff)
            # tf.summary.scalar('negative_transpose_diff', nd_transpose_diff)


            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_phs[-1]))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            tf.get_variable_scope().reuse_variables()

            grads = trainer.compute_gradients(loss)

            tower_grads.append(grads)
    
    grads = average_gradients(tower_grads)

    train_op = trainer.apply_gradients(grads, global_step=batch)
    
    saver = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep=10)
    
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Add summary writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                  sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

    # Init variables for two GPUs
    init = tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer())
    sess.run(init)

    ops = {'pointclouds_phs': pointclouds_phs,
         'labels_phs': labels_phs,
         'is_training_phs': is_training_phs,
         'sparse_mask_phs': sparse_mask_phs,
         'pred': pred,
         'loss': loss,
         'train_op': train_op,
         'merged': merged,
         'step': batch,
         'l_var0': l_var0,
         'l_dist0': l_dist0,
         'l_var': l_var,
         'l_dist': l_dist}

    for epoch in range(MAX_EPOCH):
      log_string('**** EPOCH %03d ****' % (epoch))
      sys.stdout.flush()
       
      train_one_epoch(sess, ops, train_writer)
      
      # Save the variables to disk.
      if epoch % 5 == 0:
        save_path = saver.save(sess, os.path.join(LOG_DIR,'epoch_' + str(epoch)+'.ckpt'))
        log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
  """ ops: dict mapping from string to tf ops """
  is_training = True
  
  log_string('----')
  # current_data, current_label, _ = provider.shuffle_data(r_train_data[:,0:NUM_POINT,:], r_train_label)
  shuffle_idx = np.arange(len(r_train_data0))
  np.random.shuffle(shuffle_idx)

  r_train_data = r_train_data0[shuffle_idx]
  r_train_mask = r_train_mask0[shuffle_idx]
  r_train_label = r_train_label0[shuffle_idx]
  l_train_data = l_train_data0[shuffle_idx]
  l_train_mask = l_train_mask0[shuffle_idx]
  l_train_label = l_train_label0[shuffle_idx]

  current_data = r_train_data
  current_label = r_train_label
  # print(current_data.shape)
  file_size = current_data.shape[0]
  num_batches = 2 * file_size // (NUM_GPU * BATCH_SIZE)
  
  total_correct = 0
  total_seen = 0
  loss_sum = 0
  start_idxs, end_idxs = [], []
  for batch_idx in range(num_batches):
    if batch_idx % 100 == 0:
      print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
    data_all, label_all, mask_all = [], [], []
    for j in range(NUM_GPU):
        start_idxs.append(int((batch_idx+j) * BATCH_SIZE // 2))
        end_idxs.append(int((batch_idx+j+1) * BATCH_SIZE // 2))
        start_idx = int((batch_idx+j) * BATCH_SIZE // 2)
        end_idx = int((batch_idx+j+1) * BATCH_SIZE // 2)

        cur_data = np.hstack((r_train_data[start_idx:end_idx, :, :], l_train_data[start_idx:end_idx, :, :])).reshape((-1, NUM_POINT, 3))
        cur_mask = np.hstack((r_train_mask[start_idx:end_idx, :], l_train_mask[start_idx:end_idx, :])).reshape((-1, NUM_POINT))
        cur_label = np.hstack((r_train_label[start_idx:end_idx, :], l_train_label[start_idx:end_idx, :])).reshape((-1, NUM_POINT))

        for i in range(len(cur_data)):
            cur_data[i] = np.matmul(cur_data[i], SO3.from_rpy(0, 0, *np.random.rand(1)*np.pi*2).as_matrix())
            cur_data[i, :, :2] += (np.random.rand(2) - 0.5)*TRANS_NOISE
        
        data_all.append(cur_data)
        label_all.append(cur_label)
        mask_all.append(cur_mask)
    
    feed_dict = {}
    for j in range(NUM_GPU):
        feed_dict[ops['pointclouds_phs'][j]] = data_all[j]
        feed_dict[ops['sparse_mask_phs'][j]] = mask_all[j]
        feed_dict[ops['labels_phs'][j]] = label_all[j]
        feed_dict[ops['is_training_phs'][j]] = is_training
    # feed_dict = {ops['pointclouds_phs'][0]: cur_data,
    #               ops['pointclouds_phs'][1]: cur_data_1,
    #               ops['sparse_mask_phs'][0]: cur_mask,
    #               ops['sparse_mask_phs'][1]: cur_mask_1,
    #               ops['labels_phs'][0]: cur_label,
    #               ops['labels_phs'][1]: cur_label_1,
    #               ops['is_training_phs'][0]: is_training,
    #               ops['is_training_phs'][1]: is_training
    #              }
    summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
    train_writer.add_summary(summary, step)
    summary1 = tf.Summary(
        value=[tf.Summary.Value(tag='iou', simple_value=step)])
    train_writer.add_summary(summary1, step)

    # print(pred_val.shape)
    pred_val = np.argmax(pred_val, 2)
    correct = np.sum(pred_val == label_all[start_idxs[-1]:end_idxs[-1]])
    total_correct += correct
    total_seen += (BATCH_SIZE*NUM_POINT)
    loss_sum += loss_val
  
    if VIS and batch_idx % 50 == 0:
      label_gt = label_all[-1][0]
      pc_val = data_all[-1][0]
      label_pred = pred_val[0]

      gt_pc = pc_val[np.where(label_gt>0)[0], :]
      pred_pc = pc_val[np.where(label_pred>0)[0], :]
      
      # if len(gt_pc)==0:
    #   gt_pc = np.array([[0, 0, 0]])
      if len(pred_pc)==0:
        pred_pc = np.array([[0, 0, 0]])
      src_data_vis_np, src_data_vis_color_np = build_pc_node_keypoint_visual(pc_val, gt_pc, pred_pc, None, None)
      visuals = OrderedDict([('src_data_vis', (src_data_vis_np, src_data_vis_color_np))
                      ])
      
      visualizer.display_current_results(visuals)

  log_string('mean loss: %f' % (loss_sum / float(num_batches)))
  log_string('accuracy: %f' % (total_correct / float(total_seen)))

if __name__ == "__main__":
  train()
  LOG_FOUT.close()
