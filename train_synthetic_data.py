import argparse
import math

import numpy as np
import tensorflow as tf
import socket

import os
import sys

from os.path import join

from visualizer import Visualizer
from collections import OrderedDict

class opt:
    display_id = 13
    display_winsize = 256
    name = 'vis'

import tf_util
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=1, help='the number of GPUs to use [default: 1]')
parser.add_argument('--vis', type=bool, default=True, help='enable visualization [default: False]')
parser.add_argument('--log_dir', default='log_line_pole_noise_si', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=5000, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=101, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training for each GPU [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=2000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
parser.add_argument('--load_folder', type=str, default='./scripts/line_pole_noise/', help='dataset folder')
FLAGS = parser.parse_args()

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

load_folder = FLAGS.load_folder

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) 
os.system('cp train_synthetic_data.py %s' % (LOG_DIR)) 
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

if VIS:
  visualizer = Visualizer(opt)

train_files = []
test_files = []
print(load_folder)
if os.path.isdir(load_folder):
    train_path = join(load_folder, 'train', 'npy')
    test_path = join(load_folder, 'test', 'npy')

    cls_train_files = os.listdir(train_path)
    cls_test_files = os.listdir(test_path)

    for i in range(len(cls_train_files)):
        train_files.append(join(train_path, cls_train_files[i]))

    for i in range(len(cls_test_files)):
        test_files.append(join(test_path, cls_test_files[i]))


train_num = len(train_files)
test_num = len(test_files)


shuffle_idx = np.arange(train_num)
np.random.shuffle(shuffle_idx)
train_files = list(np.array(train_files)[shuffle_idx])

train_data = np.zeros((len(train_files), NUM_POINT, 3))
train_label = np.zeros((len(train_files), NUM_POINT))

for i in range(len(train_files)):
    dat = np.load(train_files[i])
    train_data[i, ...] = dat[:, :3]
    train_label[i, ...] = dat[:, 3]

test_data = np.zeros((len(test_files), NUM_POINT, 3))
test_label = np.zeros((len(test_files), NUM_POINT))

for i in range(len(test_files)):
    dat = np.load(test_files[i])
    test_data[i, ...] = dat[:, :3]
    test_label[i, ...] = dat[:, 3]

print(train_data.shape, train_label.shape)
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
    
    tower_grads = []
    pointclouds_phs = []
    labels_phs = []
    is_training_phs =[]

    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(FLAGS.num_gpu):
        i += 1
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
      
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            pointclouds_phs.append(pointclouds_pl)
            labels_phs.append(labels_pl)
            is_training_phs.append(is_training_pl)
      
            pred = get_seg_model(pointclouds_phs[-1], is_training_phs[-1], k=20, stride=1, scale_invariant=True, bn_decay=bn_decay)
            # loss = get_loss(pred, labels_phs[-1])
            loss = get_seg_loss(pred, labels_phs[-1], class_weight)
            tf.summary.scalar('loss', loss)

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
    config.gpu_options.allow_growth = True
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
         'pred': pred,
         'loss': loss,
         'train_op': train_op,
         'merged': merged,
         'step': batch}

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
  current_data = train_data
  current_label = train_label
  # print(current_data.shape)
  file_size = current_data.shape[0]
  num_batches = file_size // (FLAGS.num_gpu * BATCH_SIZE)
  
  total_correct = 0
  total_seen = 0
  loss_sum = 0
  
  for batch_idx in range(num_batches):
    if batch_idx % 100 == 0:
      print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
    start_idx_0 = batch_idx * BATCH_SIZE
    end_idx_0 = (batch_idx+1) * BATCH_SIZE
    start_idx_1 = (batch_idx+1) * BATCH_SIZE
    end_idx_1 = (batch_idx+2) * BATCH_SIZE
    
    # print(start_idx_1, end_idx_1)
    feed_dict = {ops['pointclouds_phs'][0]: current_data[start_idx_0:end_idx_0, :, :],
                #  ops['pointclouds_phs'][1]: current_data[start_idx_1:end_idx_1, :, :],
                 ops['labels_phs'][0]: current_label[start_idx_0:end_idx_0],
                #  ops['labels_phs'][1]: current_label[start_idx_1:end_idx_1],
                 ops['is_training_phs'][0]: is_training,
                #  ops['is_training_phs'][1]: is_training
                }
    summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
    train_writer.add_summary(summary, step)
    # summary1 = tf.Summary(
    #     value=[tf.Summary.Value(tag='iou', simple_value=step)])
    # train_writer.add_summary(summary1, step)

    pred_val = np.argmax(pred_val, 2)
    correct = np.sum(pred_val == current_label[start_idx_0:end_idx_0])
    total_correct += correct
    total_seen += (BATCH_SIZE*NUM_POINT)
    loss_sum += loss_val
  
    # print(pred_val.shape, current_data.shape, start_idx_0, end_idx_0)
    if VIS and batch_idx % 50 == 0:
      label_gt = current_label[start_idx_0]
      pc_val = current_data[start_idx_0]
      label_pred = pred_val[0]

      gt_pc = pc_val[np.where(label_gt>0)[0], :]
      pred_pc = pc_val[np.where(label_pred>0)[0], :]
      
      # if len(gt_pc)==0:
      # gt_pc = np.array([[0, 0, 0]])
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
