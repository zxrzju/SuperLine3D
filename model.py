import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util

def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32,
                   shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32,
                             shape=(batch_size, num_point))
  return pointclouds_pl, labels_pl

def feature_encoder(point_cloud, is_training, k=20, stride=1, scale_invariant=False, bn_decay=False):
  """ ConvNet baseline, input is BxNx9 gray image """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  input_image = tf.expand_dims(point_cloud, -1)

  k = 20
  adj0 = tf_util.pairwise_distance(point_cloud)
  nn_idx0 = tf_util.knn(adj0, k=k)  # (batch, num_points, k)

  if scale_invariant:
    neighb0_xyz = gather_neighbour(point_cloud, nn_idx0)
    xyz_tile = tf.tile(tf.expand_dims(point_cloud, axis=2), [1, 1, tf.shape(nn_idx0)[-1], 1])
    relative_xyz = (xyz_tile - neighb0_xyz)
    relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True)) # # BxNxkx1
    relative_dis_sum = tf.reduce_sum(relative_dis, axis=-2, keepdims=True) + 1e-7
    relative_xyz_sum = tf.reduce_sum(relative_xyz, axis=-2, keepdims=True)
    
    feature0 = tf.divide(relative_xyz_sum, relative_dis_sum)
    feature = tf.tile(feature0, [1, 1, k, 1])
    relative_xyz_dis = tf.divide(relative_xyz, relative_dis_sum)
    edge_feature = tf.concat([feature, relative_xyz_dis - feature], axis=-1) # # rel 6
  else:
    edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx0, k=k)


  out1 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=0.0,
                       scope='adj_conv1', bn_decay=bn_decay, is_dist=True)
  
  out2 = tf_util.conv2d(out1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=0.0,
                       scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

  net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)

  # k=60
  adj = tf_util.pairwise_distance(net_1)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

  out3 = tf_util.conv2d(edge_feature, 64, [1, 1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=0.0,
                       scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

  out4 = tf_util.conv2d(out3, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=0.0,
                       scope='adj_conv4', bn_decay=bn_decay, is_dist=True)
  
  net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)

  adj = tf_util.pairwise_distance(net_2)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

  out5 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=0.0,
                       scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

  out6 = tf_util.conv2d(out5, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=0.0,
                       scope='adj_conv6', bn_decay=bn_decay, is_dist=True)

  net_3 = tf.reduce_max(out6, axis=-2, keep_dims=True)

  out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')

  expand = tf.tile(out_max, [1, num_point, 1, 1])

  concat = tf.concat(axis=3, values=[expand, 
                                     net_1,
                                     net_2,
                                     net_3])
  return concat

def decoder(feat, is_training, scope, out_dim):
    # CONV
  net = tf_util.conv2d(feat, 512, [1, 1], padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training, scope=scope+'/conv1', is_dist=True)
  net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training, scope=scope+'/conv2', is_dist=True)
  net = tf_util.dropout(
      net, keep_prob=0.7, is_training=is_training, scope=scope+'/dp1')
  net = tf_util.conv2d(net, out_dim, [1, 1], padding='VALID', stride=[1, 1],
                        is_training=is_training, activation_fn=None, scope=scope+'/conv3', is_dist=True)
  net = tf.squeeze(net, [2])
  return net

def get_seg_model(point_cloud, is_training, k=20, stride=1, scale_invariant=False, bn_decay=None):
  feature = feature_encoder(point_cloud, is_training, k, stride, scale_invariant, bn_decay)
  seg = decoder(feature, is_training, 'seg', 2)
  return seg

def get_superline3d_model(point_cloud, is_training, k=20, stride=4, desc_dim=64, scale_invariant=False, bn_decay=None):
  feature = feature_encoder(point_cloud, is_training, k, stride, scale_invariant, bn_decay)
  seg = decoder(feature, is_training, 'seg', 2)
  desp = decoder(feature, is_training, 'desp', desc_dim)
  return seg, desp

def get_loss(pred, label):
  """ pred: B,N,13; label: B,N """
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
  return tf.reduce_mean(loss)

def get_seg_loss(pred, label, pre_cal_weights):
    # print(pred)
    logits = tf.reshape(pred, [-1, 2])
    labels = tf.reshape(label, [-1])
    class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=2)
    weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=one_hot_labels)
    weighted_losses = unweighted_losses * weights
    output_loss = tf.reduce_mean(weighted_losses)
    return output_loss

def gather_neighbour(pc, neighbor_idx):
    # gather the coordinates or features of neighboring points
    batch_size = tf.shape(pc)[0]
    num_points = tf.shape(neighbor_idx)[1]
    d = pc.get_shape()[2].value
    index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
    features = tf.batch_gather(pc, index_input)
    features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
    return features


def discriminative_loss_single(prediction, correct_label_in, feature_dim,
                               delta_v, delta_d, param_var, param_dist, param_reg):
  ''' Discriminative loss for a single prediction/label pair.
  :param prediction: inference of network
  :param correct_label: instance label
  :feature_dim: feature dimension of prediction
  :param label_shape: shape of label
  :param delta_v: cutoff variance distance
  :param delta_d: curoff cluster distance
  :param param_var: weight for intra cluster variance
  :param param_dist: weight for inter cluster distances
  :param param_reg: weight regularization
  '''

  ### Reshape so pixels are aligned along a vector
  #correct_label = tf.reshape(correct_label, [label_shape[1] * label_shape[0]])
  reshaped_pred = tf.reshape(prediction, [-1, feature_dim])
  correct_label = tf.reshape(correct_label_in, [tf.shape(reshaped_pred)[0]])
  # print(correct_label_in.shape, correct_label.shape)

  ### Count instances
  unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)

  counts = tf.cast(counts, tf.float32)
  num_instances = tf.size(unique_labels)
  
  segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)

  mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
  mu_expand = tf.gather(mu, unique_id)

  ### Calculate l_var
  #distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
  #tmp_distance = tf.subtract(reshaped_pred, mu_expand)
  tmp_distance = reshaped_pred - mu_expand
  distance = tf.norm(tmp_distance, ord=1, axis=1)

  distance = tf.subtract(distance, delta_v)
  distance = tf.clip_by_value(distance, 0., distance)
  distance = tf.square(distance)

  l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)

  labels_sort = tf.argsort(tf.cast(unique_labels, tf.int32))
  l_var = tf.gather(l_var, labels_sort[1:])
  counts = tf.gather(counts, labels_sort[1:])

  l_var = tf.div(l_var, counts)

  # line_idx = tf.where(tf.greater(unique_labels, 0))
  # l_var_line = tf.reshape(tf.gather(l_var, line_idx), [-1])
  # l_var_top_k, _ = tf.nn.top_k(l_var_line, tf.cast(num_instances/4 + 1, tf.int32))
  # l_var = tf.reduce_mean(l_var_top_k)

  # l_var = tf.reduce_sum(l_var)
  # l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

  mask0 = tf.greater(unique_labels, 0)
  l_var = tf.boolean_mask(l_var, mask0)
  l_var = tf.reduce_sum(l_var)
  l_var = tf.divide(l_var, tf.cast(num_instances-1, tf.float32))

  ### Calculate l_dist

  # Get distance for each pair of clusters like this:
  #   mu_1 - mu_1
  #   mu_2 - mu_1
  #   mu_3 - mu_1
  #   mu_1 - mu_2
  #   mu_2 - mu_2
  #   mu_3 - mu_2
  #   mu_1 - mu_3
  #   mu_2 - mu_3
  #   mu_3 - mu_3

  mu = tf.boolean_mask(mu, mask0)

  mu_interleaved_rep = tf.tile(mu, [num_instances-1, 1])
  mu_band_rep = tf.tile(mu, [1, num_instances-1])
  mu_band_rep = tf.reshape(mu_band_rep, ((num_instances-1) * (num_instances-1), feature_dim))

  mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

  # Filter out zeros from same cluster subtraction
  eye = tf.eye(num_instances-1)
  zero = tf.zeros(1, dtype=tf.float32)
  diff_cluster_mask = tf.equal(eye, zero)
  diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])
  mu_diff_bool = tf.boolean_mask(mu_diff, diff_cluster_mask)

  #intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
  #zero_vector = tf.zeros(1, dtype=tf.float32)
  #bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
  #mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

  mu_norm = tf.norm(mu_diff_bool, ord=1, axis=1)
  mu_norm = tf.subtract(2. * delta_d, mu_norm)
  mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
  mu_norm = tf.square(mu_norm)

  # norm_k_cnt = tf.cond(tf.greater(num_instances*5, num_instances*num_instances-num_instances), lambda:(num_instances*num_instances-num_instances), lambda:(num_instances*5))
  # mu_norm_top_k , _ = tf.nn.top_k(mu_norm, tf.cast(norm_k_cnt, tf.int32))
  # l_dist = tf.reduce_mean(mu_norm_top_k)
  l_dist = tf.reduce_mean(mu_norm)

  def rt_0(): return 0.
  def rt_l_dist(): return l_dist
  l_dist = tf.cond(tf.equal(1, num_instances), rt_0, rt_l_dist)
  
  ### Calculate l_reg
  l_reg = tf.reduce_mean(tf.norm(mu, ord=1, axis=1))

  param_scale = 1.
  l_var = param_var * l_var
  l_dist = param_dist * l_dist
  l_reg = param_reg * l_reg

  loss = param_scale * (l_var + l_dist + 0*l_reg)

  return loss, l_var, l_dist, l_reg

def discriminative_loss_single2(prediction, correct_label_in, feature_dim,
                               delta_v, delta_d, param_var, param_dist, param_reg):
  ''' Discriminative loss for a single prediction/label pair.
  :param prediction: inference of network
  :param correct_label: instance label
  :feature_dim: feature dimension of prediction
  :param label_shape: shape of label
  :param delta_v: cutoff variance distance
  :param delta_d: curoff cluster distance
  :param param_var: weight for intra cluster variance
  :param param_dist: weight for inter cluster distances
  :param param_reg: weight regularization
  '''

  ### Reshape so pixels are aligned along a vector
  #correct_label = tf.reshape(correct_label, [label_shape[1] * label_shape[0]])
  reshaped_pred = tf.reshape(prediction, [2, -1, feature_dim])
  correct_label = tf.reshape(correct_label_in, [2, -1])

  # print(correct_label_in.shape, correct_label.shape)
  correct_label0 = correct_label[0]
  reshaped_pred0 = reshaped_pred[0]
  ### Count instances
  unique_labels, unique_id, counts = tf.unique_with_counts(correct_label0)
  counts = tf.cast(counts, tf.float32)
  num_instances = tf.size(unique_labels)
  segmented_sum = tf.unsorted_segment_sum(reshaped_pred0, unique_id, num_instances)

  labels_sort = tf.argsort(tf.cast(unique_labels, tf.int32))
  segmented_sum = tf.gather(segmented_sum, labels_sort[1:])
  counts = tf.gather(counts, labels_sort[1:])

  mu0 = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))


  # print(correct_label_in.shape, correct_label.shape)
  correct_label1 = correct_label[1]
  reshaped_pred1 = reshaped_pred[1]
  ### Count instances
  unique_labels, unique_id, counts = tf.unique_with_counts(correct_label1)
  counts = tf.cast(counts, tf.float32)
  segmented_sum = tf.unsorted_segment_sum(reshaped_pred1, unique_id, num_instances)

  labels_sort = tf.argsort(tf.cast(unique_labels, tf.int32))
  segmented_sum = tf.gather(segmented_sum, labels_sort[1:])
  counts = tf.gather(counts, labels_sort[1:])

  mu1 = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))


  ### Calculate l_dist

  # Get distance for each pair of clusters like this:
  #   mu_1 - mu_1
  #   mu_2 - mu_1
  #   mu_3 - mu_1
  #   mu_1 - mu_2
  #   mu_2 - mu_2
  #   mu_3 - mu_2
  #   mu_1 - mu_3
  #   mu_2 - mu_3
  #   mu_3 - mu_3

  mu_interleaved_rep = tf.tile(mu0, [num_instances-1, 1])
  mu_band_rep = tf.tile(mu1, [1, num_instances-1])
  mu_band_rep = tf.reshape(mu_band_rep, ((num_instances-1) * (num_instances-1), feature_dim))

  mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

  # Filter out zeros from same cluster subtraction
  eye = tf.eye(num_instances-1)
  zero = tf.zeros(1, dtype=tf.float32)
  one = tf.ones(1, dtype=tf.float32)
  same_cluster_mask = tf.equal(eye, one)
  same_cluster_mask = tf.reshape(same_cluster_mask, [-1])
  diff_cluster_mask = tf.equal(eye, zero)
  diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])

  mu_same_bool = tf.boolean_mask(mu_diff, same_cluster_mask)
  
  mu_same_norm = tf.norm(mu_same_bool, ord=1, axis=1)
  mu_same_norm = tf.subtract(mu_same_norm, delta_v)
  mu_same_norm = tf.clip_by_value(mu_same_norm, 0., mu_same_norm)
  mu_same_norm = tf.square(mu_same_norm)

  # same_norm_k_cnt = tf.cond(tf.greater(num_instances*5, num_instances), lambda:(num_instances), lambda:(num_instances*5))
  # mu_same_norm_top_k , _ = tf.nn.top_k(mu_same_norm, tf.cast(same_norm_k_cnt, tf.int32))
  # l_var = tf.reduce_mean(mu_same_norm_top_k)


  # l_var = tf.reduce_mean(mu_same_norm)

  # mask0 = tf.cast(tf.greater(unique_labels, 0), tf.float32)
  # l_var = tf.reduce_sum(mu_same_norm*mask0)

  l_var = tf.reduce_sum(mu_same_norm)
  l_var = tf.divide(l_var, tf.cast(num_instances-1, tf.float32))

  mu_diff_bool = tf.boolean_mask(mu_diff, diff_cluster_mask)

  mu_norm = tf.norm(mu_diff_bool, ord=1, axis=1)
  mu_norm = tf.subtract(2. * delta_d, mu_norm)
  mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
  mu_norm = tf.square(mu_norm)

  # norm_k_cnt = tf.cond(tf.greater(num_instances*5, num_instances*num_instances-num_instances), lambda:(num_instances*num_instances-num_instances), lambda:(num_instances*5))
  # mu_norm_top_k , _ = tf.nn.top_k(mu_norm, tf.cast(norm_k_cnt, tf.int32))

  # l_dist = tf.reduce_mean(mu_norm_top_k)
  l_dist = tf.reduce_mean(mu_norm)


  def rt_0(): return 0.
  def rt_l_dist(): return l_dist
  l_dist = tf.cond(tf.equal(1, num_instances), rt_0, rt_l_dist)
  
  ### Calculate l_reg
  # l_reg = tf.reduce_mean(tf.norm(mu, ord=1, axis=1))

  param_scale = 1.
  l_var = param_var * l_var
  l_dist = param_dist * l_dist
  l_reg = param_reg * 0

  loss = param_scale * (l_var + l_dist + l_reg)

  return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim,
                        delta_v, delta_d, param_var, param_dist, param_reg):
  ''' Iterate over a batch of prediction/label and cumulate loss
  :return: discriminative loss and its three components
  '''

  def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
      return tf.less(i, tf.shape(batch)[0])

  def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
      disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim,
                                                                    delta_v, delta_d, param_var, param_dist, param_reg)

      out_loss = out_loss.write(i, disc_loss)
      out_var = out_var.write(i, l_var)
      out_dist = out_dist.write(i, l_dist)
      out_reg = out_reg.write(i, l_reg)

      return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

  # TensorArray is a data structure that support dynamic writing
  output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                  size=0,
                                  dynamic_size=True)
  output_ta_var = tf.TensorArray(dtype=tf.float32,
                                  size=0,
                                  dynamic_size=True)
  output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                  size=0,
                                  dynamic_size=True)
  output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                  size=0,
                                  dynamic_size=True)

  _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [correct_label,
                                                                                          prediction,
                                                                                          output_ta_loss,
                                                                                          output_ta_var,
                                                                                          output_ta_dist,
                                                                                          output_ta_reg,
                                                                                          0])
  out_loss_op = out_loss_op.stack()
  out_var_op = out_var_op.stack()
  out_dist_op = out_dist_op.stack()
  out_reg_op = out_reg_op.stack()

  disc_loss = tf.reduce_mean(out_loss_op)
  l_var = tf.reduce_mean(out_var_op)
  l_dist = tf.reduce_mean(out_dist_op)
  l_reg = tf.reduce_mean(out_reg_op)

  return disc_loss, l_var, l_dist, l_reg


def discriminative_loss2(prediction, correct_label, feature_dim,
                         delta_v, delta_d, param_var, param_dist, param_reg):
  ''' Iterate over a batch of prediction/label and cumulate loss
  :return: discriminative loss and its three components
  '''

  def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
      return tf.less(2*i, tf.shape(batch)[0])

  def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
      disc_loss, l_var, l_dist, l_reg = discriminative_loss_single2(prediction[2*i:2*(i+1)], correct_label[2*i:2*(i+1)], feature_dim,
                                                                    delta_v, delta_d, param_var, param_dist, param_reg)

      out_loss = out_loss.write(i, disc_loss)
      out_var = out_var.write(i, l_var)
      out_dist = out_dist.write(i, l_dist)
      out_reg = out_reg.write(i, l_reg)

      return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

  # TensorArray is a data structure that support dynamic writing
  output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                  size=0,
                                  dynamic_size=True)
  output_ta_var = tf.TensorArray(dtype=tf.float32,
                                  size=0,
                                  dynamic_size=True)
  output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                  size=0,
                                  dynamic_size=True)
  output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                  size=0,
                                  dynamic_size=True)

  _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [correct_label,
                                                                                          prediction,
                                                                                          output_ta_loss,
                                                                                          output_ta_var,
                                                                                          output_ta_dist,
                                                                                          output_ta_reg,
                                                                                          0])
  out_loss_op = out_loss_op.stack()
  out_var_op = out_var_op.stack()
  out_dist_op = out_dist_op.stack()
  out_reg_op = out_reg_op.stack()

  disc_loss = tf.reduce_mean(out_loss_op)
  l_var = tf.reduce_mean(out_var_op)
  l_dist = tf.reduce_mean(out_dist_op)
  l_reg = tf.reduce_mean(out_reg_op)

  return disc_loss, l_var, l_dist, l_reg

def get_desc_loss(desp, ins_label):

  feature_dim = desp.get_shape()[-1]
  delta_v = 0.2
  delta_d = 1.0
  param_var = 1.
  param_dist = 1.
  param_reg = 0.0001

  disc_loss, l_var, l_dist, l_reg = discriminative_loss2(desp, ins_label, feature_dim,
                                                        delta_v, delta_d, param_var, param_dist, param_reg)

  disc_loss0, l_var0, l_dist0, l_reg0 = discriminative_loss(desp, ins_label, feature_dim,
                                                        delta_v, delta_d, param_var, param_dist, param_reg)

  # disc_loss, l_var, l_dist, l_reg = disc_loss0, l_var0, l_dist0, l_reg0
  return disc_loss, l_var, l_dist, l_reg, disc_loss0, l_var0, l_dist0, l_reg0
