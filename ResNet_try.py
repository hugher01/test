# 利用TFRecorder格式数据进行网络训练。
#  -*- coding:UTF-8 -*-
import collections
import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
import math
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

# ----------------------------------------------------------------------------
# -----------------------ResNet架构-------------------------------------------
slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    '''''
    使用collections.namedtuple设计ResNet基本block模块组的named tuple,
    定义一个典型的Block需要输入三个参数：
    scope：  Block的名称
    unit_fn：ResNet V2中的残差学习单元
    args：   它决定该block有几层,格式是[(depth, depth_bottleneck, stride)]

    示例：Block('block1', bottleneck, [(256,64,1),(256,64,1),(256,64,2)])
    '''


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    """
    if stride>1, then we do explicit zero-padding, followed by conv2d with 'VALID' padding
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)
        # ---------------------定义堆叠Blocks的函数-------------------


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    """
    Args:
      net: A Tensor of size [batch, height, width, channels].输入
      blocks: 是之前定义的Block的class的列表。
      outputs_collections: 收集各个end_points的collections

    Returns:
      net: Output tensor
    """
    # 循环Block类对象的列表blocks,即逐个Residual Unit地堆叠
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net, depth=unit_depth, depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


# 创建ResNet通用的arg_scope,arg_scope用来定义某些函数的参数默认值
def resnet_arg_scope(is_training=True,  # 训练标记
                     weight_decay=0.0001,  # 权重衰减速率
                     batch_norm_decay=0.997,  # BN的衰减速率
                     batch_norm_epsilon=1e-5,  # BN的epsilon默认1e-5
                     batch_norm_scale=True):  # BN的scale默认值

    batch_norm_params = {  # 定义batch normalization（标准化）的参数字典
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(  # 通过slim.arg_scope将[slim.conv2d]的几个默认参数设置好
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),  # 权重正则器设置为L2正则
            weights_initializer=slim.variance_scaling_initializer(),  # 权重初始化器
            activation_fn=tf.nn.relu,  # 激活函数
            normalizer_fn=slim.batch_norm,  # 标准化器设置为BN
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc  # 最后将基层嵌套的arg_scope作为结果返回


# ------------------定义核心的bottleneck残差学习单元--------------------
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    """
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth、depth_bottleneck:、stride三个参数是前面blocks类中的args
      rate: An integer, rate for atrous convolution.
      outputs_collections: 是收集end_points的collection
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)  # 最后一个维度,即输出通道数
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if depth == depth_in:
            # 如果残差单元的输入通道数和输出通道数一致，那么按步长对inputs进行降采样
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            # 如果不一样就按步长和1*1的卷积改变其通道数，使得输入、输出通道数一致
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

            # 先是一个1*1尺寸，步长1，输出通道数为depth_bottleneck的卷积
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        # 然后是3*3尺寸，步长为stride，输出通道数为depth_bottleneck的卷积
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
        # 最后是1*1卷积，步长1，输出通道数depth的卷积，得到最终的residual。最后一层没有正则项也没有激活函数
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        # 将降采样的结果和residual相加
        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

        # -------------------定义生成resnet_v2网络的主函数------------------


def resnet_v2(inputs, blocks, num_classes=None, global_pool=True,
              include_root_block=True, reuse=None, scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense],
                            outputs_collections=end_points_collection):

            net = inputs
            if include_root_block:  # 根据标记值,创建resnet最前面的64输出通道的步长为2的7*7卷积,然后接最大池化
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                # 经历过两个步长为2的层图片缩为1/4

            net = stack_blocks_dense(net, blocks)  # 将残差学习模块组生成好
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

            if global_pool:
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)  # tf.reduce_mean实现全局平均池化效率比avg_pool高

            if num_classes is not None:  # 是否有通道数
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,  # 无激活函数和正则项
                                  normalizer_fn=None, scope='logits')  # 添加一个输出通道num_classes的1*1的卷积
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)  # 将collection转化为python的dict

            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')  # 输出网络结果
                # logits=slim.softmax(net, scope='predictions')
            return net, end_points

            # -------------------建立模型 ResNet-50/101/152/200 model--------------------


# unit提升的主要场所是block3
def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=None,
                  scope='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""

    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def inference(image_batch, num_classes=None):
    # 预测网络
    # logit = None
    with slim.arg_scope(resnet_arg_scope(is_training=True)):
        net, end_points = resnet_v2_152(image_batch, num_classes)
        # net, end_points = resnet_v2_152(image_batch, 1000, global_pool=True, reuse=None,
        #                             scope='resnet_v2_152')
    logit = end_points['predictions']
    return logit


def calc_loss(logit, label_batch):
    # 计算损失函数
    cross_entropy_mean = None
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label_batch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean


def read_and_decode(filename, image_size):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从 TFRecord 读取内容并保存到 serialized_example 中
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(  # 读取 serialized_example 的格式
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )  # 解析从 serialized_example 读取到的内容
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [image_size, image_size, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label


if __name__ == '__main__':
    learning_rate = 0.01
    num_batches = 40
    batch_size = 2
    image_size = 299
    num_class = 5

    total_duration = 0.0
    total_duration_squared = 0.0

    OneCoder = OneHotEncoder()
    LabelCoder = LabelEncoder()

    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    LabelCoder.fit(classes)
    list0=LabelCoder.transform(classes)
    # print(LabelCoder.inverse_transform(0))
    # list2 = list([i] for i in list0.tolist())
    # OneCoder.fit(list2)

    # image_batch = tf.random_uniform((batch_size, image_size, image_size, 3))

    image, label = read_and_decode('data.tfrecords', image_size)
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity=10 * batch_size,
                                                      min_after_dequeue=2 * batch_size)
    # num_threads=1)

    logit = inference(image_batch, num_classes=num_class)

    label_batch_tmp = tf.one_hot(label_batch, num_class, 1, 0)
    loss = calc_loss(logit, label_batch_tmp)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#  TensorBoard
#     writer = tf.summary.FileWriter("F://python/try", tf.get_default_graph())
#     writer.close()

    #cmd:  tensorboard --logdir=F://python/try

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程  # 启动QueueRunner, 此时文件名队列已经进队。
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(num_batches):
            start_time = time.time()
            image_train, label_train = sess.run([image_batch, label_batch])

            # label_batch_tmp = OneCoder.transform(label_train).toarray()
            # train_step = sess.run(train_step,feed_dict={label_batch:label_batch_tmp})

            sess.run(logit)
            print(logit)

            sess.run(loss)
            print(loss.eval())

            duration = time.time() - start_time
            if not i % 10:
                print('%s:step %d,duration = %.3f' % (datetime.now(), i, duration))
                total_duration += duration
                total_duration_squared += duration * duration

        mn = total_duration / num_batches
        vr = total_duration_squared / num_batches - mn * mn
        sd = math.sqrt(vr)
        print('%s: across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), num_batches, mn, sd))

        coord.request_stop()
        coord.join(threads)
