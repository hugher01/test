#  -*- coding:UTF-8 -*-
import collections
import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from PIL import Image


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
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return img, label


def print_activtions(t):
    print(t.op.name, '', t.get_shape().as_list())


def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activtions(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activtions(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activtions(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activtions(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activtions(pool5)

    return pool5, parameters


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__ == '__main__':
    learning_rate = 0.01
    num_batches = 100
    batch_size = 18
    image_size = 224
    # num_class = 5

    total_duration = 0.0
    total_duration_squared = 0.0

    OneCoder = OneHotEncoder()
    LabelCoder = LabelEncoder()

    # classes = ['daisy']
    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    LabelCoder.fit(classes)
    list0 = LabelCoder.transform(classes)
    # print(LabelCoder.inverse_transform(0))
    # list2 = list([i] for i in list0.tolist())
    # OneCoder.fit(list2)

    # image_batch = tf.random_uniform((batch_size, image_size, image_size, 3))

    image, label = read_and_decode('data.tfrecords', image_size)
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity=10 * batch_size,
                                                      min_after_dequeue=2 * batch_size)
    # num_threads=1)

    pool5, parameters = inference(image_batch)
    objective = tf.nn.l2_loss(pool5)
    grad = tf.gradients(objective, parameters)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程  # 启动QueueRunner, 此时文件名队列已经进队。
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(num_batches):
            start_time = time.time()
            sess.run(grad)
            duration = time.time() - start_time

            if step % 10 == 0:
                example_per_sec = batch_size / duration
                sec_per_batch = float(duration)

                img1, label1 = sess.run([image_batch, label_batch])
                for i in range(0,batch_size,3):
                    img1 = sess.run(tf.cast(img1, tf.uint8))
                    ar = Image.fromarray(img1[i], 'RGB')
                    Image._show(ar)
                    print('Reading Pictures:', i, '@ step:', step)

                format_str = ('step %d,loss= %.2f (%.1f example/sec; %.3f sec/batch)')
                print(format_str % (step, sess.run(objective), example_per_sec, sec_per_batch))

        coord.request_stop()
        coord.join(threads)
