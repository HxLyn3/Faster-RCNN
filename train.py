# python lib
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

# my lib
from data.imdb import imdb as imdb2
from data import roidb as rdl_roidb
from data.pascal_voc import pascal_voc
from network.Faster_RCNN import Network
from data.roi_data_layer import RoIDataLayer

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if True:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb

def train():
    # datasets
    imdb = pascal_voc('trainval', '2007', os.path.join(os.getcwd(), 'data', 'VOCdevkit2007'))
    roidb = get_training_roidb(imdb)
    data_layer = RoIDataLayer(roidb, imdb.num_classes)

    # network
    net = Network(num_classes=imdb.num_classes)

    # create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    tfconfig = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=tfconfig)

    with sess.graph.as_default():
        # random seed
        tf.set_random_seed(3)

        layers = net.createArchitecture(sess)
        loss = layers['total_loss']                 # total loss of network
        lr = tf.Variable(0.001, trainable=False)    # learning rate
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        gvs = optimizer.compute_gradients(loss)     # gradients and vars

        # double bias
        final_gvs = []
        with tf.variable_scope('Gradient_Multiply'):
            for grad, var in gvs:
                scale = 1.
                if '/biases:' in var.name:
                    scale *= 2
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gvs.append((grad, var))
        train_op = optimizer.apply_gradients(final_gvs)
        # train_op = optimizer.apply_gradients(gvs)

        saver = tf.train.Saver(max_to_keep=100000)      # saver

        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))

        # load pretrained model
        print('Loading initial model weights from {:s}'.format('./model/pretrained/vgg_16.ckpt'))
        var_keep_dic = get_variables_in_ckpt('./model/pretrained/vgg_16.ckpt')
        variables_to_restore = net.get_variables_to_restore(variables, var_keep_dic)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, './model/pretrained/vgg_16.ckpt')
        print('Loaded.')

        # fix variables
        net.fix_variables(sess, './model/pretrained/vgg_16.ckpt')
        print('Fixed.')
        sess.run(tf.assign(lr, 0.001))

        snapshot_iter = 0
        init_time = time.time()

        # train 40000 times
        while snapshot_iter < 40000:
            # reduce learning rate when trained 30000 times
            if snapshot_iter == 30000:
                sess.run(tf.assign(lr, 0.0001))

            t0 = time.time()

            # get training data, one batch at a time
            blobs = data_layer.forward()

            # train without summary
            rpn_cls_loss, rpn_bbox_loss, cls_loss, bbox_loss, total_loss = net.train_step(sess, blobs, train_op)

            t1 = time.time()
            snapshot_iter += 1

            # display train info
            if snapshot_iter % 10 == 0:
                print("iter: %d / 40000, total loss: %.6f\n >>> rpn_cls_loss: %.6f\n"\
                    " >>> rpn_bbox_loss: %.6f\n >>> cls_loss: %.6f\n >>> bbox_loss: %.6f\n" % \
                    (snapshot_iter, total_loss, rpn_cls_loss, rpn_bbox_loss, cls_loss, bbox_loss))
                print("speed: {:.3f}s/iter".format((t1-init_time)/snapshot_iter))

            # snapshot (store a ckpt)
            if snapshot_iter % 5000 == 0:
                snapshot(net, sess, saver, snapshot_iter)

def get_variables_in_ckpt(filename):
    """ get variables in checkpoint file """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(filename)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed with SNAPPY.")

def snapshot(net, sess, saver, it):
    """ take a snapshot for model """
    # sotre the model snapshot
    filename = './model/vgg16_faster_rcnn_iter_{:d}'.format(it) + '.ckpt'
    saver.save(sess, filename)
    print("Wrote snapshot to: {:s}".format(filename))

if __name__ == "__main__":
    train()