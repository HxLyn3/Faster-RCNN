# python lib
import numpy as np
import tensorflow as tf 
import tensorflow.contrib.slim as slim

# my lib
from .lib.rpn.anchors import generate_anchors, anchor_target
from .lib.rpn.roi import proposal, proposal_top, proposal_target

class Network():
    """ A Faster RCNN network """

    # ····································································································
    # !                                      Init Network                                                ！
    # ····································································································

    def __init__(self, num_classes, batch_size=1, anchor_ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32), mode="train"):
        """ Init network """
        self.num_classes = num_classes

        self.batch_size = batch_size
        self.layers = {}
        self.predictions = {}
        self.losses = {}

        # input image placeholder
        self.img = tf.placeholder(tf.float32, shape=[self.batch_size, None, None, 3])
        self.imgInfo = tf.placeholder(tf.float32, shape=[self.batch_size, 3])
        self.boxes = tf.placeholder(tf.float32, shape=[None, 5])

        # targets (or labels)
        self.anchor_targets = {}
        self.proposal_targets = {}

        # feature info
        self.feat_stride = 16
        self.feat_compress = 1/16

        # anchors info
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.ratios_num = len(self.anchor_ratios)
        self.scales_num = len(self.anchor_scales)
        self.anchor_num = self.ratios_num*self.scales_num

        # summaries
        self.summaries_act = []     # act of nn, such as CNN, RPN
        self.summaries_score = {}   # targets and predictions
        self.summaries_train = []   # trainable var
        self.summaries_event = {}   # losses

        # ?
        self.variable_to_fix = {}

        # train or test
        self.mode = mode

    # ····································································································
    # !                                  Network Architecture                                            ！
    # ····································································································

    def createArchitecture(self, sess):
        """ Create architecture of Faster RCNN """
        # mode
        isTraining = self.mode == 'train'
        isTesting = self.mode == 'test'

        # regularizer
        weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
        biases_regularizer = tf.no_regularizer

        # build Faster RCNN network
        with slim.arg_scope(
            [slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
            weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer, biases_initializer=tf.constant_initializer(0.0)):
            rois, cls_prob, bbox_pred = self.buildNetwork(sess, isTraining)

        # output layer
        output_layers = {'rois': rois}
        output_layers.update(self.predictions)

        # trainable var
        for var in tf.trainable_variables():
            self.summaries_train.append(var)

        if self.mode == 'test':
            stds = np.tile(np.array((0.1, 0.1, 0.1, 0.1)), (self.num_classes))
            means = np.tile(np.array((0.0, 0.0, 0.0, 0.0)), (self.num_classes))
            self.predictions["bbox_pred"] *= stds
            self.predictions["bbox_pred"] += means
        # train
        else:
            self.add_losses()
            output_layers.update(self.losses)

        summaries_val = []
        with tf.device("/cpu:0"):
            summaries_val.append(self.add_summary_image(self.img, self.boxes))
            for key, var in self.summaries_event.items():
                summaries_val.append(tf.summary.scalar(key, var))
            for key, var in self.summaries_score.items():
                tf.summary.histogram('SCORE/' + var.op.name + '/' + key + '/scores', var)
            for var in self.summaries_act:
                tf.summary.histogram('ACT/' + var.op.name + '/activations', var)
                tf.summary.scalar('ACT/' + var.op.name + '/zero_fraction', tf.nn.zero_fraction(var))
            for var in self.summaries_train:
                tf.summary.histogram('TRAIN/' + var.op.name, var)

        self.summary_op = tf.summary.merge_all()
        if not isTesting:
            self.summary_op_val = tf.summary.merge(summaries_val)

        return output_layers

    def buildNetwork(self, sess, isTraining=True):
        """ Faster RCNN """
        with tf.variable_scope("vgg_16", "vgg_16"):
            # normal initializer
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)

            # cnn head
            net = self.vgg16_cnn(isTraining)
            # rpn
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.RPN(net, isTraining, initializer)
            # proposals
            rois = self.regionProposal(isTraining, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)
            # predictions
            cls_score, cls_prob, bbox_pred = self.predict(net, rois, isTraining, initializer, initializer_bbox)

            self.predictions["rpn_cls_score"] = rpn_cls_score
            self.predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self.predictions["rpn_cls_prob"] = rpn_cls_prob
            self.predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self.predictions["cls_score"] = cls_score
            self.predictions["cls_prob"] = cls_prob
            self.predictions["bbox_pred"] = bbox_pred
            self.predictions["rois"] = rois
            self.summaries_score.update(self.predictions)

            return rois, cls_prob, bbox_pred

    def vgg16_cnn(self, isTraining):
        """ using vgg16 as CNN """
        
        # layer 1 (2 conv layers and 1 pool layer)
        net = slim.repeat(self.img, 2, slim.conv2d, 64, [3, 3], trainable=False, scope="conv1")
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        # layer 2 (2 conv layers and 1 pool layer)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # layer 3 (3 conv layers and 1 pool layer)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=isTraining, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # layer 4 (3 conv layers and 1 pool layer)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=isTraining, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # layer 5 (3 conv layers)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=isTraining, scope='conv5')

        # append network to summaries
        self.summaries_act.append(net) # conv out (feature map)

        # append network to layers
        self.layers['conv'] = net

        return net
    
    def RPN(self, net, isTraining, initializer):
        with tf.variable_scope("Initial_Anchors"):
            # size of feature map
            height = tf.to_int32(tf.ceil(self.imgInfo[0, 0]/np.float32(self.feat_stride)))
            width = tf.to_int32(tf.ceil(self.imgInfo[0, 1]/np.float32(self.feat_stride)))
            
            # anchors
            anchors, n_anchors = tf.py_func(generate_anchors, 
                [height, width, self.feat_stride, self.anchor_ratios, np.array(self.anchor_scales)],
                [tf.float32, tf.int32], name="anchors")
            anchors.set_shape([None, 4])
            n_anchors.set_shape([])
            self.anchors = anchors
            self.anchor_num_total = n_anchors
        
        # RPN layer
        rpn = slim.conv2d(net, 512, [3, 3], trainable=isTraining, weights_initializer=initializer, scope="rpn_conv")
        self.summaries_act.append(rpn)

        # rpn_cls_score:
        #     - scores of background @anchor0
        #     - scores of background @anchor1
        #     ......
        #     - scores of background @anchor_N-1
        #     - scores of foreground @anchor0
        #     - scores of foreground @anchor1
        #     ......
        #     - scores of foreground @anchor_N-1
        # PS: a point correponds to score of an anchor (with shift)
        rpn_cls_score = slim.conv2d(
            rpn, 2*self.anchor_num, [1, 1], trainable=isTraining, weights_initializer=initializer, 
            padding='VALID', activation_fn=None, scope='rpn_cls_score')

        # reshape
        rpn_cls_score_shape = tf.shape(rpn_cls_score)
        with tf.variable_scope("rpn_cls_score_reshape"):
            # rpn_cls_score_reshape:
            #                   ___________
            #     - background \ @ anchor0 \
            #                   \ @ anchor1 \
            #                    \ ......    \
            #                     \ @ anchor_N-1
            #                      ------------
            #                   ___________
            #     - foreground \ @ anchor0 \
            #                   \ @ anchor1 \
            #                    \ ......    \
            #                     \ @ anchor_N-1
            #                      ------------
            rpn_cls_score_caffe = tf.transpose(rpn_cls_score, [0, 3, 1, 2])
            # force to channel 2
            reshaped = tf.reshape(rpn_cls_score_caffe, [self.batch_size, 2, -1, rpn_cls_score_shape[2]])
            rpn_cls_score_reshape = tf.transpose(reshaped, [0, 2, 3, 1])

        # softmax
        shape = tf.shape(rpn_cls_score_reshape)
        rpn_cls_score_temp = tf.reshape(rpn_cls_score_reshape, [-1, shape[-1]]) # shape[-1] = 2
        rpn_cls_prob_reshape = tf.nn.softmax(rpn_cls_score_temp, name="rpn_cls_prob_reshape")
        rpn_cls_prob_reshape = tf.reshape(rpn_cls_prob_reshape, shape)

        # reshape back
        rpn_cls_prob_shape = tf.shape(rpn_cls_prob_reshape)
        with tf.variable_scope("rpn_cls_prob"):
            rpn_cls_prob_caffe = tf.transpose(rpn_cls_prob_reshape, [0, 3, 1, 2])
            # force to channel 2*a
            reshaped = tf.reshape(rpn_cls_prob_caffe, [self.batch_size, 2*self.anchor_num, -1, rpn_cls_prob_shape[2]])
            rpn_cls_prob = tf.transpose(reshaped, [0, 2, 3, 1])

        # bounding box prediction
        # rpn_bbox_pred:
        #     - Δx @anchor0 at each pos
        #     - Δy @anchor0 at each pos
        #     - Δw @anchor0 at each pos
        #     - Δh @anchor0 at each pos
        #     - ...@anchor1
        #     - ......
        rpn_bbox_pred = slim.conv2d(rpn, 4*self.anchor_num, [1, 1], trainable=isTraining, weights_initializer=initializer,
            padding="VALID", activation_fn=None, scope='rpn_bbox_pred')
        
        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def regionProposal(self, isTraining, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):
        """ region proposal """
        # train
        if isTraining:
            # generate ROIs
            with tf.variable_scope("ROIs"):
                rois, roi_scores = tf.py_func(proposal,
                    [rpn_cls_prob, rpn_bbox_pred, self.imgInfo[0], self.mode, self.feat_stride, self.anchors, self.anchor_num],
                    [tf.float32, tf.float32])
                rois.set_shape([None, 5])
                roi_scores.set_shape([None, 1])

            # calculate out labels for softmax and bbox of rpn
            with tf.variable_scope("anchor"):
                rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                    anchor_target, [rpn_cls_score, self.boxes, self.imgInfo[0], self.feat_stride, self.anchors, self.anchor_num],
                    [tf.float32, tf.float32, tf.float32, tf.float32])
                
                # set shape
                rpn_labels.set_shape([1, 1, None, None])
                rpn_labels = tf.to_int32(rpn_labels, name="rpn_labels_to_int32")
                rpn_bbox_targets.set_shape([1, None, None, self.anchor_num*4])
                rpn_bbox_inside_weights.set_shape([1, None, None, self.anchor_num*4])
                rpn_bbox_outside_weights.set_shape([1, None, None, self.anchor_num*4])

                # update
                self.anchor_targets['rpn_labels'] = rpn_labels
                self.anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
                self.anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
                self.anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
                self.summaries_score.update(self.anchor_targets)
            
            # take a deterministic order for the computing graph
            with tf.control_dependencies([rpn_labels]):
                with tf.variable_scope("rpn_rois"):
                    network_batch_size = 256
                    # calculate out labels for softmax and bbox at end
                    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
                        tf.py_func(proposal_target, [rois, roi_scores, self.boxes, self.num_classes, network_batch_size],
                        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
                    
                    # set shape
                    rois.set_shape([network_batch_size, 5])
                    roi_scores.set_shape([network_batch_size])
                    labels.set_shape([network_batch_size, 1])
                    bbox_targets.set_shape([network_batch_size, 4*self.num_classes])
                    bbox_inside_weights.set_shape([network_batch_size, 4*self.num_classes])
                    bbox_outside_weights.set_shape([network_batch_size, 4*self.num_classes])

                    # update
                    self.proposal_targets['rois'] = rois
                    self.proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
                    self.proposal_targets['bbox_targets'] = bbox_targets
                    self.proposal_targets['bbox_inside_weights'] = bbox_inside_weights
                    self.proposal_targets['bbox_outside_weights'] = bbox_outside_weights
                    self.summaries_score.update(self.proposal_targets)
        # test
        else:
            # generate ROIs
            with tf.variable_scope("ROIs"):
                topN = 300
                rois, roi_scores = tf.py_func(proposal_top,
                    [rpn_cls_prob, rpn_bbox_pred, self.imgInfo[0], self.feat_stride, self.anchors, self.anchor_num, topN],
                    [tf.float32, tf.float32])
                rois.set_shape([topN, 5])
                roi_scores.set_shape([topN, 1])

        return rois

    # ····································································································
    # !                                   Network Prediction                                             ！
    # ····································································································

    def predict(self, net, rois, isTraining, initializer, initializer_bbox):
        """
        Brief:
            Build predicition of class and bounding box for
        each ROIs.

        Arg:
            net: output of CNN (feature map)
            ROIs: just ROIs (0, left, top, right, bottom)
            initializer: initializer of class weights
            initializer_bbox: initializer of bounding box weights
        """

        with tf.variable_scope("pool5"):
            batch_idxs = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_idx"), [1])
            # normalize coordinates of bboxes
            map_shape = tf.shape(net)       # shape of feature map
            height = (tf.to_float(map_shape[1])-1.)*np.float32(self.feat_stride)
            width = (tf.to_float(map_shape[2])-1.)*np.float32(self.feat_stride)
            lefts = tf.slice(rois, [0, 1], [-1, 1], name="left")/width
            tops = tf.slice(rois, [0, 2], [-1, 1], name="top")/height
            rights = tf.slice(rois, [0, 3], [-1, 1], name="right")/width
            bottoms = tf.slice(rois, [0, 4], [-1, 1], name="bottom")/height
            # rois shouldn't be backpropagated anyway
            bboxes_ = tf.stop_gradient(tf.concat([tops, lefts, bottoms, rights], axis=1))
            roi_pooling_size = 7
            pre_pool_size  = roi_pooling_size*2
            # cut out image of ROI
            crops = tf.image.crop_and_resize(net, bboxes_, tf.to_int32(batch_idxs), [pre_pool_size, pre_pool_size], name="crops")
            # pool5
            pool5 = slim.max_pool2d(crops, [2, 2], padding='SAME')

        # [rois_num, roi_height, roi_width, channels_num] --> [rois_num, None]
        pool5_flat = slim.flatten(pool5, scope='flatten')

        # fully connected layers
        full6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
        if isTraining:
            # dropout when train
            full6 = slim.dropout(full6, keep_prob=0.5, is_training=True, scope='dropout6')
        
        full7 = slim.fully_connected(full6, 4096, scope='fc7')
        if isTraining:
            # dropout when train
            full7 = slim.dropout(full7, keep_prob=0.5, is_training=True, scope='dropout7')

        # scores
        cls_score = slim.fully_connected(full7, self.num_classes, \
            weights_initializer=initializer, trainable=isTraining, activation_fn=None, scope='class_score')
        # predictions
        cls_prob = tf.nn.softmax(cls_score, name='cls_prob')
        bbox_prediction = slim.fully_connected(full7, self.num_classes*4, \
            weights_initializer=initializer_bbox, trainable=isTraining, activation_fn=None, scope='bbox_pred')
        
        return cls_score, cls_prob, bbox_prediction

    # ····································································································
    # !                                     Network Losses                                               ！
    # ····································································································

    def add_losses(self, sigma_rpn=3.0):
        """ add losses for Faster RCNN """
        with tf.variable_scope('loss_default'):
            # class loss in RPN
            rpn_cls_score = tf.reshape(self.predictions['rpn_cls_score_reshape'], [-1, 2])
            rpn_label = tf.reshape(self.anchor_targets['rpn_labels'], [-1])
            # calculate cross entropy for anchors which is foreground or background
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(\
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # bbox loss in RPN
            rpn_bbox_pred = self.predictions['rpn_bbox_pred']
            rpn_bbox_targets = self.anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self.anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self.anchor_targets['rpn_bbox_outside_weights']
            rpn_bbox_loss = self.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,\
                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # class loss in RCNN
            cls_score = self.predictions["cls_score"]
            label = tf.reshape(self.proposal_targets["labels"], [-1])
            cross_entropy = tf.reduce_mean(\
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(cls_score, [-1, self.num_classes]), labels=label))

            # bbox loss in RCNN
            bbox_pred = self.predictions["bbox_pred"]
            bbox_targets = self.proposal_targets["bbox_targets"]
            bbox_inside_weights = self.proposal_targets["bbox_inside_weights"]
            bbox_outside_weights = self.proposal_targets["bbox_outside_weights"]
            bbox_loss = self.smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self.losses['rpn_cross_entropy'] = rpn_cross_entropy
            self.losses['rpn_bbox_loss'] = rpn_bbox_loss
            self.losses['cross_entropy'] = cross_entropy
            self.losses['bbox_loss'] = bbox_loss
            loss = rpn_cross_entropy + rpn_bbox_loss + cross_entropy + bbox_loss
            self.losses['total_loss'] = loss
            self.summaries_event.update(self.losses)

        return loss

    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        """ calculate smooth l1 loss """
        sigma_2 = sigma**2
        # difference
        bbox_diff = bbox_pred - bbox_targets
        bbox_diff_inside = bbox_inside_weights*bbox_diff    # only consider positive samples
        bbox_diff_inside_abs = tf.abs(bbox_diff_inside)

        # calculate loss
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(bbox_diff_inside_abs, 1./sigma_2)))
        bbox_loss_inside = tf.pow(bbox_diff_inside, 2)*(sigma_2/2.)*smoothL1_sign + \
            (bbox_diff_inside_abs - (0.5/sigma_2))*(1. - smoothL1_sign)
        bbox_loss_outside = bbox_outside_weights*bbox_loss_inside
        bbox_loss = tf.reduce_mean(tf.reduce_sum(bbox_loss_outside, axis=dim))
        return bbox_loss

    # ····································································································
    # !                                     Network Summary                                              ！
    # ····································································································

    def add_summary_image(self, img, boxes):
        """ image summary """
        # add back mean
        img += np.array([[[102.9801, 115.9465, 122.7717]]])     # pixel mean
        # bgr to rgb (opencv uses bgr)
        channels = tf.unstack(img, axis=-1)
        img = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        # size
        width = tf.to_float(tf.shape(img)[2])
        height = tf.to_float(tf.shape(img)[1])
        # from [x1, y1, x2, y2, cl] to normalize [y1, x1, y1, x1]
        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[1]/height, cols[0]/width, cols[3]/height, cols[2]/width], axis=1)
        boxes = tf.expand_dims(boxes, dim=0)

        img = tf.image.draw_bounding_boxes(img, boxes)
        return tf.summary.image("ground_truth", img)

    # ····································································································
    # !                                       Variables                                                  ！
    # ····································································································

    def get_variables_to_restore(self, variables, var_keep_dic):
        """ get variables from pretrained ckpt and restore """
        variables_to_restore = []

        for var in variables:
            # exclude the conv weights that are full connected weights in vgg16
            if var.name == 'vgg_16/fc6/weights:0' or var.name == 'vgg_16/fc7/weights:0':
                self.variable_to_fix[var.name] = var
                continue
            # exclude the first conv layer to swap RGB to BGR
            if var.name == 'vgg_16/conv1/conv1_1/weights:0':
                self.variable_to_fix[var.name] = var
                continue
            if var.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s'%var.name)
                variables_to_restore.append(var)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        """ fix variables of vgg16"""
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16'):
            with tf.device("/cpu:0"):
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({
                    "vgg_16/fc6/weights": fc6_conv,\
                    "vgg_16/fc7/weights": fc7_conv,\
                    "vgg_16/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self.variable_to_fix['vgg_16/fc6/weights:0'],
                    tf.reshape(fc6_conv, self.variable_to_fix['vgg_16/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self.variable_to_fix['vgg_16/fc7/weights:0'],\
                    tf.reshape(fc7_conv, self.variable_to_fix['vgg_16/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self.variable_to_fix['vgg_16/conv1/conv1_1/weights:0'],\
                    tf.reverse(conv1_rgb, [2])))

    # ····································································································
    # !                                      Run Something                                               ！
    # ····································································································
    def extract_feature_map(self, sess, image):
        """ extract feature map """
        feed_dict = {self.img: image}
        feature_map = sess.run(self.layers["conv"], feed_dict=feed_dict)
        return feature_map

    def test_image(self, sess, image, imgInfo):
        """ input image, output rois and classes """
        feed_dict = {self.img: image, self.imgInfo: imgInfo}
        cls_score, cls_prob, bbox_pred, rois = sess.run(
            [self.predictions["cls_score"], self.predictions["cls_prob"],\
            self.predictions["bbox_pred"], self.predictions["rois"]], feed_dict=feed_dict)
        return cls_score, cls_score, bbox_pred, rois

    def get_summary(self, sess, blobs):
        """ get summary """
        feed_dict = {self.img: blobs['data'], self.imgInfo: blobs['im_info'], self.boxes: blobs['gt_boxes']}
        summary = sess.run(self.summary_op_val, feed_dict=feed_fict)
        return summary

    def train_step(self, sess, blobs, train_op):
        """ train operation """
        feed_dict = {self.img: blobs['data'], self.imgInfo: blobs['im_info'], self.boxes: blobs['gt_boxes']}
        rpn_cls_loss, rpn_bbox_loss, cls_loss, bbox_loss, loss, _ = sess.run([
            self.losses['rpn_cross_entropy'],
            self.losses['rpn_bbox_loss'],
            self.losses['cross_entropy'],
            self.losses['bbox_loss'],
            self.losses['total_loss'],
            train_op],
            feed_dict=feed_dict)
        return rpn_cls_loss, rpn_bbox_loss, cls_loss, bbox_loss, loss

    def train_step_with_summary(self, sess, blobs, train_step):
        """ train operation with summary """
        feed_dict = {self.img: blobs['data'], self.imgInfo: blobs['im_info'], self.boxes: blobs['gt_boxes']}
        rpn_cls_loss, rpn_bbox_loss, cls_loss, bbox_loss, loss, summary, _ = sess.run([\
            self.losses["rpn_cross_entropy"],
            self.losses["rpn_bbox_loss"],
            self.losses["cross_entropy"],
            self.losses["bbox_loss"],
            self.losses["total_loss"],
            self.summary_op,
            train_op],
            feed_dict=feed_dict)
        return rpn_cls_loss, rpn_bbox_loss, cls_loss, bbox_loss, loss, summary

    def train_step_no_return(self, sess, blobs, train_op):
        """ train step with no return """
        feed_fict = {self.img: blobs['data'], self.imgInfo: blobs['im_info'], self.boxes: blobs['gt_boxes']}
        sess.run([train_op], feed_fict=feed_fict)