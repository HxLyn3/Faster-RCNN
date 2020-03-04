# python lib
import os
import cv2
import pickle
import numpy as np
import tensorflow as tf

# my lib
from network.lib.rpn.nms import nms
from data.pascal_voc import pascal_voc
from network.Faster_RCNN import Network
from network.lib.rpn.roi import bbox_plus_delta, bbox_amend

def test():
    # datasets
    print("Reading images...")
    imdb = pascal_voc('test', '2007', os.path.join(os.getcwd(), 'data', 'VOCdevkit2007'))
    num_imgs = len(imdb.image_index)
    print("Read.")

    # network
    net = Network(num_classes=imdb.num_classes, mode="test")
    sess = tf.Session()
    net.createArchitecture(sess)
    # restore net
    model_dir = "./model"
    ckpt = tf.train.latest_checkpoint(model_dir)
    saver = tf.train.Saver()
    print("Loading model...")
    saver.restore(sess, ckpt)
    print("Loaded.")

    # seed
    np.random.seed(3)

    # all_boxes[cls][image] = (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_imgs)] for _ in range(imdb.num_classes)]

    for i in range(num_imgs):
        img = cv2.imread(imdb.image_path_at(i))
        scores, boxes = img_detect(sess, net, img)

        for j in range(1, imdb.num_classes):
            idxs = np.where(scores[:, j] > 0.05)[0]
            cls_scores = scores[idxs, j]
            cls_boxes = boxes[idxs, 4*j:4*(j+1)]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(cls_dets, 0.3)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        max_per_image = 100
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, imdb.num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, imdb.num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[keep, :]

    det_file = "./evalution/detections.pkl"
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print("Eavluating detections...")
    imdb.evaluate_detections(all_boxes, "./evalution")

def img_detect(sess, net, img):
    """ image dection """
    blobs = {}
    blobs['data'], img_scales = get_image_blob(img)
    img_blob = blobs['data']
    blobs['im_info'] = np.array([[img_blob.shape[1], img_blob.shape[2], img_blob.shape[0]]], dtype=np.float32)

    # test img
    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5]/img_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

    # apply bbox regression deltas
    bbox_deltas = bbox_pred
    boxes_pred = bbox_plus_delta(boxes, bbox_deltas)
    boxes_pred = bbox_amend(boxes_pred, img.shape)

    return scores, boxes_pred

def get_image_blob(img):
    """ convert an image into a network input. """
    img = img.astype(np.float32, copy=True)
    img -= np.array([[[102.9801, 115.9465, 122.7717]]])

    img_shape = img.shape
    img_size_min = np.min(img_shape[0:2])
    img_size_max = np.max(img_shape[0:2])

    # scale
    img_scale = 600./float(img_size_min)
    if np.round(img_scale*img_size_max) > 1000:
        img_scale = 1000./float(img_size_max)
    img = cv2.resize(img, None, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)
    blobs = np.reshape(img, [1, img.shape[0], img.shape[1], 3])
    img_scales = np.array([img_scale])

    return blobs, img_scales

if __name__ == '__main__':
    test()