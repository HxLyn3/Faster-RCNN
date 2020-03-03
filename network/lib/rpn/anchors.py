import math
import numpy as np
from .roi import bbox_transform, bbox_overlaps

# ····································································································
# !                                    Generate Anchors                                              ！
# ····································································································

def generate_init_anchors(
    base_size=16, ratios=(0.5, 1, 2), scales=np.array([8, 16, 32])):
    """ generate initial anchors """
    # basic anchor
    baseW, baseH = base_size, base_size
    xCenter, yCenter = 0.5*(baseW-1), 0.5*(baseH-1) 

    # keep area, change w:h
    ws = [round(baseW/math.sqrt(ratio)) for ratio in ratios]    
    hs = [round(ws[i]*ratios[i]) for i in range(len(ratios))]
    # scale
    ws = [w*scale for w in ws for scale in scales]
    hs = [h*scale for h in hs for scale in scales]
    # a series of anchors
    anchors = []
    for i in range(len(ws)):
        left = xCenter - 0.5*(ws[i]-1)
        top = yCenter - 0.5*(hs[i]-1)
        right = xCenter + 0.5*(ws[i]-1)
        bottom = yCenter + 0.5*(hs[i]-1)
        anchors.append([left, top, right, bottom])

    return np.array(anchors)

def generate_anchors(
    height, width, feat_stride, 
    anchor_ratios=(0.5, 1, 2), anchor_scales=np.array([8, 16, 32])):
    """ 
    Brief:
        Generate anchors (on image).

    Args:
        height: height of feature map
        width: width of feature map
        feat_stride: feature stride when conv
    """
    # initial anchors
    anchors = generate_init_anchors(ratios=anchor_ratios, scales=anchor_scales)
    n_anchors = anchors.shape[0]

    # shifts
    x_shift = np.arange(0, width) * feat_stride
    y_shift = np.arange(0, height) * feat_stride
    x_shift, y_shift = np.meshgrid(x_shift, y_shift)
    shifts = np.vstack((x_shift.ravel(), y_shift.ravel(), x_shift.ravel(), y_shift.ravel())).transpose()
    n_shifts = shifts.shape[0]

    # anchors shift    PS: shape(1, 3) + shape(4, 1) = shape(4, 3)
    anchors = anchors.reshape((1, n_anchors, 4)) + shifts.reshape((1, n_shifts, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape(n_anchors*n_shifts, 4).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    """
        anchors:
            - init anchor0 -> shift0
            - init anchor1 -> shift0
            - init anchor2 -> shift0
            - ......
            - init anchor8 -> shift0
            - init anchor0 -> shift1
            - init anchor1 -> shift1
            - ......
            - init anchor8 -> shift1
            - ......
            - init anchor0 -> shift_N-1
            - init anchor1 -> shift_N-1
            - ......
            - ......
            - init anchor8 -> shift_N-1
            - ......
    """

    return anchors, length

# ····································································································
# !                     Prepare Labels for Softmax and Bbox of RPN                                   ！
# ····································································································

def anchor_target(rpn_cls_score, boxes, imgInfo, feat_stride, all_anchors, num):
    """
    Brief:
        Labels for rpn, including labels for softmax in rpn
    and (Δx, Δy, Δw, Δh) for bbox.

    Args:
        rpn_cls_score:
            - scores of background @anchor0
            - scores of background @anchor1
            ......
            - scores of background @anchor_N-1
            - scores of foreground @anchor0
            - scores of foreground @anchor1
            ......
            - scores of foreground @anchor_N-1
            PS: a point correponds to score of an anchor (with shift)
        boxes: all boxes need to be generated at an img (left, top, right, bottom, score)
        imgInfo: information of image
        feat_stride: total stride of conv when generate feature map
        num: num of anchors at a pos
    """
    # shape of map (batch_size, height, width, channels)
    height, width = rpn_cls_score.shape[1:3]

    # allow boxes to sit over the edge by a small amount
    allowed_border = 0
    # keep anchors inside the image
    inside_idx = np.where(
        (all_anchors[:, 0] >= -allowed_border) & 
        (all_anchors[:, 1] >= -allowed_border) &
        (all_anchors[:, 2] < imgInfo[1] + allowed_border) &
        (all_anchors[:, 3] < imgInfo[0] + allowed_border))[0]
    anchors = all_anchors[inside_idx, :]
    
    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inside_idx), ), dtype=np.float32)
    labels.fill(-1)

    # IOU between anchors and boxes, overlaps -- (anchor_num, box_num)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(boxes, dtype=np.float))
    
    iou_argmax_to_anchors = overlaps.argmax(axis=1)     # index of boxes(max IOU with the anchor) for each anchors
    iou_max_to_anchors = overlaps[np.arange(len(inside_idx)), iou_argmax_to_anchors]

    iou_argmax_to_boxes = overlaps.argmax(axis=0)       # index of anchors(max IOU with the box) for each boxes
    iou_max_to_boxes = overlaps[iou_argmax_to_boxes, np.arange(overlaps.shape[1])]
    iou_argmax_to_boxes = np.where(overlaps == iou_max_to_boxes)[0]     # sort

    labels[iou_max_to_anchors < 0.3] = 0    # IOU with each boxes < 0.3 must be negative
    labels[iou_argmax_to_boxes] = 1         # anchors whose IOU with one of boxes is max must be positive
    labels[iou_max_to_anchors >= 0.7] = 1   # IOU with one of boxes >= 0.7 must be positive

    positive_limit = 128
    positive_idx = np.where(labels == 1)[0]
    # if too many positive labels
    positive_num = len(positive_idx)
    if positive_num > positive_limit:
        disable_idx = np.random.choice(positive_idx, size=(positive_num-positive_limit), replace=False)
        labels[disable_idx] = -1

    negative_limit = 256 - np.sum(labels == 1)
    negative_idx = np.where(labels == 0)[0]
    # if too many negative labels
    negative_num = len(negative_idx)
    if negative_num > negative_limit:
        disable_idx = np.random.choice(negative_idx, size=(negative_num-negative_limit), replace=False)
        labels[disable_idx] = -1

    max_iou_box_to_anchors = boxes[iou_argmax_to_anchors, :]    # box whose IOU with the anchor is max, for each anchors
    # delta between anchor and box whose IOU with the anchor is max
    bbox_targets = bbox_transform(anchors, max_iou_box_to_anchors[:, :4]).astype(np.float32, copy=False)

    # only the positive ones have regression targets
    bbox_inside_weights = np.zeros((len(inside_idx), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0, 1.0, 1.0))

    bbox_outside_weights = np.zeros((len(inside_idx), 4), dtype=np.float32)
    num_samples = np.sum(labels >= 0)   # positive samples and negative samples
    bbox_outside_weights[labels >= 0, :] = np.ones((1, 4))*1.0/num_samples

    total_num = all_anchors.shape[0]    # num of all anchors
    labels = unmap(labels, total_num, inside_idx, fill=-1)
    bbox_targets = unmap(bbox_targets, total_num, inside_idx, fill=0)
    bbox_inside_weights = unmap(bbox_inside_weights, total_num, inside_idx, fill=0)     # (1.0, 1.0, 1.0, 1.0) at positive
    bbox_outside_weights = unmap(bbox_outside_weights, total_num, inside_idx, fill=0)   # (1/n, 1/n, 1/n, 1/n) at positive and negative, n = num of (po + ne)

    # labels
    labels = labels.reshape((1, height, width, num)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, num*height, width))
    rpn_labels = labels
    # rpn_labels:
    #     - labels @anchor0
    #     - labels @anchor1
    #     - ...
    #     - labels @anchor_N-1

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, num*4))
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, num*4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, num*4))
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

def unmap(data, cnt, idx, fill=0):
    """ unmap a subset of data back to the original set of data """
    if len(data.shape) == 1:
        res = np.empty((cnt, ), dtype=np.float32)
        res.fill(fill)
        res[idx] = data
    else:
        res = np.empty((cnt, ) + data.shape[1:], dtype=np.float32)
        res.fill(fill)
        res[idx, :] = data
    return res
