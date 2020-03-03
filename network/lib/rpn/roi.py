import numpy as np
from .nms import nms

# ····································································································
# !                                   Generate Proposals                                             ！
# ····································································································

def proposal(rpn_cls_prob, rpn_bbox_pred, imgInfo, mode, feat_stride, anchors, num):
    """ 
    Brief:
        Generate ROIs using nms.
    
    Args:
        rpn_cls_prob:
            (probs = probabilities)
            - probs of background @anchor0
            - probs of background @anchor1
            ......
            - probs of background @anchor_N-1
            - probs of foreground @anchor0
            - probs of foreground @anchor1
            ......
            - probs of foreground @anchor_N-1
            PS: a point correponds to prob of an anchor (with shift)
        rpn_bbox_pred:
            - Δx @anchor0 at each pos
            - Δy @anchor0 at each pos
            - Δw @anchor0 at each pos
            - Δh @anchor0 at each pos
            - ...@anchor1
            - ......
        imgInfo: information of image
        mode: "train" or "test"
        feat_stride: total stride of conv when generate feature map
        num: num of anchors at a pos
    """

    if str(mode, encoding='utf-8') == "train":
        pre_topN = 12000    # number of top scoring boxes to keep before applying nms
        post_topN = 2000    # number of top scoring boxes to keep after applying nms
        threshold = 0.7     # nms threshold used on proposals
    elif str(mode, encoding='utf-8') == "test":
        pre_topN = 6000
        post_topN = 300
        threshold = 0.7

    # rpn_cls_prob ( , , , 2*num) --cut--> scores( , , , num)
    # scores:
    # probs of foreground @anchor0
    #     - probs of foreground @anchor0
    #     - probs of foreground @anchor1
    #     ......
    #     - probs of foreground @anchor_N-1
    #     PS: a point correponds to prob of an anchor (with shift)
    scores = rpn_cls_prob[:, :, :, num:]
    # --> (anchor0, anchor1, ..., anchor_N-1, anchor0, ...) traverse pos
    scores = scores.reshape((-1, 1))
    # --> ((Δx,Δy,Δw,Δh)@anchor0, @anchor1, ..., @anchor_N-1, @anchor0, ...) traverse pos
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    # update anchors with deltas
    proposals = bbox_plus_delta(anchors, rpn_bbox_pred)
    # amend that left >= 0, top >= 0, right < imgW, bottom <= imgH for each proposal
    proposals = bbox_amend(proposals, imgInfo[:2])  # imgInfo[0] = imgH, imgInfo[1] = imgW

    # pick top N proposals, pre
    order = scores.ravel().argsort()[::-1]      # index of score from big --> small
    if pre_topN > 0:
        order = order[:pre_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), threshold)
    # pick top N proposal, post
    if post_topN > 0:
        keep = keep[:post_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # only support single image as input
    batch_idxs = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    rois = np.hstack((batch_idxs, proposals.astype(np.float32, copy=False)))

    return rois, scores

def proposal_top(rpn_cls_prob, rpn_bbox_pred, imgInfo, feat_stride, anchors, num, topN):
    """ 
    Brief:
        Generate ROIs just select top scores ROIs.
    
    Args:
        rpn_cls_prob:
            (probs = probabilities)
            - probs of background @anchor0
            - probs of background @anchor1
            ......
            - probs of background @anchor_N-1
            - probs of foreground @anchor0
            - probs of foreground @anchor1
            ......
            - probs of foreground @anchor_N-1
            PS: a point correponds to prob of an anchor (with shift)
        rpn_bbox_pred:
            - Δx @anchor0 at each pos
            - Δy @anchor0 at each pos
            - Δw @anchor0 at each pos
            - Δh @anchor0 at each pos
            - ...@anchor1
            - ......
        imgInfo: information of image
        feat_stride: total stride of conv when generate feature map
        num: num of anchors at a pos
    """

    selectN = topN      # number of top scoring boxes to keep

    # rpn_cls_prob ( , , , 2*num) --cut--> scores( , , , num)
    # scores:
    # probs of foreground @anchor0
    #     - probs of foreground @anchor0
    #     - probs of foreground @anchor1
    #     ......
    #     - probs of foreground @anchor_N-1
    #     PS: a point correponds to prob of an anchor (with shift)
    scores = rpn_cls_prob[:, :, :, num:]
    # --> (anchor0, anchor1, ..., anchor_N-1, anchor0, ...) traverse pos
    scores = scores.reshape((-1, 1))
    # --> ((Δx,Δy,Δw,Δh)@anchor0, @anchor1, ..., @anchor_N-1, @anchor0, ...) traverse pos
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))

    total_num = scores.shape[0]     # number of all anchors
    if total_num < selectN:
        # random selection
        top_idx = np.random.choice(total_num, size=selectN, replace=True)
    else:
        # select top scores roi
        top_idx = scores.argsort(0)[::-1]
        top_idx = top_idx[:selectN]
        top_idx = top_idx.reshape(selectN, )
    
    # selection
    anchors = anchors[top_idx, :]
    rpn_bbox_pred = rpn_bbox_pred[top_idx, :]
    scores = scores[top_idx]

    # update anchors with deltas
    proposals = bbox_plus_delta(anchors, rpn_bbox_pred)
    # amend that left >= 0, top >= 0, right < imgW, bottom <= imgH for each proposal
    proposals = bbox_amend(proposals, imgInfo[:2])  # imgInfo[0] = imgH, imgInfo[1] = imgW

    # only support single image as input
    batch_idxs = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    rois = np.hstack((batch_idxs, proposals.astype(np.float32, copy=False)))

    return rois, scores

# ····································································································
# !                                    Bbox Operations                                               ！
# ····································································································

def bbox_plus_delta(boxes, deltas):
    """ each box in boxes transform with corresponding delta in deltas """
    # when boxes empty
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    
    # (left, top, right, bottom) --> (centerx, centery, w, h)
    boxes = boxes.astype(deltas.dtype, copy=False)
    ws = boxes[:, 2] - boxes[:, 0] + 1.0
    hs = boxes[:, 3] - boxes[:, 1] + 1.0
    center_xs = boxes[:, 0] + 0.5*ws
    center_ys = boxes[:, 1] + 0.5*hs

    # deltas
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    # prediction
    centerx_pred = dx*ws[:, np.newaxis] + center_xs[:, np.newaxis]
    centery_pred = dy*hs[:, np.newaxis] + center_ys[:, np.newaxis]
    w_pred = np.exp(dw)*ws[:, np.newaxis]
    h_pred = np.exp(dh)*hs[:, np.newaxis]

    boxes_pred = np.zeros(deltas.shape, dtype=deltas.dtype)
    boxes_pred[:, 0::4] = centerx_pred - 0.5*w_pred     # left
    boxes_pred[:, 1::4] = centery_pred - 0.5*h_pred     # top
    boxes_pred[:, 2::4] = centerx_pred + 0.5*w_pred     # right
    boxes_pred[:, 3::4] = centery_pred + 0.5*h_pred     # bottom

    return boxes_pred

def bbox_amend(boxes, img_shape):
    """ amend that left >= 0, top >= 0, right < imgW, bottom <= imgH for each box """
    imgH = img_shape[0]
    imgW = img_shape[1]
    # left >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], imgW-1), 0)
    # top >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], imgH-1), 0)
    # right < imgW
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], imgW-1), 0)
    # bottom < imgH
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], imgH-1), 0)

    return boxes

def bbox_transform(rois, boxes):
    """ calculate the deltas between rois and boxes """
    # roi
    roi_ws = rois[:, 2] - rois[:, 0] + 1.0
    roi_hs = rois[:, 3] - rois[:, 1] + 1.0
    roi_centerx = rois[:, 0] + 0.5*roi_ws
    roi_centery = rois[:, 1] + 0.5*roi_hs

    # box
    box_ws = boxes[:, 2] - boxes[:, 0] + 1.0
    box_hs = boxes[:, 3] - boxes[:, 1] + 1.0
    box_centerx = boxes[:, 0] + 0.5*box_ws
    box_centery = boxes[:, 1] + 0.5*box_hs

    # deltas
    dx = (box_centerx-roi_centerx)/roi_ws
    dy = (box_centery-roi_centery)/roi_hs
    dw = np.log(box_ws/roi_ws)
    dh = np.log(box_hs/roi_hs)
    deltas = np.vstack((dx, dy, dw, dh)).transpose()

    return deltas

def bbox_overlaps(anchors, boxes):
    """ calculate IOU of each anchors and each boxes """
    # init
    N = anchors.shape[0]
    K = boxes.shape[0]
    overlaps = np.zeros((N, K), np.float)

    # for each boxes
    for k in range(K):
        box_w = boxes[k, 2] - boxes[k, 0] + 1
        box_h = boxes[k, 3] - boxes[k, 1] + 1
        box_area = box_w*box_h

        # for each anchors
        for n in range(N):
            # width of intersection
            inter_w = min(anchors[n, 2], boxes[k, 2]) - max(anchors[n, 0], boxes[k, 0]) + 1
            if inter_w > 0:
                # heigth of intersection
                inter_h = min(anchors[n, 3], boxes[k, 3]) - max(anchors[n, 1], boxes[k, 1]) + 1
                if inter_h > 0:
                    anchor_w = anchors[n, 2] - anchors[n, 0] + 1
                    anchor_h = anchors[n, 3] - anchors[n, 1] + 1
                    anchor_area = anchor_w*anchor_h
                    union_area = box_area + anchor_area - inter_w*inter_h
                    overlaps[n, k] = inter_w*inter_h/union_area
    
    return overlaps

# ····································································································
# !                       Prepare Labels for Softmax and bbox at end                                 ！
# ····································································································

def proposal_target(rois, roi_scores, boxes, num_classes, net_batch_size):
    """
    Brief:
        Labels for proposals, including labels for softmax at end
    and (Δx, Δy, Δw, Δh) for bbox.

    Args:
        rois: rois produced by rpn
        rois_scores: scores of foreground for each roi
        boxes: ground-truth boxes
        num_classes: number of classes
    """

    # proposal ROIs (0, left, top, right, bottom) coming from RPN
    all_rois = rois
    all_scores = roi_scores

    rois_per_image = net_batch_size                     # rois in an image
    fg_rois_per_image = np.round(0.25*rois_per_image)   # foreground rois in an image

    # sample rois with classification labels and bounding box regression targets
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = \
        rois_sample(all_rois, all_scores, boxes, fg_rois_per_image, rois_per_image, num_classes)
    
    # reshape
    rois = rois.reshape(-1, 5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, 4*num_classes)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, 4*num_classes)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

def rois_sample(all_rois, all_scores, boxes, fg_rois_per_image, rois_per_image, num_classes):
    """ """
    # overlaps (num_rois x num_boxes), IOU of a roi and a box
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(boxes[:, :4], dtype=np.float))
    iou_argmax_to_rois = overlaps.argmax(axis=1)    # index of boxes(max IOU with the roi) for each rois
    iou_max_to_rois = overlaps.max(axis=1)
    # class labels for each rois
    labels = boxes[iou_argmax_to_rois, 4]

    # select foreground rois
    foreground_idx = np.where(iou_max_to_rois >= 0.5)[0]
    # select background rois
    background_idx = np.where((iou_max_to_rois < 0.5) & (iou_max_to_rois >= 0.1))[0]
    # background_idx = np.where(iou_max_to_rois < 0.5)[0]

    # fix number of rois
    if foreground_idx.size > 0 and background_idx.size > 0:
        # fix foreground rois
        fg_rois_per_image = min(fg_rois_per_image, foreground_idx.size)
        foreground_idx = np.random.choice(foreground_idx, size=int(fg_rois_per_image), replace=False)
        # fix background rois
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = background_idx.size < bg_rois_per_image
        background_idx = np.random.choice(background_idx, size=int(bg_rois_per_image), replace=to_replace)
    elif foreground_idx.size > 0:
        # fix foreground rois
        to_replace = foreground_idx.size < rois_per_image
        foreground_idx = np.random.choice(foreground_idx, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif background_idx.size > 0:
        # fix background rois
        to_replace = background_idx.size < rois_per_image
        background_idx = np.random.choice(background_idx, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    else:
        raise Exception()

    # indices of selected rois
    keep_idx = np.append(foreground_idx, background_idx)
    # labels of selected roi, and labels of background rois are set 0
    labels = labels[keep_idx]
    labels[int(fg_rois_per_image):] = 0
    # selected rois
    rois = all_rois[keep_idx]
    roi_scores = all_scores[keep_idx]

    # bbox deltas for each roi to its corresponding box
    targets = bbox_transform(rois[:, 1:5], boxes[iou_argmax_to_rois[keep_idx], :4])

    # normalize
    bbox_normalize_means = (0.0, 0.0, 0.0, 0.0)
    bbox_normalize_stds = (0.1, 0.1, 0.1, 0.1)
    targets = ((targets - np.array(bbox_normalize_means))/np.array(bbox_normalize_stds))

    bbox_target_data = np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

    # N: number of proposals, K: number of classes
    # bbox_targets: (NxK)
    # if a proposal is class k
    # its target is 0 0 0 0 .... dx, dy, dw, dh (4k--4k+4 pos) .... 0 0 0 0
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4*num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    idxs = np.where(clss > 0)[0]
    for idx in idxs:
        _cls = clss[idx]
        start = int(4*_cls)
        end = start + 4
        bbox_targets[idx, start:end] = bbox_target_data[idx, 1:]
        bbox_inside_weights[idx, start:end] = (1.0, 1.0, 1.0, 1.0)
    
    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
