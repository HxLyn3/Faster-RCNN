import numpy as np

def nms(rois, threshold):
    """
    Brief:
        Pick ROIs using non-maximum suppression
    
    Args:
        rois:
            - [left, top, right, bottom, score] @ an anchor
            - ......
        threshold: 
            threshold of IOU, pick a proposal whose score is
        maximum at a region, then throw away other proposals
        whose IOU with max score proposal is > threshold.
    
    Return:
        keep: index of proposals need to be kept after nms.
    """

    if rois.shape[0] == 0:
        return []

    # left, top, right, bottom and area of rois
    lefts = rois[:, 0]
    tops = rois[:, 1]
    rights = rois[:, 2]
    bottoms = rois[:, 3]
    areas = (rights-lefts+1)*(bottoms-tops+1)

    scores = rois[:, 4]
    order = scores.argsort()[::-1]      # index of rois from big -> small

    keep = []
    while order.size > 0:
        # pick a proposal whose score is maximum
        i = order[0]
        keep.append(i)
        order = order[1:]

        # intersection of proposal i and the rest proposals
        lefts_Inter = np.maximum(lefts[i], lefts[order])
        tops_Inter = np.maximum(tops[i], tops[order])
        rights_Inter = np.minimum(rights[i], rights[order])
        bottoms_Inter = np.minimum(bottoms[i], bottoms[order])
        ws_Inter = np.maximum(0.0, rights_Inter - lefts_Inter + 1)
        hs_Inter = np.maximum(0.0, bottoms_Inter - tops_Inter + 1)
        inters = ws_Inter*hs_Inter
        unions = areas[i] + areas[order] - inters
        IOUs = inters/unions

        # whose IOU with proposal i < threshold
        stay_idxs = np.where(IOUs <= threshold)[0]
        order = order[stay_idxs]

    return keep
