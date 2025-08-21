import os
import numpy as np
import pandas as pd
import json
import math
from .load_txt import read_rodnet_res
from .rod_eval_utils import compute_ols_dts_gts, evaluate_img, accumulate, summarize

olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)

def read_gt_csv(csv_path, n_frame, dataset):
    n_class = dataset.object_cfg.n_class
    classes = dataset.object_cfg.classes
    data = pd.read_csv(csv_path)
    n_row, n_col = data.shape
    dets = [None] * n_frame

    for row in range(n_row):
        filename = data['filename'][row]
        frame_id = int(filename.split('.')[0].split('_')[-1])
        region_count = data['region_count'][row]

        region_shape_attri = json.loads(data['region_shape_attributes'][row])
        region_attri = json.loads(data['region_attributes'][row])
        a = region_shape_attri['cx']
        r = region_shape_attri['cy']
        class_name = region_attri['class']
        class_id = classes.index(class_name)

        obj_dict = dict(
            frame_id=frame_id,
            range=r,
            angle=a,
            class_name=class_name,
            class_id=class_id
        )
        if dets[frame_id] is None:
            dets[frame_id] = [obj_dict]
        else:
            dets[frame_id].append(obj_dict)

    gts = {(i, j): [] for i in range(n_frame) for j in range(n_class)}
    id = 1
    for frameid, obj_info in enumerate(dets):
        # for each frame
        if obj_info is None:
            continue
        for obj_dict in obj_info:
            rng = obj_dict['range']
            agl = obj_dict['angle']
            class_id = obj_dict['class_id']
            if rng > 93 or rng < 2:
                continue
            if agl > math.radians(64) or agl < math.radians(-64):
                continue
            obj_dict_gt = obj_dict.copy()
            obj_dict_gt['id'] = id
            obj_dict_gt['score'] = 1.0
            gts[frameid, class_id].append(obj_dict_gt)
            id += 1

    return gts


def evaluate_rodnet_seq(res_path, gt_path, n_frame, dataset):
    gt_dets = read_gt_csv(gt_path, n_frame, dataset)
    sub_dets = read_rodnet_res(res_path, n_frame, dataset)

    olss_all = {(imgId, catId): compute_ols_dts_gts(gt_dets, sub_dets, imgId, catId, dataset) \
                for imgId in range(n_frame)
                for catId in range(1)}

    evalImgs = [evaluate_img(gt_dets, sub_dets, imgId, catId, olss_all, olsThrs, recThrs, dataset)
                for imgId in range(n_frame)
                for catId in range(1)]

    return evalImgs

