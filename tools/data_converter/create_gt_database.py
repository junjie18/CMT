# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from UVTR (https://github.com/dvlab-research/UVTR)
# Copyright (c) 2022 Li, Yanwei
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import pickle
from os import path as osp

import mmcv
import numpy as np
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_prefix,
                                info_path=None,
                                mask_anno_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                with_mask=False):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
    """
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path, return_gt_info=True)

    if dataset_class_name == 'CustomNuScenesDataset':
        dataset_cfg.update(
            use_valid_flag=True,
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f'{info_prefix}_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path,
                                     f'{info_prefix}_dbinfos_train.pkl')

    database_pts_path = osp.join(database_save_path, 'pts_dir')
    database_img_path = osp.join(database_save_path, 'img_dir')
    mmcv.mkdir_or_exist(database_save_path)
    mmcv.mkdir_or_exist(database_pts_path)
    mmcv.mkdir_or_exist(database_img_path)
    all_db_infos = dict()

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].tensor.numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
        names = annos['gt_names']
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        # load multi-view image
        input_img = {}
        input_info = {}
        for _cam in example['info']['cams']:
            cam_info = example['info']['cams'][_cam]
            _path = cam_info['data_path']
            _img = mmcv.imread(_path, 'unchanged')
            input_img[_cam] = _img

            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)

            input_info[_cam]={
                'lidar2img': lidar2img_rt,              
                'lidar2cam': lidar2cam_rt,
                'cam_intrinsic': viewpad}

        for i in range(num_obj):
            pts_filename = f'{image_idx}_{names[i]}_{i}.bin'
            img_filename = f'{image_idx}_{names[i]}_{i}.png'
            abs_filepath = osp.join(database_pts_path, pts_filename)
            abs_img_filepath = osp.join(database_img_path, img_filename)
            rel_filepath = osp.join(f'{info_prefix}_gt_database', 'pts_dir', pts_filename)
            rel_img_filepath = osp.join(f'{info_prefix}_gt_database', 'img_dir', img_filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            img_crop, crop_key, crop_depth = find_img_crop(annos['gt_bboxes_3d'][i].corners.numpy(), input_img, input_info,  points[point_indices[:, i]])
            if img_crop is not None:
                mmcv.imwrite(img_crop, abs_img_filepath)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'image_path': rel_img_filepath if img_crop is not None else '',
                    'image_crop_key': crop_key if img_crop is not None else '',
                    'image_crop_depth': crop_depth,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


def find_img_crop(gt_boxes_3d, input_img, input_info,  points):
    coord_3d = np.concatenate([gt_boxes_3d, np.ones_like(gt_boxes_3d[..., :1])], -1)
    coord_3d = coord_3d.squeeze(0)
    max_crop, crop_key = None, None
    crop_area, crop_depth = 0, 0

    for _key in input_img:
        lidar2img = np.array(input_info[_key]['lidar2img'])
        coord_img = coord_3d @ lidar2img.T
        coord_img[:,:2] /= coord_img[:,2,None]
        image_shape = input_img[_key].shape
        if (coord_img[2] <= 0).any():
            continue
        
        avg_depth = coord_img[:,2].mean()
        minxy = np.min(coord_img[:,:2], axis=-2)
        maxxy = np.max(coord_img[:,:2], axis=-2)
        bbox = np.concatenate([minxy, maxxy], axis=-1)
        bbox[0::2] = np.clip(bbox[0::2], a_min=0, a_max=image_shape[1]-1)
        bbox[1::2] = np.clip(bbox[1::2], a_min=0, a_max=image_shape[0]-1)
        bbox = bbox.astype(int)
        if ((bbox[2:]-bbox[:2]) <= 10).any():
            continue

        img_crop = input_img[_key][bbox[1]:bbox[3],bbox[0]:bbox[2]]
        if img_crop.shape[0] * img_crop.shape[1] > crop_area:
            max_crop = img_crop
            crop_key = _key
            crop_depth = avg_depth
    
    return max_crop, crop_key, crop_depth