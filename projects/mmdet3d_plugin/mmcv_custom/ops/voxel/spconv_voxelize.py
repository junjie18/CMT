# Copyright (c) 2023 megvii-model. All Rights Reserved.

import numpy as np
from torch import nn
from spconv.pytorch.utils import PointToVoxel  # spconv-cu111  2.1.21
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class SPConvVoxelization(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels, num_point_features, device=torch.device("cuda")):
        super().__init__()
        assert len(voxel_size) == 3
        assert len(point_cloud_range) == 6
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_num_points = max_num_points
        self.num_point_features = num_point_features
        self.device = device
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.voxel_generator = PointToVoxel(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            max_num_points_per_voxel=max_num_points,
            max_num_voxels=self.max_voxels[0],
            num_point_features=num_point_features,
            device=device,
        )
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

    def train(self, mode: bool = True):
        if mode:
            self.voxel_generator = PointToVoxel(
                vsize_xyz=self.voxel_size.tolist(),
                coors_range_xyz=self.point_cloud_range.tolist(),
                max_num_points_per_voxel=self.max_num_points,
                max_num_voxels=self.max_voxels[0],
                num_point_features=self.num_point_features,
                device=self.device,
            )
        else:
            self.voxel_generator = PointToVoxel(
                vsize_xyz=self.voxel_size.tolist(),
                coors_range_xyz=self.point_cloud_range.tolist(),
                max_num_points_per_voxel=self.max_num_points,
                max_num_voxels=self.max_voxels[1],
                num_point_features=self.num_point_features,
                device=self.device,
            )

        return super().train(mode)

    def forward(self, points):
        voxel_output = self.voxel_generator(points)
        voxels, coordinates, num_points = voxel_output
        return torch.clone(voxels), torch.clone(coordinates), torch.clone(num_points)
    
    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', num_point_features=' + str(self.num_point_features)
        tmpstr += ')'
        return tmpstr
