# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F



rng = np.random.RandomState(234)

########################################################################################################################
# ray batch sampling
########################################################################################################################


def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def dilate_img(img, kernel_size=20):
    import cv2
    assert img.dtype == np.uint8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img / 255, kernel, iterations=1) * 255
    return dilation


class RaySamplerSingleImage(object):
    def __init__(self, data, device, resize_factor=1, render_stride=1):
        super().__init__()
        self.render_stride = render_stride
        self.rgb = data['rgb'] if 'rgb' in data.keys() else None
        self.camera = data['camera']
        self.rgb_path = data['rgb_path'] if 'rgb_path' in data.keys() else None
        self.depth_ex = data['depth_ex']
        # self.warped_depths = data['warped_depths']
        # self.depth_masks = data['depth_masks']
        self.depth_range = data['depth_range']
        self.device = device
        W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        self.batch_size = len(self.camera)

        self.H = int(H[0])
        self.W = int(W[0])

        # S_V = self.depth_ex.size()[1]

        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrinsics[:, :2, :3] *= resize_factor
            if self.rgb is not None:
                self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)

        # self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat, self.depth)
        self.rays = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat, self.depth_ex)

        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3)

        if 'src_rgbs' in data.keys():
            self.src_rgbs = data['src_rgbs']
        else:
            self.src_rgbs = None
        if 'src_cameras' in data.keys():
            self.src_cameras = data['src_cameras']
        else:
            self.src_cameras = None
        if 'depth' in data.keys():
            self.depth = data['depth']
        else:
            self.depth = None
        if 'warped_depths' in data.keys():
            self.warped_depths = data['warped_depths']
        else:
            self.warped_depths= None
        if 'depth_masks' in data.keys():
            self.depth_masks = data['depth_masks']
        else:
            self.depth_masks= None
        if 'src_flows' in data.keys():
            self.src_flows = data['src_flows']
        else:
            self.src_flows = None
        if 'occ' in data.keys():
            self.occ = data['occ']
        else:
            self.occ = None
        if 'num_view' in data.keys():
            self.num_view = data['num_view']
        else:
            self.num_view = None

    def get_rays_single_image(self, H, W, intrinsics, c2w, depth_ex):
        '''
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        '''
        # B,depth_ex.size()

        u, v = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        grid = np.stack((u, v), axis=-1)
        grid[...,0] = 2* (grid[...,0]/ (W - 1) - 0.5)
        grid[...,1] = 2* (grid[...,1]/ (H - 1) - 0.5)
        grid = torch.from_numpy(grid)

        l = depth_ex.shape[-1]
        depth_ex = depth_ex.reshape(-1,l)
        rays_depth, rays_std, mask1, mask2, mask3, masks = torch.split(depth_ex, split_size_or_sections=[1,1,1,1,1,l-5],dim=1)
        
        mask1 = (mask1>0.99).int()
        mask2 = (mask2>0.99).int()*(1-mask1)
        mask3 = 1-mask1-mask2

        flag = mask1+2*mask2+3*mask3

        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)  # B x HW x 3
        

        rays = torch.concat([rays_o, rays_d, rays_depth, rays_std, flag, masks], dim=-1)
        return rays

    def get_all(self):
        ret = {'rays': self.rays.cuda(),
               'depth_range': self.depth_range.cuda(),
               'camera': self.camera.cuda(),
               'rgb': self.rgb.cuda() if self.rgb is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
                'src_flows': self.src_flows.cuda() if self.src_flows is not None else None,
                'occ': self.occ.cuda() if self.occ is not None else None,
                'num_view': self.num_view
        }
        return ret

    def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
        if sample_mode == 'center':
            border_H = int(self.H * (1 - center_ratio) / 2.)
            border_W = int(self.W * (1 - center_ratio) / 2.)

            # pixel coordinates
            u, v = np.meshgrid(np.arange(border_H, self.H - border_H),
                               np.arange(border_W, self.W - border_W))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == 'uniform':
            # Random from one image
            select_inds = rng.choice(self.H*self.W, size=(N_rand,), replace=False)
        else:
            raise Exception("unknown sample mode!")

        return select_inds

    def random_sample(self, N_rand, sample_mode, center_ratio=0.8):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''

        select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio)

        # rays_o = self.rays_o[select_inds]
        # rays_d = self.rays_d[select_inds]
        rays = self.rays[select_inds]

        if self.rgb is not None:
            rgb = self.rgb[select_inds]
        else:
            rgb = None

        ret = {'rays': rays.cuda(),
            #    'ray_d': rays_d.cuda(),
               'camera': self.camera.cuda(),
               'depth_range': self.depth_range.cuda(),
               'rgb': rgb.cuda() if rgb is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
               'selected_inds': select_inds,
               'src_flows': self.src_flows.cuda() if self.src_flows is not None else None,
                'occ': self.occ.cuda() if self.src_flows is not None else None,
                'num_view': self.num_view
        }
        return ret
