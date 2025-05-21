"""Evaluation script for the NUS Benchmark."""

import collections
import math
import os
import time
from config import config_parser
import cv2
from ibrnet.data_loaders.llff_data_utils import batch_parse_llff_poses
from ibrnet.data_loaders.llff_data_utils import load_llff_data
from ibrnet.projection import Projector
from ibrnet.sample_ray import RaySamplerSingleImage
import imageio
import numpy as np
import skimage.metrics
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import glob
import softsplat
from utils import colorize_np

from kornia.utils import create_meshgrid

import scipy
from scipy.spatial.transform import Rotation as R
import albumentations


class NUS(Dataset):
  """This class loads data from Nvidia benchmarks, including camera scene and image information from source views."""

  def __init__(self, args, mode, scenes, **kwargs):
    self.args = args
    self.rgb_dir = args.images_path
    self.base_dir = args.in_folder
    self.out_folder = args.out_folder
    scenes.sort()
    self.scenes = scenes
    print('loading {} for rendering'.format(self.scenes))

    self.render_rgb_files = []
    self.render_intrinsics = []
    self.render_poses = []
    self.render_train_set_ids = []
    self.render_depth_range = []

    self.train_intrinsics = []
    self.train_poses = []
    self.train_rgb_files = []
    self.train_depth = []
    self.h = []
    self.w = []
    self.render_scene = []

    self.neighbor_list = args.neighbor_list

    self.width = args.width
    self.height = args.height

    intrinsics, poses, depths, bds, rgb_files = self.load_nus()

    near_depth = np.min(bds)
    far_depth = np.max(bds)

    i_train = np.array(np.arange(int(len(rgb_files))))
    i_render = i_train

    smoothed_poses, _ = smooth_trajectory(poses)

    self.train_intrinsics.extend(intrinsics)
    self.train_poses.extend([c2w_mat for c2w_mat in poses[i_train]])
    self.train_rgb_files.extend(rgb_files)
    self.train_depth.extend([depth for depth in depths[i_train]])
    num_render = len(i_render)
    self.render_intrinsics.extend(intrinsics)
    self.render_poses.extend([c2w_mat for c2w_mat in smoothed_poses[i_render]])
    self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
    self.render_train_set_ids.extend(i_render.tolist()) 
    self.h.extend([int(self.height)]*num_render)
    self.w.extend([int(self.width)]*num_render)


  def load_nus(self):
    
    base_dir = self.base_dir
    rgb_files = glob.glob(os.path.join(self.rgb_dir, '*.png'))
    rgb_files.sort()
    rgb_files = rgb_files

    if self.args.preprocessing_model == "Deep3D":
# ###################################### for Deep3D
        poses = np.load(os.path.join(base_dir,'poses.npy'))
        depths = []
        for idx in range(len(poses)):
            depth = np.load(os.path.join(base_dir,'depths/{:05}.npy'.format(idx)))
            depths.append(depth)
        depths = np.concatenate(depths, axis=0)
        origin_height, origin_width = depths.shape[-2:]
        
        #1687.271606
        intrinsic = np.array([[1687.271606*(origin_width/self.width), 0, origin_width*0.5, 0], [0, 1687.271606*(origin_height/self.height), origin_height*0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# ###################################### for Deep3D
    elif  self.args.preprocessing_model == "MonST3R":
# ###################################### for monst3r
        monst3r_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)),'MonST3R/{}/{}'.format(os.path.basename(base_dir),"output"))
        def qvec2rotmat(qvec):
            return np.array([
                [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
                [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
                [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
        poses = []
        with open(os.path.join(monst3r_dir, "pred_traj.txt"), "r") as file:
            while True:
                line = file.readline()
                if line:
                    line = list(map(float, line.split(" ")))
                    R = np.transpose(qvec2rotmat(line[4:]))
                    T = np.array(line[1:4])
                    Rt = np.zeros((4, 4))
                    Rt[:3, :3] = R.transpose()
                    Rt[:3, 3] = T
                    Rt[3, 3] = 1.0
                    poses.append(Rt)
                else:
                    break
        poses = np.stack(poses, axis=0)

        intrinsics = []
        with open(os.path.join(monst3r_dir, "pred_intrinsics.txt"), "r") as file:
            while True:
                line = file.readline()
                if line:
                    line = list(map(float, line.split(" ")))
                    intrinsics.append(line)
                else:
                    break
        intrinsic = np.eye(4)
        intrinsic[:3,:3] = np.array(intrinsics[0]).reshape(3,3)

        intrinsic

        disps =[]
        for i in range(poses.shape[0]):
            disp_path = os.path.join(monst3r_dir, '{:05d}.npy'.format(i))
            disps.append(np.load(disp_path))
        depths = np.stack(disps, axis=0)

        origin_height, origin_width =  2*intrinsic[1][2], 2*intrinsic[0][2]
# ###################################### for monst3r

    length = poses.shape[0]

    intrinsic_res = intrinsic[:]
    intrinsic_res[0] *= (self.width / origin_width)
    intrinsic_res[1] *= (self.height / origin_height)

    bds = np.stack([np.min(depths.reshape(length,-1),axis=1),np.max(depths.reshape(length,-1),axis=1)],axis=1)
    

    return [intrinsic_res]*length, poses, depths, bds, rgb_files

  def __len__(self):
    return len(self.render_poses) # number of viewpoints

  def __getitem__(self, total_idx):
    offset = 0
    idx = total_idx%len(self.render_train_set_ids)

    render_idx = self.render_train_set_ids[idx]

    render_pose = self.render_poses[total_idx]
    intrinsics = self.render_intrinsics[idx]
    depth_range = self.render_depth_range[idx]
    scene_path = self.base_dir

    train_rgb_files = self.train_rgb_files
    train_poses = self.train_poses
    train_intrinsics = self.train_intrinsics
    train_depth = self.train_depth

    h, w = self.h[idx], self.w[idx]
    camera = np.concatenate(
        ([h, w], intrinsics.flatten(), render_pose.flatten())
    ).astype(np.float32)
    nearest_pose_ids = np.sort([np.clip(render_idx+offset, 0,len(self.render_poses)-1) for offset in self.neighbor_list])
    depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

    src_rgbs = []
    src_cameras = []

    train_pose_list = []
    ref_depth = []

    for id in nearest_pose_ids:
        src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
        train_pose = train_poses[id]
        train_intrinsics_ = train_intrinsics[id]
        depth = train_depth[id]

        src_rgbs.append(src_rgb)
        img_size = src_rgb.shape[:2]
        src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                      train_pose.flatten())).astype(np.float32)
        src_cameras.append(src_camera)

        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, (h, w), mode='area')
        ref_depth.append(np.array(depth.squeeze()))
        train_pose_list.append(train_pose)

    flows = []
    if not self.args.no_color_correction:
        flow_h, flow_w = np.load(scene_path+'/flows/1/{:05d}.npy'.format(0))[1,:].shape[:-1]
        for id in nearest_pose_ids:
            try:
                if id==render_idx-1:
                    flows.append(np.load(scene_path+'/flows/{}/{:05d}.npy'.format(str(render_idx-id), id + offset))[1,:])
                elif id==render_idx+1:
                    flows.append(np.load(scene_path+'/flows/{}/{:05d}.npy'.format(str(id-render_idx), render_idx + offset))[0,:])
                else:
                    flows.append(np.zeros((flow_h,flow_w,2)))
            except:
                flows.append(np.zeros((flow_h,flow_w,2)))
        
        flows = torch.from_numpy(np.stack(flows, axis=0)).float()

        flows[..., 0] = flows[..., 0] / flow_w * w
        flows[..., 1] = flows[..., 1] / flow_h * h
        flows = flows.permute(0, 3, 1, 2)
        flows = F.interpolate(flows, (h, w), mode='area')
        flows = flows.permute(0, 2, 3, 1)

    src_rgbs = np.stack(src_rgbs, axis=0)
    src_rgbs = torch.from_numpy(src_rgbs[..., :3]).float().permute(0,3,1,2)
    src_rgbs = F.interpolate(src_rgbs, (h, w), mode='bilinear')

    src_cameras = np.stack(src_cameras, axis=0)
    depth = np.stack(depth, axis=0)
    depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

    train_pose_list = np.stack(train_pose_list, axis=0)
    ref_depth = np.stack(ref_depth, axis=0)

    warped_depth, std, depth_masks, hybrid_rgb = self.depth_reprojection(src_rgbs,torch.from_numpy(ref_depth).unsqueeze(0), torch.from_numpy(render_pose).float().unsqueeze(0), torch.from_numpy(train_pose_list).float().unsqueeze(0),torch.from_numpy(intrinsics).float()[...,:3,:3].unsqueeze(0))


    depth_ex = torch.concat([warped_depth.unsqueeze(1), self.args.sample_range_gain*std.unsqueeze(1), depth_masks], dim=1)
    depth_ex = np.array(depth_ex.cpu().squeeze(0).permute(1,2,0))
    return {
        'camera': torch.from_numpy(camera),
        'depth_ex': torch.from_numpy(depth_ex),
        'src_rgbs': src_rgbs.permute(0,2,3,1),
        'src_cameras': torch.from_numpy(src_cameras).float(),
        'hybrid_rgb': hybrid_rgb,
        'depth_range': depth_range.float(),
        'src_flows': torch.tensor(flows),
        'num_view': len(self.args.neighbor_list)
    }

  def depth_reprojection(self, rgb, ref_depth, ext, ref_ext, intrinsics):
    #ref_depth S_V, H, W
    #ext 4, 4
    #ref_ext S_V, 4, 4
    #intrinsics 3, 3
    
    B, S_V, H, W = ref_depth.shape
    src_ext = ref_ext
    src_ixt = intrinsics.unsqueeze(1)
    src_projs = src_ixt @ torch.inverse(src_ext)[:, :, :3, :]

    tar_ext = ext.unsqueeze(1)
    tar_ixt = intrinsics.unsqueeze(1)
    # tar_ixt[:, :2] *= tar_scale
    tar_projs = tar_ixt @ torch.inverse(tar_ext)[:,:, :3, :]

    src_ones = torch.zeros((B, S_V, 1, 4)).to(src_projs.device)
    src_ones[:, :, :, 3] = 1
    tar_ones = torch.zeros((B, 1, 1, 4)).to(tar_projs.device)
    tar_ones[:, :, :, 3] = 1
    src_projs_h = torch.cat((src_projs, src_ones), dim=-2)
    tar_projs_h = torch.cat((tar_projs, tar_ones), dim=-2)
    src_projs_inv = torch.inverse(src_projs_h)
    tar_projs_inv = torch.inverse(tar_projs_h)

    src_projs_inv = src_projs_inv.view( B, S_V, 4, 4)
    tar_projs_inv = tar_projs_inv.view( B, 1, 4, 4)
    tar_projs = tar_projs.view(B, 1, 3, 4)

    proj_mats = tar_projs @ src_projs_inv

    flow, projection_mask, computed_depth = self.flow_from_reprojection(ref_depth, proj_mats)
    projection_mask = projection_mask.cuda()

    # computed_depth = ref_depth
  
    flow = flow.contiguous().view(B*S_V,2,H,W).cuda()
    computed_depth = computed_depth.view(B*S_V,1,H,W).cuda()
    mask = torch.ones_like(computed_depth)
    rgb = rgb.view(B*S_V,3,H,W).cuda()
    input_depth = torch.concat([computed_depth, mask, rgb], dim=1)

    out_depth = softsplat.FunctionSoftsplat(tenInput=input_depth, 
                                tenFlow=flow, 
                                tenMetric=None, 
                                strType='average')

    warped_depths, masks, warped_rgb = torch.split(out_depth, split_size_or_sections=[1,1,3], dim=1)

    warped_depths = warped_depths.view( B, S_V, H, W)

    warped_rgb = warped_rgb.view( S_V, 3, H, W).permute(1,0,2,3)

    masks = masks.view( B, S_V, H, W)
    masks = (masks>0.999).int()
    masks = tensor_erode(masks)
    # weight_map = torch.sum(masks, dim=1)-masks[:,S_V//2,:,:]

    mask1 = masks[:,S_V//2,:,:]*0

    mask2 = (torch.sum(masks,dim=1)>2).int()*(1-mask1)
    mask3 = (1-mask1)*(1-mask2)

    warped_depths = (warped_depths*masks)#[:,S_V//2-3:S_V//2+4,...]
    timestamp = torch.tensor(self.args.neighbor_list).cuda()
    alpha = 0.5
    weight_time1 = torch.exp(-alpha*timestamp.abs()).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
    # weight_time1 = torch.ones_like(weight_time1)
    weights = masks*weight_time1
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)

    weight_time1 = torch.ones_like(weight_time1)
    weights1 = masks*weight_time1
    weights1 = weights1 / (torch.sum(weights1, dim=1, keepdim=True) + 1e-8)

    hybrid_depths = self.weighted_average(warped_depths, weights)

    completed_depths = torch.from_numpy(cv2.inpaint(np.array(hybrid_depths[0].cpu()),np.array(mask3[0].cpu(), dtype=np.uint8),9,cv2.INPAINT_NS)).unsqueeze(0).cuda()

    std = self.weighted_std(warped_depths, completed_depths, weights1+1e-8)
    completed_std = torch.from_numpy(cv2.inpaint(np.array(std[0].cpu()),np.array(mask3[0].cpu(), dtype=np.uint8),9,cv2.INPAINT_NS)).unsqueeze(0).cuda()
    std = completed_std

    hybrid_rgb = self.weighted_average(warped_rgb, weights)
    hybrid_depths = warped_depths[:,S_V//2,...]
    
    masks = torch.concat([torch.stack([mask1,mask2,mask3],dim=1), masks], dim=1)
    return completed_depths, std, masks, hybrid_rgb
  
  
  
  def weighted_average(self, tensors, weights):
    z = torch.mul(tensors, weights)
    weighted_average = torch.sum(z, dim=1) / (torch.sum(weights, dim=1)+ 1e-8)
    return weighted_average
  
  def weighted_std(self, tensors, mean, weights):
    z = torch.mul((tensors-mean)**2, weights)
    weighted_std = (torch.sum(z, dim=1) / torch.sum(weights, dim=1))**0.5
    return weighted_std

  def flow_from_reprojection(self, depth, proj_mat):
    #depth
    #proj_mat

    B, S_V, H, W = depth.shape

    R = proj_mat[:, :, :, :3] # (B, S_V, 3, 3)
    T = proj_mat[:, :, :, 3:] # (B, S_V, 3, 1)
    # create grid from the ref frame
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False, device=proj_mat.device) # (1, H, W, 2)
    ref_grid = ref_grid.permute(0, 3, 1, 2) # (1, 2, H, W)
    ref_grid = ref_grid.reshape(1, 2, H*W) # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1) # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:,:1])), 1) # (B, S_V, 3, H*W)
    # ref_grid_d = ref_grid.repeat(1, 1, D) # (B, 3, D*H*W)
    reprojection_grid_d = R @ ref_grid + T/depth.view(B, S_V, 1, H*W)
    reprojection_grid_d = reprojection_grid_d/reprojection_grid_d[:,:,2:,:]
    # reprojection_grid_d_test = R @ ref_grid + + T/(depth.view(B, S_V, 1, H*W)*0.01)
    # reprojection_grid_d_test = reprojection_grid_d_test/reprojection_grid_d_test[:,:,2:,:]
    flow = reprojection_grid_d - ref_grid.unsqueeze(1)
    output  = flow.view(B, S_V, 3, H, W)
    flow = output[:,:,:2,:,:]

    mask = None

    flow, mask, computed_depth, _, project_3d = project_pixel(depth, pose=proj_mat)
    flow = flow.permute(0,1,4,2,3)
    return flow, mask, computed_depth



  def warp(self, x, flo):
        B, C, H, W = x.size()
        # mesh grid
       
        flo =  flo.permute(0, 2, 3, 1)
        grid = create_meshgrid(H, W, normalized_coordinates=False, device=flo.device)
        grid_flow = grid + flo
        vgrid = grid_flow.clone()

        # scale grid to [-1,1]
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0

        # vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        # tools.check_tensor(x, 'x')
        # tools.check_tensor(vgrid, 'vgrid')
        output = F.grid_sample(x, vgrid, padding_mode='zeros')
        return output, grid_flow

  def forward_occ_check(self, flow_fw, flow_bw, scale=1):
      def length_sq(x):
          # torch.sum(x ** 2, dim=1, keepdim=True)
          temp = torch.sum(x ** 2, dim=1, keepdim=True)
          temp = torch.pow(temp, 0.5)
          # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
          return temp


      sum_func = length_sq
      mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
      flow_bw_warped, _ = self.warp(flow_bw, flow_fw)  # torch_warp(img,flow)
      # flow_fw_warped = warp(flow_fw, flow_bw)
      flow_diff_fw = flow_fw + flow_bw_warped
      # flow_diff_bw = flow_bw + flow_fw_warped
      occ_alpha_1=1.0
      occ_alpha_2=0.05
      occ_thresh = occ_alpha_1 * mag_sq + occ_alpha_2 / scale
      occ_thresh = 5
      occ = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
      # occ_bw = sum_func(flow_diff_bw) < occ_thresh
      return occ

def calculate_psnr(img1, img2, mask):
  """Compute PSNR between two images.

  Args:
    img1: image 1
    img2: image 2
    mask: mask indicating which region is valid.

  Returns:
    PSNR: PSNR error
  """

  # img1 and img2 have range [0, 1]
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  mask = mask.astype(np.float64)

  num_valid = np.sum(mask) + 1e-8

  mse = np.sum((img1 - img2) ** 2 * mask) / num_valid

  if mse == 0:
    return 0  # float('inf')

  return 10 * math.log10(1.0 / mse)


def calculate_ssim(img1, img2, mask):
  """Compute SSIM between two images.

  Args:
    img1: image 1
    img2: image 2
    mask: mask indicating which region is valid.

  Returns:
    PSNR: PSNR error
  """
  if img1.shape != img2.shape:
    raise ValueError('Input images must have the same dimensions.')

  _, ssim_map = skimage.metrics.structural_similarity(
      img1, img2, multichannel=True, full=True
  )
  num_valid = np.sum(mask) + 1e-8

  return np.sum(ssim_map * mask) / num_valid


def im2tensor(image, cent=1.0, factor=1.0 / 2.0):
  """Convert image to Pytorch tensor.

  Args:
    image: input image
    cent: shift
    factor: scale

  Returns:
    Pytorch tensor
  """
  return torch.Tensor(
      (image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
  )


def smooth_trajectory(poses_mat):
    poses_vec = mat2vec(poses_mat, rotation_mode='quat')
    
    window = 29
    half = window // 2

    L = poses_vec.shape[0]

    stability = 12
    
    smooth_vec = poses_vec.copy()
    for t in range(0, L):
        window_begin, window_end = max(0, t - half), min(L, t + half + 1)
        min_half = min(t - window_begin, window_end - 1 - t)
        window_begin, window_end = t - min_half, t + min_half + 1 
        weights = scipy.signal.windows.gaussian(2 * min_half + 1, stability)
        weights /= weights.sum()

        vec_window = poses_vec[window_begin:window_end]
        smooth_vec[t] = weighted_pose(vec_window, weights, 'quat')
    
    smooth_mat = vec2mat(smooth_vec, rotation_mode= 'quat')
    compensate_mat = inverse_posemat(smooth_mat) @ poses_mat
    return smooth_mat, compensate_mat

def mat2vec(poses_mat, rotation_mode):
    
    r = R.from_matrix(poses_mat[:, :3, :3])
    if rotation_mode == 'euler':
        r_vec = r.as_euler('xyz')
    else:
        r_vec = r.as_quat()
    t_vec = poses_mat[:, :3, 3]
    vec = np.concatenate([t_vec, r_vec], axis=1)
    return vec

def vec2mat(poses_vec, rotation_mode):
    if rotation_mode == 'euler':
        r = R.from_euler('xyz', poses_vec[:, 3:])
    elif rotation_mode == 'quat':
        r = R.from_quat(poses_vec[:, 3:])
    r_mat = r.as_matrix()
    mat = np.concatenate([r_mat, np.expand_dims(poses_vec[:, :3], 2)], axis=2)
    mat = np.concatenate([mat, np.zeros((poses_vec.shape[0], 1, 4))], axis=1)
    mat[:, 3, 3] = 1.
    return mat


def inverse_posemat(posemat):
    R = posemat[:, :3, :3]
    t = posemat[:, :3, 3:]
    R_T = np.transpose(R, (0, 2, 1))
    t_inv = -R_T @ t
    pose_inv = np.concatenate([R_T, t_inv], axis=2)
    bot = np.zeros([posemat.shape[0], 1, 4])
    bot[:, :, -1] = 1.
    pose_inv = np.concatenate([pose_inv, bot], axis=1)
    return pose_inv

def weighted_pose(window, weights, rotation_mode):
    t = np.average(window[:, :3], axis=0, weights=weights)
    if rotation_mode == 'quat':
        r = R.from_quat(window[:, 3:]).mean(weights=weights).as_quat()
    else:
        r = R.from_euler('xyz', window[:, 3:]).mean(weights=weights).as_euler('xyz')
    pose = np.concatenate([t, r], axis=0)
    return pose

def inverse_flow(forward_flows):
    # inverse optical flow: given a forward flow, return the backward flow

    bs, h, w, _ = forward_flows.size()


    sw, sh = 1, 1
    idxs = {i*sw+j: [] for i in range(sh) for j in range(sw)}
    for i in range(h):
        for j in range(w):
            key = ((i % sh) * sw) + j % sw
            idxs[key].append(i * w + j)
    idx_set = [torch.Tensor(v).long() for v in idxs.values()]

    pixel_map = create_meshgrid(h, w, normalized_coordinates=False, device=forward_flows.device)

    # forward_flows = pixel_map + forward_flows
    
    x = forward_flows[..., 0].reshape(bs, -1)
    y = forward_flows[..., 1].reshape(bs, -1)
    l = torch.floor(x); r = l + 1
    t = torch.floor(y); b = t + 1
    # mask = (l >= 0) * (t >= 0) * (r < w) * (b < h)
    # l *= mask; r *= mask; t *= mask; b *= mask
    # x *= mask; y *= mask
    w_rb = torch.abs(x - l + 1e-5) * torch.abs(y - t + 1e-5)
    w_rt = torch.abs(x - l + 1e-5) * torch.abs(b - y + 1e-5)
    w_lb = torch.abs(r - x + 1e-5) * torch.abs(y - t + 1e-5)
    w_lt = torch.abs(r - x + 1e-5) * torch.abs(b - y + 1e-5)
    l = l.long(); r = r.long(); t = t.long(); b = b.long()

    weight_maps = torch.zeros(bs, h, w).to(forward_flows.device).double()
    # grid_x = self.pixel_map[..., 0].view(-1).long()
    # grid_y = self.pixel_map[..., 1].view(-1).long()

    for i in range(bs):
        for j in idx_set:
            weight_maps[i, t[i, j], l[i, j]] += w_lt[i, j]
            weight_maps[i, t[i, j], r[i, j]] += w_rt[i, j]
            weight_maps[i, b[i, j], l[i, j]] += w_lb[i, j]
            weight_maps[i, b[i, j], r[i, j]] += w_rb[i, j]


    forward_shifts = (-forward_flows + pixel_map.repeat(bs, 1, 1, 1)).double()
    backward_flows = torch.zeros(forward_flows.size()).to(forward_shifts.device)
    for i in range(bs):
        for c in range(2):
            for j in idx_set:
                backward_flows[i, t[i, j], l[i, j], c] += \
                    forward_shifts[i, :, :, c].reshape(-1)[j] * w_lt[i, j]
                backward_flows[i, t[i, j], r[i, j], c] += \
                    forward_shifts[i, :, :, c].reshape(-1)[j] * w_rt[i, j]
                backward_flows[i, b[i, j], l[i, j], c] += \
                    forward_shifts[i, :, :, c].reshape(-1)[j] * w_lb[i, j]
                backward_flows[i, b[i, j], r[i, j], c] += \
                    forward_shifts[i, :, :, c].reshape(-1)[j] * w_rb[i, j]
    for c in range(2):
        backward_flows[..., c] /= weight_maps

    backward_flows[torch.isinf(backward_flows)] = 0
    backward_flows[torch.isnan(backward_flows)] = 0
    backward_flows += pixel_map.repeat(bs, 1, 1, 1)

    backward_flows[weight_maps == 0] = -2

    return backward_flows

def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combinaison of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w)

def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)

def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    # check_sizes(img, 'img', 'B3HW')
    # check_sizes(depth, 'depth', 'B1HW')
    # check_sizes(ref_depth, 'ref_depth', 'B1HW')
    # check_sizes(pose, 'pose', 'B6')
    # check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    # pose_mat = pose_vec2mat(pose)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    return projected_img, valid_mask, projected_depth, computed_depth

def tensor_erode(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


def project_pixel(depth, pose):
    B, S_V, H, W = depth.size()
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False, device=depth.device) # (1, H, W, 2)
    # ref_grid = ref_grid.permute(0, 3, 1, 2) # (1, 2, H, W)
    ref_grid = ref_grid.unsqueeze(1).expand(B, S_V, -1, -1, -1) # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[...,:1])), -1) # (B, S_V, 3, H*W)
    cam_coords = ref_grid * depth.unsqueeze(-1)

    # proj_cam_to_src_pixel = self.intrinsic.expand(bs, 3, 3) @ pose[:, :3]
    proj_cam_to_tar_pixel = pose
    R = proj_cam_to_tar_pixel[:, :, :, :3]
    t = proj_cam_to_tar_pixel[:, :, :, -1:]
    
    src_pixel_coords, computed_depth, project_3d, norm_pixel_coords = cam2pixel(cam_coords, R, t)

    valid_points = norm_pixel_coords.abs().max(dim=-1)[0] <= 1
    mask = valid_points.float()
    flow = src_pixel_coords-ref_grid[..., :2]
    return flow, mask, computed_depth, cam_coords, project_3d

def cam2pixel(cam_coords, R, t):
    B, S_V, H, W , _ = cam_coords.size()
    cam_coords_flat= cam_coords.reshape(B, S_V, H*W, 3).permute(0,1,3,2)
    pcoords = R @ cam_coords_flat + t

    pcoords = pcoords.permute(0,1,3,2)

    X = pcoords[..., 0]
    Y = pcoords[..., 1]
    Z = pcoords[..., 2]

    X_norm = 2*(X / Z) / (W - 1) - 1
    Y_norm = 2*(Y / Z) / (H - 1) - 1


    norm_pixel_coords = torch.stack([X_norm, Y_norm], dim=-1) #[B, H*W, 2]

    pixel_coords = pcoords[...,:2]/pcoords[...,2:]
    return pixel_coords.reshape(B, S_V, H, W, 2), Z.reshape(B, S_V, H, W), \
            pcoords.reshape(B, S_V, H, W, 3), norm_pixel_coords.reshape(B, S_V, H, W, 2)