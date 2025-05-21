import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict
import time
import imageio.v2 as imageio
import options
from sequence_io import *
from smooth import smooth_trajectory, get_smooth_depth_kernel
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter as scipy_gaussian
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def get_cropping_area(warp_maps, h, w):
    border_t = warp_maps[:, 0, :, 1][warp_maps[:, 0, :, 1] >= 0]
    border_b = warp_maps[:, -1, :, 1][warp_maps[:, -1, :, 1] >= 0]
    border_l = warp_maps[:, :, 0, 0][warp_maps[:, :, 0, 0] >= 0]
    border_r = warp_maps[:, :, -1, 0][warp_maps[:, :, -1, 0] >= 0]
    
    t = int(torch.ceil(torch.clamp(torch.max(border_t), 0, h))) if border_t.shape[0] != 0 else 0
    b = int(torch.floor(torch.clamp(torch.min(border_b), 0, h)))if border_b.shape[0] != 0 else 0
    l = int(torch.ceil(torch.clamp(torch.max(border_l), 0, w))) if border_l.shape[0] != 0 else 0
    r = int(torch.floor(torch.clamp(torch.min(border_r), 0, w)))if border_r.shape[0] != 0 else 0
    return t, b, l, r

@torch.no_grad()
def compute_warp_maps(seq_io, warper, compensate_poses, post_process=False):
    # compute all warp maps
    batch_begin = 0
    warp_maps = []
    ds = []
    W, H = seq_io.origin_width, seq_io.origin_height
    w, h = seq_io.width, seq_io.height
    crop_t, crop_b, crop_l, crop_r = 0, H, 0, W
    
    # post processing
    if post_process:
        smooth_depth = get_smooth_depth_kernel().to(device)

    while batch_begin < len(seq_io):
    
        batch_end = min(len(seq_io), batch_begin + seq_io.batch_size)
        segment = list(range(batch_begin, batch_end))
        depths = seq_io.load_depths(segment).to(device)

        if post_process:
            # load error maps
            error_maps = seq_io.load_errors(segment)
            thresh = 0.5
            error_maps[error_maps > thresh] = 1
            error_maps[error_maps < thresh] = 0
           
            # remove the noise in error map
            for i in range(error_maps.shape[0]):
                eroded_map = np.expand_dims(binary_erosion(error_maps[i].squeeze(0), iterations=1), 0)
                error_maps[i] = binary_dilation(eroded_map, iterations=8)
            
            # spatial-variant smoother according to the error map
            softmasks = scipy_gaussian(error_maps, sigma=[0, 0, 7, 7])
            softmasks = torch.from_numpy(softmasks).float().to(device)

            smoothed_depths = smooth_depth(depths) #smooth_depths(depths)
            depths = depths * (1 - softmasks) + smoothed_depths * softmasks

        # compute warping fields
        batch_warps, _, _, _, _ = warper.project_pixel(depths, compensate_poses[batch_begin:batch_end])
        batch_warps = (batch_warps + 1) / 2

        batch_warps[..., 0] *= (W - 1) 
        batch_warps[..., 1] *= (H - 1)
        t, b, l, r = get_cropping_area(batch_warps, H, W)
        crop_t = max(crop_t, t); crop_b = min(crop_b, b); crop_l = max(crop_l, l); crop_r = min(crop_r, r)
        
        batch_warps[..., 0] *= (w - 1) / (W - 1)
        batch_warps[..., 1] *= (h - 1) / (H - 1)

        inverse_warps = warper.inverse_flow(batch_warps)
        inverse_warps[..., 0] = inverse_warps[..., 0] * 2 / (w - 1) - 1
        inverse_warps[..., 1] = inverse_warps[..., 1] * 2 / (h - 1) - 1
        
        warp_maps.append(inverse_warps.detach().cpu())

        batch_begin = batch_end
    
    warp_maps = torch.cat(warp_maps, 0)
    return warp_maps, (crop_t, crop_b, crop_l, crop_r)

@torch.no_grad()
def run(opt, args):
    if args.keep_size=='True': 
        args.height, args.width = imageio.imread(os.path.join(args.images_path,'00000.png')).shape[:2]

    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    extra_out_dir = '{}/{}'.format(args.out_folder, args.expname)
    print('saving results to {}...'.format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    opt.output_dir = extra_out_dir
    args.out_folder = extra_out_dir

    projector = Projector(device='cuda:0', args=args)

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step), 'videos')

    print('saving results to {}'.format(out_scene_dir))

    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = dataset_dict[args.eval_dataset](args, 'validation',
                                                  scenes=args.eval_scenes)

    test_loader = DataLoader(test_dataset, batch_size=1)
    out_frames = []
    for i, data in enumerate(tqdm(test_loader)):
        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)) # c=32
            
            ret = render_single_image(ray_sampler=ray_sampler,
                                      ray_batch=ray_batch,
                                      model=model,
                                      projector=projector,
                                      chunk_size=args.chunk_size,
                                      det=True,
                                      N_samples=args.N_samples,
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      featmaps=featmaps)
            torch.cuda.empty_cache()

        coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
        coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)

        coarse_pred_rgb = torch.from_numpy(cv2.inpaint(coarse_pred_rgb,1-np.array(ret['outputs_coarse']['pixel_mask'].cpu(), dtype=np.uint8),9,cv2.INPAINT_NS))

        imageio.imwrite(os.path.join(out_scene_dir, '%05d.png'%i), coarse_pred_rgb)
        out_frame = coarse_pred_rgb
        out_frames.append(out_frame)

    imageio.mimwrite(os.path.join(extra_out_dir, '{}.avi'.format(scene_name)), out_frames, fps=30, quality=8)
    print('=> Done!')



if __name__ == '__main__':
    opt = options.Options().parse()
    global device
    device = torch.device(opt.cuda)
    opt.preprocess = False

    parser = config_parser()
    args = parser.parse_args()

    args.images_path = '../output/Deep3D/{}/images'.format(args.expname)
    args.in_folder = '../output/Deep3D/{}'.format(args.expname)
    args.out_folder = '../output/RStab'
    args.eval_scenes = ['RStab_test']
    run(opt, args)




