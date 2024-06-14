import os

import torch
from .data_utils import Preprocessor
from .tracker import Tracker
from lib.test.evaluation.environment import env_settings

from ..utils.hann import hann2d
from ...train.data.processing_utils import sample_target
from ...utils.box_ops import clip_box

from ...utils.load import load_yaml
from lib.models.scanet.trackerModel import TRACKER_REGISTRY
from .registry import PIPELINE_REGISTRY


@PIPELINE_REGISTRY.register()
class scanet(Tracker):
    def __init__(self, version):
        super().__init__(fr'scanet_{version}')
        self.name = rf'scanet_{version}'

        # RGBS Network
        cfg = load_yaml(os.path.join(env_settings(0).prj_dir, f'experiments/scanet/{version}.yaml'))
        network = TRACKER_REGISTRY.get(cfg.MODEL.NETWORK)(cfg, env_num=101, training=False)
        checkpoint_dir = os.path.join(env_settings(0).save_dir, "checkpoints/train/scanet/%s/%s_ep%04d.pth.tar" %
                                      (version, cfg.MODEL.NETWORK, cfg.TEST.EPOCH))
        network.load_state_dict(torch.load(checkpoint_dir, map_location='cpu')['net'], strict=True)

        # Network and tracker state
        self.cfg = cfg
        self.network = network.cuda().eval()
        self.preprocessor = Preprocessor()
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.rgb_state = None
        self.s_state = None

        # rgb-s template and search area
        self.z_rgb_dict1 = None
        self.z_sonar_dict1 = None

        # score
        self.rgb_score = 0.5
        self.sonar_score = 0.5

    def init(self, rgb_img, sonar_img, rgb_gt, sonar_gt):
        rgb_z_patch_arr, rgb_resize_factor, rgb_z_amask_arr = sample_target(rgb_img, rgb_gt,
                                                                            self.cfg.TEST.TEMPLATE_FACTOR,
                                                                            output_sz=self.cfg.TEST.TEMPLATE_SIZE)
        rgb_template = self.preprocessor.process(rgb_z_patch_arr, rgb_z_amask_arr)
        with torch.no_grad():
            self.z_rgb_dict1 = rgb_template
        self.rgb_state = rgb_gt

        sonar_z_patch_arr, sonar_resize_factor, sonar_z_amask_arr = sample_target(sonar_img, sonar_gt,
                                                                                  self.cfg.TEST.TEMPLATE_FACTOR,
                                                                                  output_sz=self.cfg.TEST.TEMPLATE_SIZE)
        sonar_template = self.preprocessor.process(sonar_z_patch_arr, sonar_z_amask_arr)
        with torch.no_grad():
            self.z_sonar_dict1 = sonar_template
        self.s_state = sonar_gt

    def track(self, rgb_img, s_img):
        # RGB pre processing
        Hv, Wv, _ = rgb_img.shape
        rgb_x_patch_arr, rgb_resize_factor, rgb_x_amask_arr = sample_target(rgb_img, self.rgb_state,
                                                                            self.cfg.TEST.SEARCH_FACTOR,
                                                                            output_sz=self.cfg.TEST.SEARCH_SIZE)  # (x1, y1, w, h)
        rgb_search = self.preprocessor.process(rgb_x_patch_arr, rgb_x_amask_arr)

        # sonar pre processing
        Hs, Ws, _ = s_img.shape
        son_x_patch_arr, son_resize_factor, son_x_amask_arr = sample_target(s_img, self.s_state,
                                                                            self.cfg.TEST.SEARCH_FACTOR,
                                                                            output_sz=self.cfg.TEST.SEARCH_SIZE)  # (x1, y1, w, h)
        son_search = self.preprocessor.process(son_x_patch_arr, son_x_amask_arr)
        with torch.no_grad():
            rgb_x_dict = rgb_search
            son_x_dict = son_search
            out_rgb, out_sonar = self.network(template=[self.z_rgb_dict1.tensors, self.z_sonar_dict1.tensors],
                                              search=[rgb_x_dict.tensors, son_x_dict.tensors])

        if self.network.head_type == 'center_head':
            self.rgb_state, rgb_score = self.out2box_wscore_center(out_rgb, rgb_resize_factor, Hv, Wv, is_rgb=True)
        else:
            raise

        if self.network.sonar_head_type == 'center_head':
            self.s_state, sonar_score = self.out2box_wscore_center(out_sonar, son_resize_factor, Hs, Ws, is_rgb=False)
        elif self.network.sonar_head_type == 'corner_head':
            self.s_state, sonar_score = self.out2box_wscore_corner(out_sonar, son_resize_factor, Hs, Ws, is_rgb=False)

        rgb_box = [0, 0, 0, 0] if rgb_score < self.rgb_score else self.rgb_state
        sonar_box = [0, 0, 0, 0] if sonar_score < self.sonar_score else self.s_state

        return rgb_box, sonar_box

    def out2box_wscore_center(self, out_dict, resize_factor, H, W, is_rgb=False):
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, max_score = self.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.cfg.TEST.SEARCH_SIZE / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        box = clip_box(self.map_box_back(pred_box, resize_factor, is_rgb), H, W, margin=10)
        return box, max_score

    def out2box_wscore_corner(self, out_dict, resize_factor, H, W, is_rgb=False):
        pred_boxes = out_dict['pred_boxes']
        pred_boxes = pred_boxes.view(-1, 4)

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.cfg.TEST.SEARCH_SIZE / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        box = clip_box(self.map_box_back(pred_box, resize_factor, is_rgb), H, W, margin=10)
        return box, 1.0

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def map_box_back(self, pred_box: list, resize_factor: float, is_rgb=False):
        if is_rgb:
            cx_prev, cy_prev = self.rgb_state[0] + 0.5 * self.rgb_state[2], self.rgb_state[1] + 0.5 * self.rgb_state[3]
        else:
            cx_prev, cy_prev = self.s_state[0] + 0.5 * self.s_state[2], self.s_state[1] + 0.5 * self.s_state[3]

        cx, cy, w, h = pred_box
        half_side = 0.5 * self.cfg.TEST.SEARCH_SIZE / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
