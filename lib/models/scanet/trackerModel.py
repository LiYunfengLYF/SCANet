from torch import nn

from lib.utils.load import load_pretrain
from lib.utils.registry import MODEL_REGISTRY, TRACKER_REGISTRY


@TRACKER_REGISTRY.register()
class SCANet_network(nn.Module):
    """ This is the base class for TBSITrack developed on OSTrack (Ye et al. ECCV 2022) """

    def __init__(self, cfg, env_num=0, training=False, ):
        super().__init__()

        self.backbone = MODEL_REGISTRY.get(cfg.MODEL.BACKBONE.TYPE)(**cfg.MODEL.BACKBONE.PARAMS)

        if training and cfg.MODEL.BACKBONE.USE_PRETRAINED:
            load_pretrain(self.backbone, env_num=env_num, training=training, cfg=cfg, mode=cfg.MODEL.BACKBONE.LOAD_MODE)

        if hasattr(self.backbone, 'finetune_track'):
            self.backbone.finetune_track(cfg=cfg, patch_start_index=1)

        # rgb head
        self.head_type = cfg.MODEL.HEAD.TYPE
        self.sonar_head_type = cfg.MODEL.RGBS_HEAD.TYPE
        self.box_head = MODEL_REGISTRY.get(cfg.MODEL.HEAD.TYPE)(**cfg.MODEL.HEAD.PARAMS)
        self.sonar_head = MODEL_REGISTRY.get(cfg.MODEL.RGBS_HEAD.TYPE)(**cfg.MODEL.RGBS_HEAD.PARAMS)

    def forward(self, template: list, search: list, ):

        x, _ = self.backbone(template, search)

        out_rgb, out_sonar = self.forward_head(x, gt_score_map=None)
        return out_rgb, out_sonar

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)

        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]
        enc_opt2 = cat_feature[:, -num_search_token:, :]

        enc_opt1 = (enc_opt1.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = enc_opt1.size()
        enc_opt1 = enc_opt1.view(-1, C, 16, 16)

        enc_opt2 = (enc_opt2.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = enc_opt2.size()
        enc_opt2 = enc_opt2.view(-1, C, 16, 16)

        if self.head_type == "center_head" and self.sonar_head_type == "center_head":

            out = self.box_head(enc_opt1, gt_score_map)
            outputs_coord = out['pred_boxes']
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': out['score_map'],
                   'size_map': out['size_map'],
                   'offset_map': out['offset_map']}

            out_sonar = self.sonar_head(enc_opt2, gt_score_map)
            outputs_coord_sonar = out_sonar['pred_boxes']
            outputs_coord_sonar = outputs_coord_sonar.view(bs, Nq, 4)
            out_sonar = {'pred_boxes': outputs_coord_sonar,
                         'score_map': out_sonar['score_map'],
                         'size_map': out_sonar['size_map'],
                         'offset_map': out_sonar['offset_map']}
            return out, out_sonar
        elif self.head_type == "center_head" and self.sonar_head_type == "corner_head":
            out = self.box_head(enc_opt1, gt_score_map)
            outputs_coord = out['pred_boxes']
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': out['score_map'],
                   'size_map': out['size_map'],
                   'offset_map': out['offset_map']}

            out_sonar = self.sonar_head(enc_opt2, gt_score_map)
            outputs_coord_sonar = out_sonar['pred_boxes']
            outputs_coord_sonar = outputs_coord_sonar.view(bs, Nq, 4)
            out_sonar = {'pred_boxes': outputs_coord_sonar, }
            return out, out_sonar
        else:
            raise NotImplementedError
