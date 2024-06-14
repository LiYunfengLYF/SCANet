import os


class EnvironmentSettings:
    def __init__(self, env_num):
        self.workspace_dir = ''  # Base directory for saving network checkpoints.
        self.tensorboard_dir = os.path.join(self.workspace_dir, 'tensorboard')

        self.pretrained_networks = os.path.join(self.workspace_dir, 'pretrained_models')

        # SOT
        self.lasot_dir = r''
        self.got10k_dir = r''
        self.got10k_val_dir = r''
        self.trackingnet_dir = r''
        self.coco_dir = r''

        # sar det
        self.sardet_train_dir = r''
