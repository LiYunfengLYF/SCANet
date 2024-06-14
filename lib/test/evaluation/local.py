import os
from lib.test.evaluation.environment import EnvSettings


def local_env_settings(env_num):
    settings = EnvSettings()
    settings.prj_dir = '/home/liyunfeng/code/SCANet'
    settings.result_plot_path = os.path.join(settings.prj_dir, 'output/test/result_plots')
    settings.results_path = os.path.join(settings.prj_dir, 'output/test/tracking_results')
    settings.save_dir = os.path.join(settings.prj_dir, 'output')
    return settings
