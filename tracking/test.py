import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import lib.models
import multiprocessing
from lib.test import ExperimentRGBS
from lib.test.run import TrackerFactory

if __name__ == '__main__':

    data_root = r'/home/liyunfeng/code/dev/RGBS50'
    project_root = r'/home/liyunfeng/code/SCANet'
    multiprocessing.set_start_method('spawn')
    e = ExperimentRGBS(data_root, project_root)
    tracker_params = [
        {'name': 'scanet', 'version': 'baseline'},
    ]

    tracker_factory = TrackerFactory()

    for params in tracker_params:
        e.run(tracker_factory(params))
        e.eval(tracker_factory(params).name)
