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

    data_root = r''
    project_root = r''
    multiprocessing.set_start_method('spawn')
    e = ExperimentRGBS(data_root, project_root)
    tracker_params = [
        {'name': 'scanet', 'version': 'baseline'},
    ]

    tracker_factory = TrackerFactory()

    for params in tracker_params:
        e.multi_run(params, tracker_factory, threads=4)
        e.eval(tracker_factory(params).name)