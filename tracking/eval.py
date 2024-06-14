import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import multiprocessing
from lib.test import ExperimentRGBS
from lib.test.tracker import *
import lib.models

if __name__ == '__main__':
    data_root = r''
    project_root = r''

    e = ExperimentRGBS(data_root, project_root)
    tracker_list = ['scanet_baseline']

    for tracker in tracker_list:
        # e.run(tracker)
        e.eval(tracker, protocol=1)
