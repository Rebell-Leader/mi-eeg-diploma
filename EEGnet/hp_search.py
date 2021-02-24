from cv_simple_ebci_torch import cv_per_subj_test
import optuna
import os
import numpy as np
import sys

sys.path.append(os.path.join(os.path.split(os.getcwd())[0], 'data_loader'))

from data import DataBuildClassifier

if __name__ == '__main__':
    # data = DataBuildClassifier('/home/likan_blk/BCI/NewData')
    data = DataBuildClassifier('./../new_data')
    all_subjects = [25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38]
    # all_subjects = [6]
    subjects = data.get_data(all_subjects, shuffle=False, windows=[(0.2, 0.5)], baseline_window=(0.2, 0.3),
                             resample_to=params['resample_to'])
    study = optuna.create_study(direction='maximize')
