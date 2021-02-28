from data import Data
import numpy as np
from scipy.io import loadmat
import h5py
import os


class DataProjPursuit_v2(Data):
    def __init__(self, path_to_data, start_epoch=-1.2, end_epoch=1.2, sample_rate=500):
        start_epoch = -1.2  # seconds
        end_epoch = 1.2  # seconds
        sample_rate = 500
        super(DataProjPursuit_v2, self).__init__(path_to_data, start_epoch, end_epoch, sample_rate)

    def _baseline_normalization(self, X, baseline_window=()):
        bl_start = int((baseline_window[0] - self.start_epoch) * self.sample_rate)
        bl_end = int((baseline_window[1] - self.start_epoch) * self.sample_rate)
        baseline = np.expand_dims(X[:, bl_start:bl_end, :].mean(axis=1), axis=1)
        X = X - baseline
        return X

    def get_event_data(self, subj, event, resample_to=None, window=(-1.2, 1.2), eeg_ch=range(21),
                       baseline_window=(-0.4, -0.3), shuffle=False):
        '''

                :param subject: subjects's index to load
                :param shuffle: bool
                :param windows: list of tuples. Each tuple contains two floats - start and end of window in seconds
                :param baseline_window:
                :param resample_to: int, Hz - new sample rate
                :return: numpy array trials x time x channels
        '''
        # data_mat = loadmat(os.path.join(self.path_to_data, 'clean_data.mat'))
        # eeg = data_mat['all_EEG_online'][]
        # eeg = loadmat(os.path.join(self.path_to_data, 'subj %d' % subj, '%s.mat' % event))['EEG']
        print(loadmat(os.path.join(self.path_to_data, '%s' % subj, '%s.mat' % event))['EEG'])
        eeg = loadmat(os.path.join(self.path_to_data, '%s' % subj, '%s.mat' % event))['EEG']
        eeg = eeg.astype('float64')
        print(np.shape(eeg))
        eeg = eeg[:, :, eeg_ch]
        if len(baseline_window):
            eeg = self._baseline_normalization(eeg, baseline_window)
        if (resample_to is not None) and (resample_to != self.sample_rate):
            eeg = self._resample(eeg, resample_to)
            current_sample_rate = resample_to
        else:
            current_sample_rate = self.sample_rate
        time_indices = []

        win_start, win_end = window
        start_window_ind = int((win_start - self.start_epoch) * current_sample_rate)
        end_window_ind = int((win_end - self.start_epoch) * current_sample_rate)
        time_indices.extend(range(start_window_ind, end_window_ind))
        print(time_indices)
        eeg = eeg[:, time_indices, :]

        # if shuffle:
        #     X, y = data_shuffle(X, y)
        # res[subject] = (X, y)
        if shuffle:
            np.random.shuffle(eeg)
        return eeg


if __name__ == '__main__':
    # data = DataProjPursuit_v2('/home/likan_blk/BCI/ProjPursuitData/16-Mar-2020_13-35-45')
    data = DataProjPursuit_v2('../with ica')
    all_subjects = [subj for subj in range(1, 21) if subj not in [12, 13]]
    all_events = ['cl%d' % i for i in range(1, 5)]
    # data.get_event_data(subj=all_subjects[0], event = all_events[0],resample_to=None, window=(-0.2, 0), eeg_ch=range(19),
    #                     baseline_window=(-0.4, -0.3))
    eeg_ch = range(19)
    for train_subject in all_subjects:
        print('subj', train_subject)
        epochs_cl3 = data.get_event_data(train_subject, 'cl3', eeg_ch=eeg_ch, resample_to=369,
                                         window=(-0.2, 0), baseline_window=(-0.4, -0.3), shuffle=False)
        print('cl3', epochs_cl3.shape[0])
        epochs_cl1 = data.get_event_data(train_subject, 'cl1', eeg_ch=eeg_ch, resample_to=369,
                                         window=(-0.2, 0), baseline_window=(-0.4, -0.3), shuffle=False)

        epochs_cl4 = data.get_event_data(train_subject, 'cl4', eeg_ch=eeg_ch, resample_to=369,
                                         window=(-0.2, 0), baseline_window=(-0.4, -0.3), shuffle=False)
        print('cl1+cl4', epochs_cl1.shape[0] + epochs_cl4.shape[0])
        print('\n')
    pass
