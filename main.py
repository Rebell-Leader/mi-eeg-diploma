from scipy.io import loadmat
from hcnn import HSCNN
import numpy as np

data = loadmat('../ZHAO/with ica/subj 1/cl1.mat')
eegData = data.get('EEG')
transponatedData = eegData.transpose([0, 2, 1])
# transponatedData = transponatedData[:, np.newaxis, ...
transponatedData = np.expand_dims(transponatedData, 1)
# HSCNN(n_chan=21, n_sample=1201, n_class=2, n_band=1)
HSCNN(n_chan=21, n_sample=1201, n_class=2)
