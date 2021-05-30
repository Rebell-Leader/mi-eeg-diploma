import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import copy
import math

from base import compute_flat_features, compute_out_size, MaxNormConstraint


class ScaleCNN(nn.Module):
    """Basic CNN for HSCNN.
    Return flattened single scale result.
    """

    def __init__(self, n_chan, n_sample, time_kernel=65):
        super(ScaleCNN, self).__init__()
        self.time_kernel = time_kernel
        self.conv_time = nn.Conv2d(1, 10, (1, self.time_kernel), stride=(1, 3), bias=False)
        w = compute_out_size(n_sample, self.time_kernel, stride=3)

        self.conv_space = nn.Conv2d(10, 10, (n_chan, 1), stride=1, bias=False)
        h = compute_out_size(n_chan, n_chan)

        self.max_pool = nn.MaxPool2d((1, 6), (1, 6))
        w = compute_out_size(w, 6, stride=6)

    def forward(self, X):
        # X is NCHW shape
        out = self.conv_time(X)

        out = self.conv_space(out)
        out = F.elu(out)

        out = self.max_pool(out)

        out = out.view(out.size()[0], -1)
        return out


class BandCNN(nn.Module):
    """Single frequency band CNN for HSCNN.

    Return flattened single freq band result (concate 3 scale results).
    """

    def __init__(self, n_chan, n_sample):
        super(BandCNN, self).__init__()

        self.scale_cnn1 = ScaleCNN(n_chan, n_sample, time_kernel=85)
        self.scale_cnn2 = ScaleCNN(n_chan, n_sample, time_kernel=65)
        self.scale_cnn3 = ScaleCNN(n_chan, n_sample, time_kernel=45)

    def forward(self, X):
        # X in NHW shape

        print(X.shape)
        X = X.view(-1, 1, *X.size()[1:])   # additional dimension for conv layer
        print(X.shape)

        out1 = self.scale_cnn1(X)
        out2 = self.scale_cnn2(X)
        out3 = self.scale_cnn3(X)

        out = torch.cat((out1, out2, out3), dim=-1)

        return out


class HSCNN(nn.Module):
    """A CNN with hybrid convolution scale.
    The paper [1]_ proposes a hybrid scale CNN network for EEG recognition.
    The newtwork is based on BCI Competition IV 2a/2b, which sampled at 250Hz.
    The duration of a trial is 3.5s.
    Only C3, Cz, C4 are recorded.
    Raw signal is filtered by three filters (4-7, 8-13, 13-32), then each frequency band is passed into CNN
    with 3 time convolution kernel (85, 65, 45). Outputs are flattened and concated to a single feature vector.
    The paper uses elu as activation function, though they don't specify the location where it's applyed.
    In my implementation, elus are added following conv_space layer and fc_layer1.
    Some key parameters and tricks:
    dropout probabilty                 0.8
    l2 regularizer(on fc_layer1)       0.01
    SGD optimizer                      0.1 decays every 10 epochs with exp decay rate of 0.9 (400 epochs total)

    The paper also uses data augmentation strategy, see the reference for more information.

    References
    ----------
    .. [1] Dai, Guanghai, et al. "HS-CNN: A CNN with hybrid convolution scale for EEG motor imagery classification." Journal of Neural Engineering (2019).
    """

    def __init__(self, n_chan, n_sample, n_class, n_band=1):
        super(HSCNN, self).__init__()
        self.band_cnns = nn.ModuleList([BandCNN(n_chan, n_sample) for i in range(n_band)]) #n_band is number of frequency bands (3 in paper)

        with torch.no_grad():
            fake_input = torch.zeros(1, n_chan, n_sample)
            fake_output = self.band_cnns[0](fake_input)
            band_size = fake_output.size()[1]

        self.fc_layer1 = nn.Linear(band_size * n_band, 100)
        self.drop = nn.Dropout(0.8)
        self.fc_layer2 = nn.Linear(100, n_class)

    def forward(self, X):
        # X in (N, n_band, n_chan, n_sample)
        out = []
        for i, l in enumerate(self.band_cnns):
            out.append(l(X[:, i, ...]))

        out = torch.cat(out, dim=-1)

        out = self.fc_layer1(out)
        out = F.elu(out)
        out = self.fc_layer2(out)
        return out


def train_model_hcnn(x_tr, y_tr, params, validation_data, epochs=200, batch_size=64, shuffle=True, model_path=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x_val, y_val = validation_data

    # kernLength1 = params.get('kernLength1',64)
    # kernLength2 = params.get('kernLength2',16)
    # F2 = params.get('F2',params['F1']*params['D'])
    model = HSCNN(n_class=2, n_chan=x_tr.shape[2], n_sample=x_tr.shape[3])
    #    dropoutRates=(params['dropoutRate1'], params['dropoutRate1']),
    #    kernLength1=kernLength1, kernLength2=kernLength2, poolKern1=4,
    #    poolKern2=8,
    #    F1=params['F1'],
    #    D=params['D'], F2=F2)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    train_set = EBCIDataset((x_tr, y_tr))
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=shuffle, num_workers=4)

    history = {'loss': [], 'val_loss': [], 'val_auc': []}
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        running_tr_loss = 0.0
        for local_batch, local_labels in train_loader:
            optimizer.zero_grad()
            local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device,
                                                                                                   dtype=torch.long)
            predictions = model(local_batch)
            loss = criterion(predictions, local_labels)
            loss.backward()
            optimizer.step()
            running_tr_loss += loss.item()
        train_loss = running_tr_loss / len(train_loader)
        history['loss'].append(train_loss)
        print("Epoch %d: train loss %f", epoch, train_loss)
        model.eval()
        with torch.set_grad_enabled(False):
            predictions = model(torch.Tensor(x_val).to(device))
            val_loss = criterion(predictions, torch.Tensor(y_val).to(device, dtype=torch.long))
            val_auc = roc_auc_score(y_val, predictions[:, 1].cpu())
            if best_auc <= val_auc:
                best_model = copy.deepcopy(model)
                best_auc = val_auc
            print('Epoch %d: val loss %f\n' % (epoch, val_loss))
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
    torch.save(best_model.state_dict(), model_path)
    return history, model


def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    if type(dilation) is not tuple:
        dilation = (dilation, dilation)

    h = math.floor((h_w[0] + 2 * pad[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + 2 * pad[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h, w


def get_model_params(model):
    params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_dict[name] = param.data
    return params_dict


class EBCIDataset(Dataset):
    def __init__(self, subjects_data):
        self.subjects_data = subjects_data

    def __getitem__(self, item):
        sample = self.subjects_data[0][item]
        label = self.subjects_data[1][item]
        return sample, label

    def __len__(self):
        return len(self.subjects_data[1])
