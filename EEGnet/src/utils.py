import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import operator
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import tensorflow as tf


def to_categorical(y,num_classes):
    return np.eye(num_classes,dtype=np.uint8)[y]

def set_seed(seed_value = 0):
    ''' Set detereministic seed'''
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)

def single_auc_loging(history,title,path_to_save):
    """
    Function for ploting nn-classifier performance. It makes two subplots.
    First subplot with train and val losses
    Second with val auc
    Function saves plot as a picture and as a pkl file

    :param history: history field of history object, witch returned by model.fit()
    :param title: Title for picture (also used as filename)
    :param path_to_save: Path to save file
    :return:
    """
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,12))

    if 'loss' in history.keys():
        loss_key = 'loss'  # for simple NN
    elif 'class_out_loss' in history.keys():
        loss_key = 'class_out_loss'  # for DAL NN
    else:
        raise ValueError('Not found correct key for loss information in history')

    ax1.plot(history[loss_key],label='cl train loss')
    ax1.plot(history['val_%s' %loss_key],label='cl val loss')
    ax1.legend()
    min_loss_index,max_loss_value = min(enumerate(history['val_loss']), key=operator.itemgetter(1))
    ax1.set_title('min_loss_%.3f_epoch%d' % (max_loss_value, min_loss_index))
    ax2.plot(history['val_auc'])
    max_auc_index, max_auc_value = max(enumerate(history['val_auc']), key=operator.itemgetter(1))
    ax2.set_title('max_auc_%.3f_epoch%d' % (max_auc_value, max_auc_index))
    f.suptitle('%s' % (title))
    plt.savefig('%s/%s.png' % (path_to_save,title), figure=f)
    plt.close()
    with open('%s/%s.pkl' % (path_to_save,title), 'wb') as output:
        pickle.dump(history,output,pickle.HIGHEST_PROTOCOL)







