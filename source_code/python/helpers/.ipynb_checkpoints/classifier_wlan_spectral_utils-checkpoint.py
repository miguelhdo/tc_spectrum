import itertools
import time
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import trange

from helpers import tr_models as tr_models

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def reshape_for_model(model_type, X_train, X_val, X_test):
    if model_type == 'LSTM':
        print("Re-shaping for LSTM")
        X_train_lstm = np.transpose(X_train, [0,2,1])
        X_val_lstm = np.transpose(X_val, [0,2,1])
        X_test_lstm = np.transpose(X_test, [0,2,1])
        return X_train_lstm, X_val_lstm, X_test_lstm
    elif model_type == 'CNN':
        print("Re-shaping for CNN")
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        return X_train_cnn, X_val_cnn, X_test_cnn
    else:
        print("No need of Re-shaping for Wavenet")
        return X_train, X_val, X_test


def create_and_train_model(model_type, num_iq_seq, num_classes, prefix, X_train, Y_train, X_val, Y_val, X_test, Y_test, save=True, epochs = 100, dilation_depth=3, n_filters=32, dropout = 0.0, kernel_size=2, optimizer="adam", batch_size=32):
    print("Creating the model")
    if model_type=='LSTM':
        print("Model type LSTM")
        model = tr_models.LSTMClassifier((2,num_iq_seq), num_classes, prefix=prefix)
    elif model_type == 'Wavenet':
        print("Model type Wavenet")
        model = tr_models.WaveNetClassifier((2,num_iq_seq), (num_classes,), prefix=prefix, dilation_depth=dilation_depth, n_filters=n_filters, dropout=dropout, kernel_size=kernel_size)
    else:
        print("Model type CNN")
        model = tr_models.CNNClassifier((2,num_iq_seq), num_classes, prefix=prefix, n_filters=n_filters, dropout=dropout, kernel_size=kernel_size)
    print("Model created")

    print("Model start training")
    history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), save=save, epochs = epochs, optimizer=optimizer, batch_size=batch_size)

    print("Training finished, Loading best model")
    model.set_best_trained_model()
   
    model_evaluations = {}
    
    print("Evaluation Training set")
    model_evaluations['Training'] = model.evaluate(X_train, Y_train)
    
    print("Evaluation Validation set")
    model_evaluations['Validation'] = model.evaluate(X_val, Y_val)
    
    print("Evaluation Test set")
    model_evaluations['Test'] = model.evaluate(X_test, Y_test)
    
    print('Computing prediction time training')
    start = time.clock() 
    model.predict(X_train)
    end = time.clock()
 
    pred_time_training = {}
    pred_time_training['n_samples']=len(X_train)
    pred_time_training['time_pred']=end-start
    pred_time_training['t_sample']=pred_time_training['time_pred']/pred_time_training['n_samples']
    model_evaluations['prediction_time_training']=pred_time_training
    
    print('Computing prediction time test')
    start = time.clock() 
    model.predict(X_test)
    end = time.clock()

    pred_time_test = {}
    pred_time_test['n_samples']= len(X_test)
    pred_time_test['time_pred']=end-start
    pred_time_test['t_sample']=pred_time_test['time_pred']/pred_time_test['n_samples']
    model_evaluations['prediction_time_testing']=pred_time_test
    
    print('Computing confusion matrix')
    Y_pred=np.argmax(model.predict(X_test),1)
    Y_true=np.argmax(Y_test,1)

    model_evaluations['confusion_matrix_normalized']= confusion_matrix(Y_true, Y_pred, normalize='true')
    model_evaluations['confusion_matrix'] = confusion_matrix(Y_true, Y_pred)
    
    print('Computing precision, recall, fscore, and support')
    model_evaluations['precision_recall_fscore_support'] = precision_recall_fscore_support(Y_true, Y_pred)
    model_evaluations['precision_recall_fscore_support_micro'] = precision_recall_fscore_support(Y_true, Y_pred, average='micro')
    model_evaluations['precision_recall_fscore_support_macro'] = precision_recall_fscore_support(Y_true, Y_pred, average='macro')
    
    
    return model_evaluations, model

def eval_and_plot_cm_4_model(model, x, y, target_names, cm_filename, cm_dir='./', precision = "{:0.4f}"):
    y_pred = np.argmax(model.predict(x), 1)
    y_true = np.argmax(y, 1)
    plot_confusion_matrix_mc(y_true, y_pred, target_names, cm_filename, cm_dir, precision=precision)
    return model.evaluate(x, y)


def get_raw_xy_spectrum(dataset_folder, dataset_filename, num_samples=-1):
    path_to_dataset = dataset_folder + dataset_filename
    f = h5py.File(path_to_dataset, 'r')
    X_obj = f['X']
    Y_obj = f['Y']
    [r, c] = f['X'].shape
    X = []
    Y = []
    if num_samples > -1:
        if num_samples < r:
            r = num_samples
    for i in trange(r):
        X.append(np.array(f[X_obj[i][0]], dtype=np.int16))
        Y.append(np.array(f[Y_obj[i][0]], dtype=np.uint8))
    return X, Y


def get_raw_xy_packets(dataset_folder, dataset_filename, num_samples=-1):
    path_to_dataset = dataset_folder + dataset_filename
    f = h5py.File(path_to_dataset, 'r')
    X_obj = f['X_payload']
    Y_obj = f['Y']
    [r, c] = f['X_payload'].shape
    X = []
    Y = []
    if num_samples > -1:
        if num_samples < r:
            r = num_samples
    for i in trange(r):
        X.append(np.array(f[X_obj[i][0]], dtype=np.int16))
        Y.append(np.array(f[Y_obj[i][0]], dtype=np.uint8))
    return X, Y


def pad_or_trunc_x_and_scale(x_raw, num_iq_seq, padding, scale=False, padding_val=0, scale_range=(-1, 1)):
    num_samples = len(x_raw)
    
    if scale:
        data_type = np.float32
    else:
        data_type = np.int16
        
    x_padded = np.empty((num_samples, 2, num_iq_seq), dtype=data_type)
    for i in trange(num_samples):
        x_temp = np.array(x_raw[i], dtype=data_type)
        x_temp = x_temp.reshape((2, -1))
        x_temp = pad_sequences(x_temp, maxlen=num_iq_seq, dtype=data_type, padding=padding, truncating=padding,
                               value=padding_val)
        if scale:
            x_temp = x_temp.reshape((-1, 1))
            x_temp = minmax_scale(x_temp, feature_range=scale_range, axis=0)
            x_temp = x_temp.reshape((2, -1))
        x_padded[i, :] = x_temp
    return x_padded


def get_one_hot_labels(Y, num_classes, label):
    Y = np.array(Y)
    Y = Y[:, label]
    Y_cat = to_categorical(Y, num_classes=num_classes)
    print("Labels created")
    return Y_cat


def get_xy_4_training(X, Y, seed, reduce=False, ts=0.2):
    print("Starting first partitioning")
     
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=ts, random_state=seed)
    del X, Y
    
    if reduce:
        print("Using sample dataset (20 pct of original one)")
        X_train, X_val_test, Y_train, Y_val_test = train_test_split(X_val_test, Y_val_test, test_size=0.3, random_state=seed)
        
    print("First partitioning done. Starting partitioning 2")
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=seed)
    

    print("Final partitioning done")
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def get_xy(dataset_folder, dataset_filename, num_iq_seq, padding, label, num_classes, scale=False, padding_val=0,
           scale_range=(0, 1)):
    print("Reading raw X and Y from file")
    time.sleep(2)
    x_raw, y_raw = get_raw_xy(dataset_folder, dataset_filename)
    print("Reading raw X and Y from file....DONE")
    print("Padding/Truncating sequence")
    time.sleep(2)
    X = pad_or_trunc_x_and_scale(x_raw, num_iq_seq, padding, scale=scale, padding_val=padding_val,
                                 scale_range=scale_range)
    del x_raw
    print("Padding/Truncating sequence....DONE")
    print("Getting labels")
    time.sleep(2)
    Y = get_one_shot_labels(y_raw, num_classes, label)
    del y_raw
    print("Getting labels....DONE")
    return X, Y

def plot_confusion_matrix_mc(Y_true, Y_pred,
                             target_names,
                             cm_filename,
                             cm_dir='./',
                             title='Confusion matrix',
                             cmap=None,
                             normalize=True, precision = "{:0.2f}"):
    cm = confusion_matrix(Y_true, Y_pred)

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, precision.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    plt.savefig((cm_dir + cm_filename))
    plt.show()
def compute_and_save_conf_matrix(model, X_test, Y_test, labels, filename_prefix = '', precision = "{:0.2f}"):
    file_out_conf_matrix = filename_prefix+'_conf_matrix.pdf'
    Y_pred=np.argmax(model.predict(X_test),1)
    Y_true=np.argmax(Y_test,1)
    plot_confusion_matrix_mc(Y_true, Y_pred, labels, file_out_conf_matrix, precision=precision)

def add_result(results_container, NN_arq, length_iq, result, model_params, labels_string):
    #Check if the container for results of the NN have been already created. If not, create an empy one
    if NN_arq not in results_container:
        results_container[NN_arq] = {}
    results_iq_seq = results_container[NN_arq]
    
    #Check if there is a container for results for a given iq sample length. If not, create an empty one 
    if length_iq not in results_iq_seq:
        temp =[]
        results_iq_seq[length_iq]=temp
    
    results_iq_seq[length_iq].append(result)
    results_container[NN_arq]['model_parameters']=model_params
    results_container[NN_arq]['class_labels']=labels_string
    return results_container

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def save_results_to_json(result_file_name, results):
    with open(result_file_name, 'w') as fp:
        json.dump(results, fp,indent=4, cls=NumpyEncoder)

    