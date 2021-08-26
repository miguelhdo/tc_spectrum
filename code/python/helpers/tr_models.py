from datetime import datetime
import time
import pandas as pd
import tensorflow as tf

# Tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Keras API
from tensorflow.keras.callbacks import History, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Activation, Dropout, Add, TimeDistributed, \
    Multiply, Conv1D, Conv2D, GRU, AveragePooling2D, MaxPooling2D, MaxPooling1D, AveragePooling1D, BatchNormalization, LSTM
from tensorflow.keras.models import Model, load_model, Sequential

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class CNNClassifier():
    def __init__(self, input_shape, num_classes, kernel_size = 2, num_filters = 32, dropout=0.0, load=False, load_dir='./', load_file='model.h5',
                 prefix=''):
        """
        Parameters:

        """
        self.input_shape = input_shape
        self.output_shape = num_classes
        self.dropout = dropout
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        print('Parameters for CNN model')
        print('dropout: ', str(self.dropout))
        print('kernel size: ', str(self.kernel_size))
        print('Number of Filters: ', str(self.num_filters))
        
        self.prefix = prefix

        if load is True:
            self.model = load_model(load_dir + load_file, custom_objects={'tf': tf})
            self.prev_history = pd.read_csv(load_dir + load_file + "_training_history.csv")
            self.start_idx = len(self.prev_history)
            self.history = None
        else:
            self.model = self.construct_model()
            self.start_idx = 0
            self.history = None
            self.prev_history = None

    def construct_model(self):

        model = Sequential()
        model.add(Conv2D(self.num_filters, self.kernel_size, activation="relu", padding = 'same', kernel_initializer="glorot_normal", input_shape=self.input_shape + (1,)))
        model.add(MaxPooling2D(4, padding='same'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(self.num_filters, self.kernel_size, activation="relu", padding = 'same'))
        model.add(MaxPooling2D(4, padding='same'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(self.num_filters, self.kernel_size, activation="relu", padding = 'same'))
        model.add(MaxPooling2D(4, padding='same'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(self.num_filters, self.kernel_size, activation="relu", padding = 'same'))
        model.add(MaxPooling2D(4, padding='same'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Dense(self.output_shape, activation='softmax'))

        return model

    def get_model(self):
        return self.model
    
    def set_best_trained_model(self):
        file_prefix = self.prefix + "_classifier"
        save_dir = self.save_dir
        saved = save_dir + file_prefix + ".h5"
        self.model = load_model(saved, custom_objects={'tf': tf}) 

    def fit(self, X, Y, validation_data=None, epochs=100, batch_size=64, optimizer='adam', save=False, save_dir='./', patience=10):
        self.save_dir=save_dir
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']

        print('Model will be trained with the following configuration: ')
        print('Optimizer: ', optimizer)
        print('Batch size: ', batch_size)
        print('Max number of epochs: ', epochs)
        
        # set callback functions
        if save:
            print('Model will be saved at ', self.save_dir)
            file_prefix = self.prefix + "_classifier"
            saved = save_dir + file_prefix + ".h5"
            hist = save_dir + file_prefix + "_training_history.csv"

            if validation_data is None:
                checkpoints = ModelCheckpoint(filepath=saved, monitor='loss', verbose=0, save_best_only=True)
            else:
                checkpoints = ModelCheckpoint(filepath=saved, monitor='val_loss', verbose=0, save_best_only=True)
            history = History()
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
            time_callback = TimeHistory()
            callbacks = [history, checkpoints, early_stop, time_callback]

        else:
            callbacks = None

        # compile the model
        self.model.compile(optimizer, loss, metrics)
        try:
            if validation_data is None:
                self.history = self.model.fit(X, Y, shuffle=True, validation_split=0.2, 
                                              verbose=1,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              callbacks=callbacks,
                                              initial_epoch=self.start_idx)
            else:
                self.history = self.model.fit(X, Y, shuffle=True, batch_size=batch_size,
                                              epochs=epochs,
                                              verbose=1,
                                              validation_data=validation_data, callbacks=callbacks,
                                              initial_epoch=self.start_idx)
                
            self.time_epochs=time_callback.times
        except:
            if save:
                df = pd.DataFrame.from_dict(history.history)
                df.to_csv(hist, encoding='utf-8', index=False)
            raise
            sys.exit()
        return self.history

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
    
    def get_time_per_epoch(self):
        return self.time_epochs

class GRUClassifier():
    def __init__(self, input_shape, num_classes, gru_units = 128, dropout=0.3, load=False, load_dir='./', load_file='model.h5',
                 prefix=''):
        
        """
        Parameters:

        """
        self.input_shape = input_shape
        self.output_shape = num_classes
        self.dropout = dropout
        self.gru_units = gru_units
        print('Parameters for GRU model')
        print('dropout: ', str(self.dropout))
        print('gru units: ', str(self.gru_units))

        # save WaveNet prefix name for outputfiles
        self.prefix = prefix

        if load is True:
            self.model = load_model(load_dir + load_file, custom_objects={'tf': tf})
            self.prev_history = pd.read_csv(load_dir + load_file + "_training_history.csv")
            self.start_idx = len(self.prev_history)
            self.history = None
        else:
            self.model = self.construct_model()
            self.start_idx = 0
            self.history = None
            self.prev_history = None

    def construct_model(self):
        model = Sequential()

        model.add(GRU(self.gru_units, dropout=self.dropout, return_sequences=True, input_shape=(self.input_shape[1], self.input_shape[0])))
        model.add(GRU(self.gru_units, dropout=self.dropout, return_sequences=True))
        model.add(GRU(self.gru_units, dropout=self.dropout))
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Dense(self.output_shape, activation='softmax'))
        return model

    def get_model(self):
        return self.model

    def set_best_trained_model(self):
        file_prefix = self.prefix + "_classifier"
        save_dir = self.save_dir
        saved = save_dir + file_prefix + ".h5"
        self.model = load_model(saved, custom_objects={'tf': tf}) 
        
    def fit(self, X, Y, validation_data=None, epochs=100, batch_size=32, optimizer='adam', save=False, save_dir='./', patience=10):
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        self.save_dir=save_dir

        print('Model will be trained with the following configuration: ')
        print('Optimizer: ', optimizer)
        print('Batch size: ', batch_size)
        print('Max number of epochs: ', epochs)
        
        if save:
            print('Model will be saved at ', self.save_dir)
            file_prefix = self.prefix + "_classifier"
            saved = save_dir + file_prefix + ".h5"
            hist = save_dir + file_prefix + "_training_history.csv"

            if validation_data is None:
                checkpoints = ModelCheckpoint(filepath=saved, monitor='loss', verbose=1, save_best_only=True)
            else:
                checkpoints = ModelCheckpoint(filepath=saved, monitor='val_loss', verbose=1, save_best_only=True)
            history = History()
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
            time_callback = TimeHistory()
            callbacks = [history, checkpoints, early_stop, time_callback]

        else:
            callbacks = None

        # compile the model
        self.model.compile(optimizer, loss, metrics)
        try:
            if validation_data is None:
                self.history = self.model.fit(X, Y, shuffle=True, validation_split=0.2, batch_size=batch_size,
                                              epochs=epochs,
                                              callbacks=callbacks,
                                              initial_epoch=self.start_idx)
            else:
                self.history = self.model.fit(X, Y, shuffle=True, batch_size=batch_size,
                                              epochs=epochs,
                                              validation_data=validation_data, callbacks=callbacks,
                                              initial_epoch=self.start_idx)
                
            self.time_epochs=time_callback.times
        except:
            if save:
                df = pd.DataFrame.from_dict(history.history)
                df.to_csv(hist, encoding='utf-8', index=False)
            raise
            sys.exit()
        return self.history

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)
    
    def get_time_per_epoch(self):
        return self.time_epochs