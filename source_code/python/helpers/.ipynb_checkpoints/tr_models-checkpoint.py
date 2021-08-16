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
    Multiply, Conv1D, Conv2D, AveragePooling2D, MaxPooling2D, MaxPooling1D, AveragePooling1D, BatchNormalization, LSTM
from tensorflow.keras.models import Model, load_model, Sequential

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class CNNClassifier():
    def __init__(self, input_shape, num_classes, kernel_size = 2, n_filters = 32, dropout=0.0, load=False, load_dir='./', load_file='model.h5',
                 prefix=''):
        """
        Parameters:

        """
        self.input_shape = input_shape
        self.output_shape = num_classes
        self.dropout = dropout
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        print('Model parameters: kernel size ', str(self.kernel_size), ', number of filter ', str(self.n_filters), ', and dropout ', str(self.dropout))
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
        model.add(Conv2D(self.n_filters, self.kernel_size, activation="relu", padding = 'same', kernel_initializer="glorot_normal", input_shape=self.input_shape + (1,)))
        model.add(MaxPooling2D(4, padding='same'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(self.n_filters, self.kernel_size, activation="relu", padding = 'same'))
        model.add(MaxPooling2D(4, padding='same'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(self.n_filters, self.kernel_size, activation="relu", padding = 'same'))
        model.add(MaxPooling2D(4, padding='same'))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(self.n_filters, self.kernel_size, activation="relu", padding = 'same'))
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
        save_dir = './'
        saved = save_dir + file_prefix + ".h5"
        self.model = load_model(saved, custom_objects={'tf': tf}) 

    def fit(self, X, Y, validation_data=None, epochs=100, batch_size=64, optimizer='adam', save=False, save_dir='./', patience=10):
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        print("Training for ", str(epochs), ' epochs and batch size ',str(batch_size))
        # set callback functions
        if save:
            # datetime object containing current date and time
            #time_now = datetime.now()
            #dt_string = time_now.strftime("%d%m%Y%H%M%S")
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

class WaveNetClassifier():
    def __init__(self, input_shape, output_shape, kernel_size=2, dilation_depth=3, n_filters=32, dropout=0.3, task='classification',
                 regression_range=None, load=False, load_dir='./', load_file='model.h5', prefix=''):
        """
        Parameters:
          input_shape: (tuple) tuple of input shape. (e.g. If input is 6s raw waveform with sampling rate = 16kHz, (96000,) is the input_shape)
          output_shape: (tuple)tuple of output shape. (e.g. If we want classify the signal into 100 classes, (100,) is the output_shape)
          kernel_size: (integer) kernel size of convolution operations in residual blocks
          dilation_depth: (integer) type total depth of residual blocks
          n_filters: (integer) # of filters of convolution operations in residual blocks
          task: (string) 'classification' or 'regression'
          regression_range: (list or tuple) target range of regression task
          load: (bool) load previous WaveNetClassifier or not
          load_dir: (string) the directory where the previous model exists
        """
        print(kernel_size, dilation_depth, n_filters, dropout)
        # save task info
        self.task = task
        if task == 'regression':
            if regression_range[0] == 0:
                self.activation = 'sigmoid'
                self.scale_ratio = regression_range[1]
            elif regression_range[0] == - regression_range[1]:
                self.activation = 'tanh'
                self.scale_ratio = regression_range[1]
            elif regression_range == None:
                self.activation = 'linear'
                self.scale_ratio = 1
            else:
                print('ERROR: wrong regression range')
                sys.exit()
        elif task == 'classification':
            self.activation = 'softmax'
            self.scale_ratio = 1
        else:
            print('ERROR: wrong task')
            sys.exit()

        # save input info
        if len(input_shape) == 1:
            self.expand_dims = True
        elif len(input_shape) == 2:
            self.expand_dims = False
        else:
            print('ERROR: wrong input shape')
            sys.exit()
        self.input_shape = input_shape

        # save output info
        if len(output_shape) == 1:
            self.time_distributed = False
        elif len(output_shape) == 2:
            self.time_distributed = True
        else:
            print('ERROR: wrong output shape')
            sys.exit()
        self.output_shape = output_shape

        # save hyperparameters of WaveNet
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.n_filters = n_filters
        self.dropout=dropout
        self.manual_loss = None

        # save WaveNet prefix name for outputfiles
        self.prefix = prefix

        if load is True:
            self.model = load_model(load_dir + load_file, custom_objects={'tf': tf})
            # self.prev_history = pd.read_csv(load_dir+load_file+"_training_history.csv")
            # self.start_idx = len(self.prev_history)
            self.history = None
        else:
            self.model = self.construct_model()
            self.start_idx = 0
            self.history = None
            self.prev_history = None

    def residual_block(self, x, i):
        tanh_out = Conv1D(self.n_filters,
                          self.kernel_size,
                          dilation_rate=self.kernel_size ** i,
                          padding='causal',
                          name='dilated_conv_%d_tanh' % (self.kernel_size ** i),
                          activation='tanh'
                          )(x)
        sigm_out = Conv1D(self.n_filters,
                          self.kernel_size,
                          dilation_rate=self.kernel_size ** i,
                          padding='causal',
                          name='dilated_conv_%d_sigm' % (self.kernel_size ** i),
                          activation='sigmoid'
                          )(x)
        z = Multiply(name='gated_activation_%d' % (i))([tanh_out, sigm_out])
        skip = Conv1D(self.n_filters, 1, name='skip_%d' % (i))(z)
        res = Add(name='residual_block_%d' % (i))([skip, x])
        return res, skip

    def construct_model(self):
        x = Input(shape=self.input_shape, name='original_input')
        x_reshaped = Reshape(self.input_shape + (1,), name='reshaped_input')(x)
        skip_connections = []

        out = Conv2D(self.n_filters, self.kernel_size, padding='same', name='conv2d')(x_reshaped)
        out = AveragePooling2D((2,1),padding='same')(out)
        out = Dropout(self.dropout)(out)
        out = BatchNormalization(axis=-1)(out)

        out = Reshape((out.shape[2], out.shape[3]), name='conv2d_to_1d')(out)

        out = Conv1D(self.n_filters, 2, dilation_rate=1, padding='causal', name='dilated_conv_1')(out)
        for i in range(1, self.dilation_depth + 1):
                     out, skip = self.residual_block(out, i)
                     skip_connections.append(skip)
        out = Add(name='skip_connections')(skip_connections)
        out = Activation('relu')(out)

        out = Conv1D(self.n_filters, self.kernel_size, padding='same', activation='relu')(out)
        out = AveragePooling1D(10, padding='same')(out)
        out = Dropout(self.dropout)(out)
        out = BatchNormalization(axis=-1)(out)

        out = Conv1D(self.n_filters, self.kernel_size,  padding='same' , activation='relu')(out)
        out = AveragePooling1D(10, padding='same')(out)
        out = Dropout(self.dropout)(out)
        out = BatchNormalization(axis=-1)(out)

        out = Conv1D(self.n_filters, self.kernel_size, padding='same', activation='relu')(out)
        out = AveragePooling1D(5, padding='same')(out)
        out = Dropout(self.dropout)(out)
        out = BatchNormalization(axis=-1)(out)

        out = Flatten()(out)
        out = Dense(128, activation='relu')(out)
        out = Dropout(self.dropout)(out)
        out = BatchNormalization(axis=-1)(out)

        out = Dense(64, activation='relu')(out)
        out = Dropout(self.dropout)(out)
        out = BatchNormalization(axis=-1)(out)

        out = Dense(self.output_shape[0], activation='softmax')(out)
        model = Model(x, out)
                     
        return model

    def get_model(self):
        return self.model

    def add_loss(self, loss):
        self.manual_loss = loss

    def set_best_trained_model(self):
        file_prefix = self.prefix + "_classifier"
        save_dir = './'
        saved = save_dir + file_prefix + ".h5"
        self.model = load_model(saved, custom_objects={'tf': tf}) 
        
    def fit(self, X, Y, validation_data=None, epochs=100, batch_size=32, optimizer='adam', save=False, save_dir='./', patience=10):
        # set default losses if not defined
        if self.manual_loss is not None:
            loss = self.manual_loss
            metrics = None
        else:
            if self.task == 'classification':
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'mean_squared_error'
                metrics = None

        # set callback functions
        if save:
            # datetime object containing current date and time
            #time_now = datetime.now()
            #dt_string = time_now.strftime("%d%m%Y%H%M%S")
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

class LSTMClassifier():
    def __init__(self, input_shape, num_classes, dropout=0.3, load=False, load_dir='./', load_file='model.h5',
                 prefix=''):
        """
        Parameters:

        """
        self.input_shape = input_shape
        self.output_shape = num_classes
        self.dropout = dropout

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

        #model.add(LSTM(128, dropout=0.2, recurrent_activation="sigmoid",
                       #input_shape=(self.input_shape[1], self.input_shape[0])))

        model.add(LSTM(128, dropout=self.dropout, return_sequences=True, input_shape=(self.input_shape[1], self.input_shape[0])))
        # model.add(LSTM(256, dropout=self.dropout, recurrent_activation = "sigmoid",   return_sequences=True))
        model.add(LSTM(128, dropout=self.dropout))
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout))

        model.add(Dense(32, activation='relu'))

        model.add(Dense(self.output_shape, activation='softmax'))
        return model

    def get_model(self):
        return self.model

    def set_best_trained_model(self):
        file_prefix = self.prefix + "_classifier"
        save_dir = './'
        saved = save_dir + file_prefix + ".h5"
        self.model = load_model(saved, custom_objects={'tf': tf}) 
        
    def fit(self, X, Y, validation_data=None, epochs=100, batch_size=32, optimizer='adam', save=False, save_dir='./', patience=10):
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']

        # set callback functions
        if save:
            # datetime object containing current date and time
            #time_now = datetime.now()
            #dt_string = time_now.strftime("%d%m%Y%H%M%S")
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


def cnn_lstm_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, 2, strides=1, padding='same'), input_shape=(1,) + input_shape))
    model.add(TimeDistributed(Dropout(dr)))
    model.add(TimeDistributed(BatchNormalization(axis=-1)))

    model.add(TimeDistributed(Conv2D(32, 2, strides=4, padding='same')))
    model.add(TimeDistributed(Dropout(dr)))
    model.add(TimeDistributed(BatchNormalization(axis=-1)))

    model.add(TimeDistributed(Conv2D(32, 2, strides=10, padding='same')))
    model.add(TimeDistributed(Dropout(dr)))
    model.add(TimeDistributed(BatchNormalization(axis=-1)))

    model.add(TimeDistributed(Conv2D(32, 2, strides=8, padding='same')))
    model.add(TimeDistributed(Dropout(dr)))
    model.add(TimeDistributed(BatchNormalization(axis=-1)))

    # model.add(TimeDistributed(Conv2D(32, 2, strides=4, padding='same')))
    # model.add(TimeDistributed(Dropout(dr)))
    # model.add(TimeDistributed(BatchNormalization(axis=-1)))

    # model.add(TimeDistributed(Conv2D(32, 2, strides=4, padding='same')))
    # model.add(TimeDistributed(Dropout(dr)))
    # model.add(TimeDistributed(BatchNormalization(axis=-1)))

    # model.add(TimeDistributed(Flatten()))
    model.add(Flatten())
    # define LSTM model
    # model.add(LSTM(128, dropout=0.2, recurrent_activation = "sigmoid"))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dr))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
