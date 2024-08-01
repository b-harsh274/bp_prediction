import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D, Concatenate, Cropping1D, ZeroPadding1D
from tensorflow.keras.models import Model


class TimeseriesUNetModel:
    def Timeseries_Unet(self):
        C = 1  # number of channels in the input time series
        M = 1  # number of anomaly classes
        L = 200  # length of the input time series
        K = 3  # kernel size for convolution layers
        F = [16, 32, 64, 128]  # number of filters for each encoding section
        P = 2  # pool size and upsampling rate

        # Define the input layer
        input_layer = Input(shape=(L, C))

        # Define the encoding sections
        encoding_layers = []
        x = input_layer
        for f in F:
            x = Conv1D(filters=f, kernel_size=K, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv1D(filters=f, kernel_size=K, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            encoding_layers.append(x)
            x = MaxPooling1D(pool_size=P)(x)

        # Define the decoding sections
        decoding_layers = []
        for i in range(len(F) - 1, -1, -1):
            f = F[i]
            x = UpSampling1D(size=P)(x)
            # Calculate the difference in length
            length_diff = encoding_layers[i].shape[1] - x.shape[1]
            if length_diff > 0:
                x = ZeroPadding1D(padding=(0, length_diff))(x)
            elif length_diff < 0:
                x = tf.keras.layers.Cropping1D(cropping=(0, -length_diff))(x)
            x = Concatenate()([encoding_layers[i], x])
            x = Conv1D(filters=f, kernel_size=K, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv1D(filters=f, kernel_size=K, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            decoding_layers.append(x)

        # Define the output layer
        output_layer = Conv1D(filters=1, kernel_size=1)(decoding_layers[-1])
        output_layer = Activation('sigmoid')(output_layer)

        # Define the model
        model = Model(inputs=input_layer, outputs=output_layer)

        # Print the model summary
        model.summary()

        return model
