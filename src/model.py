from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, ReLU
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger
from keras.metrics import Recall, Precision

def bloco_conv(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)

    return x

def bloco_encoder(input, num_filters):
    x = bloco_conv(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p  


def bloco_decoder(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = bloco_conv(x, num_filters)
    return x 


def modelo_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = bloco_encoder(inputs, 64)
    s2, p2 = bloco_encoder(p1, 64*2) #128
    s3, p3 = bloco_encoder(p2, 64*4) #256
    s4, p4 = bloco_encoder(p3, 64*8) #512

    b1 = bloco_conv(p4, 64*16)  #1024

    d1 = bloco_decoder(b1, s4, 64*8) #512
    d2 = bloco_decoder(d1, s3, 64*4) #256
    d3 = bloco_decoder(d2, s2, 64*2) #128
    d4 = bloco_decoder(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  

    model = Model(inputs, outputs, name="UNet")
    return model
