from model import modelo_unet
from utils import dice_coef, dice_coef_loss, iou
from tensorflow.keras.optimizers import Adam
from dataset import carregar_dataset, carregar_imagens, embaralha, tf_dataset, criar_diretorio, novas_imagens
import os


#Define hiperparams
epochs = 50
batch_size = 32
lr = 1e-4
img_altura = 256
img_largura = 256



#pre-process dataset 
path_dataset = "data\\dataset"
(X_train, y_train) = carregar_dataset(path_dataset)

#create dataset 
criar_diretorio('data\\dataset_final\\train\\images')
criar_diretorio('data\\dataset_final\\train\\masks')

criar_diretorio('data\\dataset_final\\test\\images')
criar_diretorio('data\\dataset_final\\test\\mask')
novas_imagens(X_train, y_train, 'data\\dataset_final\\train\\', augmentation=True)

#load dataset created
dir_dataset = 'data\\dataset_final\\'
path_train = os.path.join(dir_dataset, 'train')
path_val = os.path.join(dir_dataset, 'train')
#print(path_train, path_val)

X_train, y_train = carregar_imagens(path_train)
X_train, y_train = embaralha(X_train, y_train)
X_val, y_val = carregar_imagens(path_val)

dataset_train = tf_dataset(X_train, y_train, batch_size=batch_size)
dataset_val = tf_dataset(X_val, y_val, batch_size=batch_size)


#create model
model = modelo_unet((img_altura, img_largura, 3))
#compile model
model.compile(loss=dice_coef_loss, optimizer=Adam(lr), metrics = [dice_coef, iou, 'accuracy'])
#model.summary()


#train model

history = model.fit(dataset_train, epochs=epochs, validation_data=dataset_val)


#save model
model.save("models/unet.keras")


#save loss curves
