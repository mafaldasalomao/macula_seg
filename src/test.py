import tensorflow as tf
from tensorflow import keras
from keras.utils import CustomObjectScope
from keras.models import load_model
from utils import iou, dice_coef, dice_coef_loss
import numpy as np



with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss}):
  modelo_teste = load_model('/content/modelo_drive_40.h5')

  modelo_teste.load_weights('/content/modelo_drive_40.h5')


#predict

def segmenta_img(img, modelo_teste):
  predicao = modelo_teste.predict(np.expand_dims(img, axis=0))[0]
  predicao = predicao > 0.5 #estudo do limiar between 
  predicao = predicao.astype(np.int32)
  predicao = np.squeeze(predicao, axis=-1)
  return predicao

#for de 0.0 ate 1.0 para determinar o melhor limiar



