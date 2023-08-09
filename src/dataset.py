import os
import numpy as np
import cv2
from tqdm import tqdm
#from matplotlib import pyplot as plt
import tensorflow
import random
from glob import glob
from imageio import mimread

from sklearn.utils import shuffle


def carregar_dataset(caminho):
  X_train = sorted(glob(os.path.join(caminho, "images", "*.tif")))
  y_train = sorted(glob(os.path.join(caminho, "masks", "*.tif")))

  #X_test = sorted(glob(os.path.join(caminho, "test", "images", "*.tif")))
  #y_test = sorted(glob(os.path.join(caminho, "test", "1st_manual", "*.gif")))

  return (X_train, y_train)


def novas_imagens(imagens, mascaras, dir_salvo, img_altura=256, img_largura=256, augmentation=True):
  for idx, (x, y) in tqdm(enumerate(zip(imagens, mascaras)), total = len(imagens)):
    nome = x.split('\\')[-1].split('.')[0]
    #print(nome)

    x = cv2.imread(x)
    y = mimread(y)[0]
    X = [x]
    y = [y]

    indice = 0
    for img, mask in zip(X, y):
      img = cv2.resize(img, (img_largura, img_altura))
      mask = cv2.resize(mask, (img_largura, img_altura))

      if len(X) == 1:
        tmp_img_nome = f"{nome}.png"
        tmp_mask_nome = f"{nome}.png"
      else:
        tmp_img_nome = f"{nome}_{indice}.png"
        tmp_mask_nome = f"{nome}_{indice}.png"

      path_imagem = os.path.join(dir_salvo, "images", tmp_img_nome)
      path_mascara = os.path.join(dir_salvo, "masks", tmp_mask_nome)

      cv2.imwrite(path_imagem, img)
      cv2.imwrite(path_mascara, mask)

      indice += 1

def carregar_imagens(path):
  x = sorted(glob(os.path.join(path, 'images', '*.png')))
  y = sorted(glob(os.path.join(path, 'masks', '*.png')))
  return x, y


def ler_img_dataset(caminho):
  caminho = caminho.decode()
  img = cv2.imread(caminho)
  img = img / 255.0
  img = img.astype(np.float32)
  return img


def ler_mask_dataset(caminho):
  caminho = caminho.decode()
  img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
  img = img / 255.0
  img = img.astype(np.float32)
  img = np.expand_dims(img, axis = -1) # (512, 512) -> (512, 512, 1)
  return img

def criar_diretorio(caminho):
  if not os.path.exists(caminho):
    os.makedirs(caminho)

def tf_parse(x, y, img_altura=256, img_largura=256):
  def _parse(x, y):
    x = ler_img_dataset(x)
    y = ler_mask_dataset(y)
    return x, y

  x, y = tensorflow.numpy_function(_parse, [x, y], [tensorflow.float32, tensorflow.float32])
  x.set_shape([img_altura, img_largura, 3])
  y.set_shape([img_altura, img_largura, 1])
  return x, y

def tf_dataset(X, y, batch_size=32):
  dataset = tensorflow.data.Dataset.from_tensor_slices((X, y))
  dataset = dataset.map(tf_parse)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(4)
  return dataset

def embaralha(x, y, seed=42):
  x, y = shuffle(x, y, random_state=seed)
  return x, y

