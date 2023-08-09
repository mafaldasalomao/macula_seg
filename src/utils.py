from keras import backend as K
from matplotlib import pyplot as plt
import cv2
""" IoU """
def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

""" Dice Coefficient """
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

""" Dice Coefficient Loss """
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
def plot_images(imgs_geradas, masks_geradas):

    for i in range(10):
        fig = plt.figure(figsize=(10,7))

        fig.add_subplot(1,2,1)
        img_teste = cv2.imread(imgs_geradas[i])
        plt.imshow(cv2.cvtColor(img_teste, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        fig.add_subplot(1,2,2)
        mask_teste = cv2.imread(masks_geradas[i])
        plt.imshow(cv2.cvtColor(mask_teste, cv2.COLOR_BGR2RGB))
        plt.axis('off')