""" Helper functions for the generation of data-augmented images.

All the functions were taken from https://github.com/rykov8/ssd_keras/blob/master/SSD_training.ipynb
"""

import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def grayscale(rgb):
    return rgb.dot([0.299, 0.587, 0.114])


def saturation(rgb, saturation_var=0.5):
    gs = grayscale(rgb)
    alpha = 2 * np.random.random() * saturation_var
    alpha = alpha + 1 - saturation_var
    rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
    return np.array(np.clip(rgb, 0, 255), dtype=np.uint8)


def brightness(rgb, brightness_var=0.5, saturation_var=0.5):
    alpha = 2 * np.random.random() * brightness_var
    alpha = alpha + 1 - saturation_var
    rgb = rgb * alpha
    return np.array(np.clip(rgb, 0, 255), dtype=np.uint8)


def contrast(rgb, contrast_var=0.5):
    gs = grayscale(rgb).mean() * np.ones_like(rgb)
    alpha = 2 * np.random.random() * contrast_var
    alpha = alpha + 1 - contrast_var
    rgb = rgb * alpha + (1 - alpha) * gs
    return np.array(np.clip(rgb, 0, 255), dtype=np.uint8)


def lighting(img, lighting_std=0.5):
    cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    noise = np.random.randn(3) * lighting_std
    noise = eigvec.dot(eigval * noise) * 255
    img = img + noise
    return np.array(np.clip(img, 0, 255), dtype=np.uint8)


def horizontal_flip(img, y, hflip_prob=0.5):
    if np.random.random() < hflip_prob:
        img = img[:, ::-1]
        y[:, [0, 2]] = 1 - y[:, [2, 0]]
    return img, y


def vertical_flip(img, y, vflip_prob=0.5):
    if np.random.random() < vflip_prob:
        img = img[::-1]
        y[:, [1, 3]] = 1 - y[:, [3, 1]]
    return img, y

#############################################################

def rotate(self, image, angle=10, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

def brightness_augment(self,img, factor=0.5):
    '''
    Change the brightness of the image
    :param img: image
    :param factor: the scale to change the brigthness
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb
    
def elastic_transform(self,image, alpha=10, sigma=2, random_state=None):
    if len(image.shape)== 2:
        if random_state is None:
            random_state = np.random.RandomState(None)
    
        shape = image.shape
    
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        
        return map_coordinates(image, indices, order=1).reshape(shape)
    elif len(image.shape)== 3:
        z = np.zeros(image.shape,dtype=type(image[0,0,0]))
        for i in range(3):
            z[:,:,i] = self.elastic_transform(image[:,:,i],alpha,sigma)
        return z
    