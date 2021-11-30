from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.python.framework import ops
import keras.backend as K
import numpy as np, cv2, os 
from keras.models import Model

from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class gradCam():
	def __init__(self, model):
		self.img_width = 50
		self.img_length = 50

		self.model = model




	def loadImage(self, path):
		img = image.load_image(path, target_size(self.img_length, self.img_width))
		arr = image.img_to_array(img)
		return np.expand_dims(arr, axis = 0)

	def getHeatMap(self):
		pass

if __name__ == '__main__':
	g = gradCam('a')