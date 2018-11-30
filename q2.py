import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def solve():
	images = np.load('datasets/random/random_imgs.npy')
	labels = np.load('datasets/random/random_labs.npy')
	labels = np.reshape(labels,(np.shape(labels)[0], 1))

	input_images = np.reshape(images, (np.shape(images)[0], np.shape(images)[1] * np.shape(images)[2]))
	num_input_images = np.shape(input_images)[0]

	print(labels.shape)
	print(input_images.shape)
	print(num_input_images)

solve()