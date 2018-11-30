import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def plot():
	weights = np.arange(-5, 5, 0.01)
	bias = np.arange(-5, 5, 0.01)

	W, B = np.meshgrid(weights, bias)

	power = -1.0 * (W + B)
	Ypred = 1.0 + np.exp(power)
	Ypred = 1.0 / Ypred
	Yexp = np.tile(0.5, Ypred.shape)

	Ydiff = Ypred - Yexp

	L2_Gradient = 2.0 * W * Ydiff * (1.0 - Ypred) * Ypred

	L2_Loss = Ydiff * Ydiff
	CE_Loss = -1.0 * (Yexp * np.log(Ypred) + (1.0 - Yexp) * np.log(1.0 - Ypred))

	# CE_Gradient = W * (Yexp * (Ypred - 1) + Ypred * (Yexp - 1))
	CE_Gradient = ((-1.0 * Yexp / Ypred) + ((Yexp - 1) / (1 - Ypred))) * W * (1 - Ypred) * Ypred

	fig = plt.figure(figsize=plt.figaspect(0.2))

	ax = fig.add_subplot(1, 5, 1, projection='3d')
	ax.plot_surface(W, B, Ypred, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_title('Sigmoid Plot')

	ax = fig.add_subplot(1, 5, 2, projection='3d')
	ax.plot_surface(W, B, L2_Loss, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_title('L2 Plot')

	ax = fig.add_subplot(1, 5, 3, projection='3d')
	ax.plot_surface(W, B, L2_Gradient, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_title('L2 Gradient Plot')

	ax = fig.add_subplot(1, 5, 4, projection='3d')
	ax.plot_surface(W, B, CE_Loss, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_title('CE Plot')

	ax = fig.add_subplot(1, 5, 5, projection='3d')
	ax.plot_surface(W, B, CE_Gradient, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_title('CE Gradient Plot')

	fig.tight_layout()
	plt.show()


plot()