import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import normal
import torch.optim as optim

device = None

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU available. {}'.format(torch.cuda.get_device_name(0)))
    print('-'*100)
else:
    device = torch.device('cpu')
    print('GPU unavailable')
    print('-'*100)

class Net(nn.Module):

	def __init__(self, inputCount, output1Count, output2Count, activationType):
		super(Net, self).__init__()

		if activationType == 'sigmoid':
			self.activation1 = nn.Sigmoid()
		elif activationType == 'relu':
			self.activation1 = nn.ReLU()

		self.fc1 = nn.Linear(inputCount, output1Count, bias = True)
		self.fc2 = nn.Linear(output1Count, output2Count, bias = True)
		self.activation2 = nn.Sigmoid()

	def forward(self, x):
		x = self.fc1(x)
		x = self.activation1(x)
		x = self.fc2(x)
		x = self.activation2(x)

		return x

	def setWeights(self, mean, std, constant_bias):
		norm_d = normal.Normal(mean, std)

		self.fc1.weight = torch.nn.Parameter(torch.clamp(norm_d.sample(self.fc1.weight.size()), min=-2*std, max=2*std))
		self.fc2.weight = torch.nn.Parameter(torch.clamp(norm_d.sample(self.fc2.weight.size()), min=-2*std, max=2*std))
		torch.nn.init.constant_(self.fc1.bias, constant_bias)
		torch.nn.init.constant_(self.fc2.bias, constant_bias)


# Referenced from: FaceLandmarksDataset in tutorial
class ImageDataset(Dataset):

	def __init__(self, rootDir, imageBlob, labelBlob):
		images = np.load(rootDir + imageBlob)
		self.labels = np.load(rootDir + labelBlob)
		self.labels = np.reshape(self.labels,(np.shape(self.labels)[0], 1))

		self.input_images = np.reshape(images, (np.shape(images)[0], np.shape(images)[1] * np.shape(images)[2]))
		self.num_input_images = np.shape(self.input_images)[0]

		print('Data Labels:', self.labels.shape)
		print('Input Images Shape:', self.input_images.shape)
		print('Num Input Images:', self.num_input_images)

	def __len__(self):
		return self.num_input_images

	def __getitem__(self, idx):
		sample = {'image': self.input_images[idx], 'label': self.labels[idx] }
		return sample

def solve():
	learningRate = 0.1
	batchSize = 64
	maxIterations = 10000

	# sigmoid, relu
	activationType = 'relu'
	# l2, ce
	lossType = 'ce'

	rootDir = 'datasets/random/'
	imageBlob = 'random_imgs.npy'
	labelBlob = 'random_labs.npy'

	imageSet = ImageDataset(rootDir, imageBlob, labelBlob)
	trainDataLoader = DataLoader(imageSet, batch_size=batchSize)

	testSet = ImageDataset(rootDir, imageBlob, labelBlob)
	testDataLoader = DataLoader(testSet, batch_size=1)

	inputCount = 16
	output1Count = 4
	output2Count = 1

	net = Net(inputCount, output1Count, output2Count, activationType)

	mean = 0
	std = 0.1
	constantBias = 0.1

	net.setWeights(mean, std, constantBias)

	if lossType == 'l2':
		lossFunction = nn.MSELoss()
	elif lossType == 'ce':
		lossFunction = nn.BCELoss()

	optimizer = optim.SGD(net.parameters(), lr = learningRate)

	accuracyList = []
	lossList = []

	for itr in range(maxIterations):
		for i, batch in enumerate(trainDataLoader):

			batchX = batch['image']
			batchY = batch['label']

			batchY = batchY.float()

			# Long tensor for CE loss
			if lossType == 'ce':
				batchY.type(torch.int64)

			out = net(batchX.float())
			loss = lossFunction(out, batchY)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		iterationLoss = 0
		iterationCorrect = 0
		for i, batch in enumerate(testDataLoader):
			batchX = batch['image']
			batchY = batch['label']

			out = net(batchX.float())
			loss = lossFunction(out, batchY.float())

			iterationLoss = iterationLoss + loss

			outNumpy = out.data.numpy()

			outNumpy[outNumpy >= 0.5] = 1
			outNumpy[outNumpy < 0.5] = 0

			if np.array_equal(outNumpy, batchY.numpy()):
				iterationCorrect += 1

		accuracy = iterationCorrect / imageSet.num_input_images
		print('--------------------------');
		print('Correct Guesses: ' + str(iterationCorrect) + '/' + str(imageSet.num_input_images))
		print('Iteration: ' + str(itr) + ' > ' + str(accuracy * 100) + '%')
		print('Loss: ' + str(iterationLoss))
		print('--------------------------')

		accuracyList.append(accuracy)
		lossList.append(iterationLoss)

		if accuracy == 1:
			print('Finished')
			break

	fig = plt.figure(figsize=plt.figaspect(0.5))

	ax = fig.add_subplot(1, 2, 1)
	ax.plot(np.arange(len(accuracyList)), accuracyList)
	ax.set_title('Accuracy')

	ax = fig.add_subplot(1, 2, 2)
	ax.plot(np.arange(len(lossList)), lossList)
	ax.set_title('Loss')

	plt.show()


solve()