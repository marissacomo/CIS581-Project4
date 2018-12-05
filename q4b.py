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

	def __init__(self, activationType):
		super(Net, self).__init__()

		if activationType == 'sigmoid':
			self.activation1 = nn.Sigmoid()
			self.activation2 = nn.Sigmoid()
			self.activation3 = nn.Sigmoid()
		elif activationType == 'relu':
			self.activation1 = nn.ReLU()
			self.activation2 = nn.ReLU()
			self.activation3 = nn.Sigmoid()


		self.conv1 = nn.Conv2d(1, 16, 7)
		self.conv2 = nn.Conv2d(16, 8, 7)

		self.fc1 = nn.Linear(8 * (4 * 4), 1, bias = True)
		self.fc2 = nn.Linear(8 * (4 * 4), 1, bias = True)

	def forward(self, x):
		# print('0 - X Size: ', x.size())
		x = self.conv1(x)
		# print('1 - X Size: ', x.size())
		x = self.activation1(x)
		# print('2 - X Size: ', x.size())	
		x = self.conv2(x)
		# print('3 - X Size: ', x.size())	
		x = self.activation2(x)
		y = x.clone()
		# print('4 - X Size: ', x.size())	
		# print('4 - X View: ', x.view(x.size()[0], -1))	
		# print('4 - X View Size: ', x.view(x.size()[0], -1).size())	

		x = self.fc1(x.view(x.size()[0], -1))
		# print('5 - X Size: ', x.size())	
		x = self.activation3(x)
		# print('6 - X Size: ', x.size())	

		y = self.fc2(y.view(y.size()[0], -1))

		return x, y


	def setWeights(self, mean, std, constant_bias):
		norm_d = normal.Normal(mean, std)

		self.conv1.weight = torch.nn.Parameter(torch.clamp(norm_d.sample(self.conv1.weight.size()), min=-2*std, max=2*std))
		self.conv2.weight = torch.nn.Parameter(torch.clamp(norm_d.sample(self.conv2.weight.size()), min=-2*std, max=2*std))
		self.fc1.weight = torch.nn.Parameter(torch.clamp(norm_d.sample(self.fc1.weight.size()), min=-2*std, max=2*std))
		self.fc2.weight = torch.nn.Parameter(torch.clamp(norm_d.sample(self.fc2.weight.size()), min=-2*std, max=2*std))
		torch.nn.init.constant_(self.fc1.bias, constant_bias)
		torch.nn.init.constant_(self.fc2.bias, constant_bias)
		torch.nn.init.constant_(self.conv1.bias, constant_bias)
		torch.nn.init.constant_(self.conv2.bias, constant_bias)

# Referenced from: FaceLandmarksDataset in tutorial
class ImageDataset(Dataset):

	def __init__(self, rootDir, imageBlob, lineBlob, labelBlob):
		images = np.load(rootDir + imageBlob)
		self.labels = np.load(rootDir + labelBlob)
		self.labels = np.reshape(self.labels,(np.shape(self.labels)[0], 1))

		self.lines = np.load(rootDir + lineBlob)
		self.lines = np.reshape(self.lines,(np.shape(self.lines)[0], 1))

		self.input_images = np.reshape(images, (np.shape(images)[0], np.shape(images)[1] * np.shape(images)[2]))
		self.num_input_images = np.shape(self.input_images)[0]

		print('Data Labels:', self.labels.shape)
		print('Input Images Shape:', self.input_images.shape)
		print('Num Input Images:', self.num_input_images)

	def __len__(self):
		return self.num_input_images

	def __getitem__(self, idx):
		sample = {'image': self.input_images[idx], 'label': self.labels[idx], 'lines': self.lines[idx] }
		return sample

def solve():
	learningRate = 0.1
	learningRate1 = 0.1
	learningRate2 = 0.001

	batchSize = 64
	maxIterations = 10000

	# sigmoid, relu
	activationType = 'relu'
	# l2, ce
	lossType1 = 'ce'
	lossType2 = 'l2'

	rootDir = 'datasets/detection/'
	imageBlob = 'detection_imgs.npy'
	lineBlob = 'detection_width.npy'
	labelBlob = 'detection_labs.npy'

	imageSet = ImageDataset(rootDir, imageBlob, lineBlob, labelBlob)
	trainDataLoader = DataLoader(imageSet, batch_size=batchSize)

	testSet = ImageDataset(rootDir, imageBlob, lineBlob, labelBlob)
	testDataLoader = DataLoader(testSet, batch_size=1)

	net = Net(activationType)

	mean = 0
	std = 0.1
	constantBias = 0.1

	net.setWeights(mean, std, constantBias)

	if lossType1 == 'l2':
		lossFunction1 = nn.MSELoss()
	elif lossType1 == 'ce':
		lossFunction1 = nn.BCELoss()

	if lossType2 == 'l2':
		lossFunction2 = nn.MSELoss()
	elif lossType2 == 'ce':
		lossFunction2 = nn.BCELoss()

	optimizer = optim.SGD(net.parameters(), lr = learningRate)

	accuracyList = []
	regressionAccuracyList = []
	lossList = []
	l2List = []
	ceList = []

	for itr in range(maxIterations):
		for i, batch in enumerate(trainDataLoader):

			batchX = batch['image']
			batchY = batch['label']
			linesY = batch['lines']

			batchY = batchY.float()
			linesY = linesY.float()

			# Long tensor for CE loss
			if lossType1 == 'ce':
				batchY.type(torch.int64)


			batchX = batchX.reshape(batchSize, 1, 16, 16)

			out1, out2 = net(batchX.float())
			loss1 = lossFunction1(out1, batchY)
			loss2 = lossFunction2(out2, linesY)

			# totalLoss = loss1 + loss2
			totalLoss = (learningRate1 / learningRate) * loss1 + (learningRate2 / learningRate) * loss2

			optimizer.zero_grad()
			totalLoss.backward()
			optimizer.step()

		iterationLoss = 0
		l2Loss = 0
		ceLoss = 0
		iterationCorrect = 0
		regressionCorrect = 0
		for i, batch in enumerate(testDataLoader):
			batchX = batch['image']
			batchY = batch['label']
			linesY = batch['lines']

			batchX = batchX.reshape(1, 1, 16, 16)

			out1, out2 = net(batchX.float())
			loss1 = lossFunction1(out1, batchY.float())
			loss2 = lossFunction2(out2, linesY.float())

			totalLoss = (learningRate1 / learningRate) * loss1 + (learningRate2 / learningRate) * loss2

			iterationLoss = iterationLoss + totalLoss
			ceLoss = ceLoss + loss1
			l2Loss = l2Loss + loss2

			outNumpy = out1.data.numpy()
			lineNumpy = out2.data.numpy()

			outNumpy[outNumpy >= 0.5] = 1
			outNumpy[outNumpy < 0.5] = 0

			if np.array_equal(outNumpy, batchY.numpy()):
				iterationCorrect += 1

			if (np.abs(lineNumpy[0,0] - linesY[0, 0]) <= 0.5):
				regressionCorrect += 1


		accuracy = iterationCorrect / imageSet.num_input_images
		regressionAccuracy = regressionCorrect / imageSet.num_input_images

		print('--------------------------');
		print('Correct Guesses: ' + str(iterationCorrect) + '/' + str(imageSet.num_input_images))
		print('Iteration: ' + str(itr))
		print('Accuracy > ' + str(accuracy * 100) + '%')
		print('Regression > ' + str(regressionAccuracy * 100) + '%')
		print('Loss: ' + str(iterationLoss))
		print('--------------------------')

		accuracyList.append(accuracy)
		regressionAccuracyList.append(regressionAccuracy)
		lossList.append(iterationLoss)
		l2List.append(l2Loss)
		ceList.append(ceLoss)

		if accuracy == 1:
			print('Finished')
			break

	fig = plt.figure(figsize=plt.figaspect(0.25))

	ax = fig.add_subplot(1, 4, 1)
	ax.plot(np.arange(len(accuracyList)), accuracyList)
	ax.set_title('Accuracy')

	ax = fig.add_subplot(1, 4, 2)
	ax.plot(np.arange(len(regressionAccuracyList)), regressionAccuracyList)
	ax.set_title('Regression Accuracy')

	ax = fig.add_subplot(1, 4, 3)
	ax.plot(np.arange(len(ceList)), ceList)
	ax.set_title('CE Loss')

	ax = fig.add_subplot(1, 4, 4)
	ax.plot(np.arange(len(l2List)), l2List)
	ax.set_title('L2 Loss')

	plt.show()


solve()