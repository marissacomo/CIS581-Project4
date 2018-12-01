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

class XORNetwork(nn.Module):

	def __init__(self):
		super(XORNetwork, self).__init__()

		self.fc1 = nn.Linear(2, 2, bias = True)
		self.fc2 = nn.Linear(2, 1, bias = True)
		self.activation1 = nn.Tanh()
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
class ArrayDataset(Dataset):

	def __init__(self, inputValues, labelValues):
		self.labels = np.copy(labelValues)
		self.values = np.copy(inputValues)
		self.num_values = np.shape(self.values)[0]

		print('Data Labels:', self.labels.shape)
		print('Input Values Shape:', self.values.shape)
		print('Num Input Values:', self.num_values)

	def __len__(self):
		return self.num_values

	def __getitem__(self, idx):
		sample = {'values': self.values[idx], 'label': self.labels[idx] }
		return sample

def solve():
	learningRate = 0.1
	batchSize = 4
	maxIterations = 10000

	# l2, ce
	lossType = 'ce'

	inputValues = np.array([
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]
	]);


	labelValues = np.array([
		[0],
		[1],
		[1],
		[0]
	]);

	xorData = ArrayDataset(inputValues, labelValues)
	trainDataLoader = DataLoader(xorData, batch_size=batchSize)

	testSet = ArrayDataset(inputValues, labelValues)
	testDataLoader = DataLoader(testSet, batch_size=1)

	net = XORNetwork()

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

			batchX = batch['values']
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
			batchX = batch['values']
			batchY = batch['label']

			out = net(batchX.float())
			loss = lossFunction(out, batchY.float())

			iterationLoss = iterationLoss + loss

			outNumpy = out.data.numpy()

			outNumpy[outNumpy >= 0.5] = 1
			outNumpy[outNumpy < 0.5] = 0

			if np.array_equal(outNumpy, batchY.numpy()):
				iterationCorrect += 1

		accuracy = iterationCorrect / xorData.num_values
		print('--------------------------');
		print('Correct Guesses: ' + str(iterationCorrect) + '/' + str(xorData.num_values))
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