import torch
import torchvision
import torchvision.transforms as transforms



# 3 hyperparameters.....
# total training epoch : epoch_num
# SGD learning rate    : learn_rate
# SGD momentum         : momentum

def cifar_10_classify_accu(epoch_num, learn_rate, momentum):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# normalize to -1,1 tensors...... torchvision data
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
		])


	# getting and loading data of trainSet

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		download = True, transform = transform)


	# batch_size can be hyperparameter of this DNN?
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
		shuffle = True, num_workers = 0)

	# Getting and Loading data of TestSet
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		download = True, transform = transform)

	testloader = torch.utils.data.DataLoader(testset, batch_size=4,
		shuffle = True, num_workers = 0)


	classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

	# Defining Convolution Neural Network
	# If we wanna change the constructure of neural network, this should be modified.....

	import torch.nn as nn
	import torch.nn.functional as F

	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()

			# input_channel 3, output_channel 6, kernel size 5*5 i think......
			self.conv1 = nn.Conv2d(3, 6, 5)

			# pool of square window of size = 2, stride=2
			self.pool = nn.MaxPool2d(2,2)

			# input_channel 6, output_channel 16, kernel size 5*5
			self.conv2 = nn.Conv2d(6, 16, 5)

			# fully connected layer 1
			# input 16*5*5, output 120
			self.fc1 = nn.Linear(16*5*5, 120)

			# input 120, output 84
			self.fc2 = nn.Linear(120, 84)

			# input 84, output 10
			self.fc3 = nn.Linear(84,10)

		def forward(self,x):
			# convolution layer 1 to pooling layer and activation function ReLU
			x = self.pool(F.relu(self.conv1(x)))

			# convolution layer 2 to pooling layer and activation function ReLU
			x = self.pool(F.relu(self.conv2(x)))

			x = x.view(-1, 16*5*5)

			# fully connected layer 1
			# and activaton function is ReLU
			x = F.relu(self.fc1(x))

			# fully connected layer 2
			# activation function ReLU
			x = F.relu(self.fc2(x))

			# fully connected layer 3
			# activation function is identity function
			x = self.fc3(x)

			return x


	net = Net()
	net.to(device)


	# Defining Loss Function and Optimizer
	import torch.optim as optim

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum = momentum)

	# Train the network!!

	for epoch in range(epoch_num):
		running_loss = 0.0

		# enumerate returns the index and the element if trainloader
		# parameter 0 means index starts with 0
		for i, data in enumerate(trainloader,0):

			# inputs, labels from data!
			inputs, labels = data[0].to(device), data[1].to(device)

			# making Gradient paramter to zero??
			# what this is mean??
			optimizer.zero_grad()

			# forward + backward propagation and optimizing 
			outputs = net(inputs) # forward
			loss = criterion(outputs, labels) # getting the loss
			loss.backward()
			optimizer.step()

			# printing statistics....
			running_loss += loss.item()
			# if i%2000 == 1999:
			# 	print('[%d, %5d] loss: %.3f'%
			# 		(epoch+1, i+1, running_loss/2000))
			# 	running_loss = 0.0

	# 'net' has been trained!!


	# Do we need to save the trained network??
	# PATH = './cifar_net.pth'
	# torch.save(net.state_dict(), PATH)

	# Check Accuray of the trained network!
	dataiter = iter(testloader)
	images, labels = dataiter.next()

	correct = 0
	total = 0

	with torch.no_grad():
		for data in testloader:
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)

			_, predicted = torch.max(outputs.data, 1)

			total += labels.size(0)

			correct += (predicted == labels).sum().item()

	accuracy = correct / total

	return accuracy

def main(job_id, params):
	return cifar_10_classify_accu(params['epoch_num'], params['learn_rate'], params['momentum'])




