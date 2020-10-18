import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.utils import shuffle
from torch.autograd import Variable
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# 2 hyperparameters
# Total training epoch : epoch_num
# Adam Optimizer learning rate : learn_rate

def boston_housing_rmse(epoch_num, learn_rate):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Loading data!
	boston = load_boston()

	# Boston housing is a dictionary based dataset
	boston_df = pd.DataFrame(boston['data'])

	# Define the y data
	y = boston_df["PRICE"]

	# Defin the X data
	X = boston_df.iloc[:,0:13]

	# Split the data into a training set and a test set
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

	batch_size = 50
	num_epochs = epoch_num
	learning_rate = learn_rate
	size_hidden = 100 # What is the meaning of size_hidden

	# Calculate some other hyperparameters based on data
	batch_no = len(X_train) // batch_size
	cols = X_train.shape[1]
	n_output = 1

	# Creating the Neural Network model
	# One hidden layer, Two layer Neural Network
	class Net(torch.nn.Module):
		def __init__(self, n_feature, size_hidden, n_output):
			super(Net, self).__init__()
			self.hidden = torch.nn.Linear(cols, size_hidden) # hidden layer
			self.predict = torch.nn.Linear(size_hidden, n_output)

		def forward(self, x):
			x = F.relu(self.hidden(x))
			x = self.predict(x)
			return x

	net = Net(cols, size_hidden, n_output)

	optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

	criterion = torch.nn.MSELoss(size_average=False) # # this is for regression mean squared loss


	# Training the model!!

	for epoch in range(num_epochs):
		# Shuffle just mixes up the dataset between epochs
		X_train, y_train = shuffle(X_train, y_train)

		# Mini batching learning
		for i in range(batch_no):
			start = i * batch_size
			end = start + batch_size
			inputs = Variable(torch.FloatTensor(X_train[start:end]))
			labels = Variable(torch.FloatTensor(y_train[start:end]))

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)

			loss = criterion(outputs, torch.unsqueeze(labels,dim=1))
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

		running_loss = 0.0
	# for loop end

	def calculate_rmse(x,y=[]):

		X = Variable(torch.FloatTensor(x))
		result = net(X)
		result = result.data[:,0].numpy()

		rmse = mean_squared_error(result,y)**0.5

		return rmse

	def calculate_r2(x,y=[]):
		X = Variable(torch.FloatTensor(x))
		result = net(X)
		result = result.data[:,0].numpy()

		r2 = r2_score(result,y)

		return r2


	final_rmse = calculate_rmse(X_test, y_test)

	return final_rmse

def main(job_id, params):
	# epoch_num, learn_rate
	return boston_housing_rmse(params['epch_num'], params['learn_rate'])


