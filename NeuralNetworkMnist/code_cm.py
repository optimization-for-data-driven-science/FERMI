''' Required modules ''' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn,optim
import torch
import sys 
from torchsummary import summary

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class ColouredData(Dataset):
	''' Class to create a colored version of the dataset '''
	def __init__(self,trainx,trainy,testx,testy):
		self.numY = len(torch.unique(torch.Tensor(trainy)))
		self.colors = torch.Tensor(self.numY,3,1,1).uniform_(0,1)

		colored_train, self.trains = self.color(trainx,trainy,True)
		colored_test, self.tests = self.color(testx,testy,False)

		self.trainx = colored_train
		self.testx = colored_test
		self.trainy = torch.Tensor(trainy)
		self.testy = torch.Tensor(testy)

		self.trainy = self.trainy.type(torch.int64)
		self.testy = self.testy.type(torch.int64)
		self.trains = self.trains.type(torch.int64)
		self.tests = self.tests.type(torch.int64)

		self.trainx.to(dev)
		self.trainy.to(dev)
		self.testx.to(dev)
		self.testy.to(dev)
		self.trains.to(dev)
		self.tests.to(dev)

	def color(self,items,labels,ch):
		new_images = []
		colors=self.colors

		if ch == False:
			colors = torch.ones(self.numY,3,1,1)

		for i in range(items.shape[0]):
			img = colors[labels[i]]*items[i]
			# img = img.permute(1,2,0)
			new_images.append(img)
		return torch.stack(new_images), torch.Tensor(labels)

	def to_linear(self):
		''' If the model is MLP, we need to flatten all the data points'''
		self.trainx = torch.flatten(self.trainx,1,3)
		self.testx = torch.flatten(self.testx,1,3)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return self.out(x)

class LeNet(nn.Module):
    def __init__(self, in_channels, out_dims):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.out   = nn.Linear(84, out_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


def get_model(model_name, lr):

    if model_name == 'lenet':
    	model = LeNet(3, numY).to(dev)
    else:
    	model = MLP(784 * 3, [300, 100], numY).to(dev)

    return model, optim.Adam(model.parameters(), lr=lr)


def bitmask(x):
	bit = 1
	mask = 0
	for i in x:
		mask += i*bit
		bit *= 2
	return mask

class Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(X.shape[1], numY)
        self.lin.weight.data.fill_(0.0)
        self.lin.bias.data.fill_(0)

    def forward(self, xb):
        return self.lin(xb)

def get_logistic(lr):
    model = Logistic().to(dev)
    return model, optim.SGD(model.parameters(), lr=lr)

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

soft = torch.nn.Softmax(dim = 1)

def loss_func_mod1(preds,Y,Lam):
	pred = soft(preds).to(dev)
	Py = torch.mean(pred,0).to(dev)
	Q = torch.zeros((numS,numY)).to(dev)

	for s in range(numS):
	    Q[s] = PS[s] * torch.mean(pred[S==s],0) / torch.sqrt(Py*PS[s])	

	return F.cross_entropy(preds,Y,reduction='mean') + Lam * torch.sum(Q**2)

def loss_func_renyi_mod1(preds,Y,Lam):
	''' If numS <= numY ''' 
	pred = soft(preds).to(dev)
	Py = torch.mean(pred,0).to(dev)
	Q = torch.zeros((numS,numY)).to(dev)
	# tmp = 0
	for s in range(numS):
	    Q[s] = PS[s] * torch.mean(pred[S==s],0) / torch.sqrt(Py*PS[s])	
	e, v = torch.symeig(torch.matmul(Q,torch.transpose(Q,0,1)),eigenvectors=True)

	return F.cross_entropy(preds,Y,reduction='mean') + Lam * e[numS - 2]	

def loss_func_renyi_mod2(preds,Y,Lam):
	''' If numS > numY ''' 
	pred = soft(preds).to(dev)
	Py = torch.mean(pred,0).to(dev)
	Q = torch.zeros((numS,numY)).to(dev)

	for s in range(numS):
	    Q[s] = PS[s] * torch.mean(pred[S==s],0) / torch.sqrt(Py*PS[s])	
	e, v = torch.symeig(torch.matmul(torch.transpose(Q,0,1),Q),eigenvectors=True)

	return F.cross_entropy(preds,Y,reduction='mean') + Lam * e[numY - 2]	

def loss_func_log_mod1(preds,Y,Lam):
	pred = soft(preds).to(dev)
	Py = torch.mean(pred,0).to(dev)
	Q = torch.zeros((numS,numY)).to(dev)

	for s in range(numS):
	    Q[s] = PS[s] * torch.mean(pred[S==s],0) / torch.sqrt(Py*PS[s])	

	return F.cross_entropy(preds,Y,reduction='mean') + Lam * torch.log(torch.sum(Q**2))

def stochastic_loss_mod1(preds,Yen,Sen,Lam):
	pred = soft(preds).to(dev)
	Py = torch.mean(pred,0).to(dev)
	Q = torch.zeros((numS,numY))
	# tmp = 0
	for s in range(numS):
		tmp = pred[Sen==s]
		if len(tmp)>0:
			Q[s] = PS[s] * torch.mean(tmp,0) / torch.sqrt(Py*PS[s])
		else:
			Q[s] = PS[s] * torch.zeros(numY) / torch.sqrt(Py*PS[s])

	return F.cross_entropy(preds,Yen,reduction='mean') + Lam * torch.sum(Q**2)

def stochastic_loss_mod2(preds,Yen,Sen,Lam):
	pred = soft(preds).to(dev)

	PyhatS = torch.zeros((numS,numY))

	Pyhat = torch.zeros((numY,numY))

	sumY = torch.mean(pred,0)
	for i in range(numY):
		Pyhat[i][i] = 1/sumY[i]

	Ps = torch.zeros((numS,numS))
	for i in range(numS):
		Ps[i][i] = 1/PS[i]

	for s in range(numS):
		tmp = pred[Sen==s]
		if len(tmp)>0:
			PyhatS[s] = PS[s] * torch.mean(tmp,0) 
		else:
			PyhatS[s] = PS[s] * torch.zeros(numY) 

	R = Ps
	R = torch.matmul(R,PyhatS)
	R = torch.matmul(R,Pyhat)
	R = torch.matmul(R,torch.transpose(PyhatS,0,1))

	return F.cross_entropy(preds,Yen,reduction='mean') + Lam * torch.trace(R)

def stochastic_loss1_mod3(preds,Yen,Sen,Lam):
	pred = soft(preds).to(dev)
	Py = torch.mean(pred,0).to(dev)
	Q = torch.zeros((numS,numY))
	# tmp = 0
	for s in range(numS):
		tmp = pred[Sen==s]
		if len(tmp)>0:
			Q[s] = PS[s] * torch.mean(tmp,0) / torch.sqrt(Py*PS[s])
		else:
			Q[s] = PS[s] * torch.zeros(numY) / torch.sqrt(Py*PS[s])

	return F.cross_entropy(preds,Yen,reduction='mean') + Lam * torch.log(torch.sum(Q**2))

def show_samples(inds):
	for i in inds:
		plt.imshow(data.trainx[i])
		plt.show()
	for i in inds:
		plt.imshow(data.testx[i])
		plt.show()

def fair(Lam,num_iterations,eta,LOSS,model_name):
	print('\n\nLam:{0}, T:{1}, Eta:{2}'.format(Lam,num_iterations,eta))
	model, opt = get_model(model_name,eta)

	for epoch in range(num_iterations):
		IND = torch.from_numpy(np.random.choice(X.shape[0], size=(512,), replace=False)) 
		IND.to(dev)
		subX, subY, subS = X[IND],Y[IND],S[IND]
		subX.to(dev)
		subY.to(dev)
		subS.to(dev)
		# print(subX.shape)
		pred = model(subX)
		loss = LOSS(pred, subY, subS, Lam)
		loss.backward()
		opt.step()
		opt.zero_grad()
		if epoch%5 == 0:
			print(epoch, loss)

	preds = model(X)
	preds = soft(preds)

	# Calculating train Accuracy
	ac1 = accuracy(preds, Y)
	print('\nAcc:',ac1)

	# Making final predictions
	preds = model(data.testx)
	preds = soft(preds)

	# Calculating Test Accuracy
	ac = accuracy(preds, data.testy)

	# Calculating Demographic parity
	dp = torch.zeros(1)
	pyhat = torch.mean(preds,0).to(dev)
	for i in range(numS):
		tmp = torch.mean(preds[data.tests==i],0)
		for j in range(numY):
			dp = torch.max(dp, torch.abs(tmp[j]-pyhat[j]))

	return ac, dp

def create_init():
	trainx = np.load('trainx.npy')
	testx = np.load('testx.npy')
	trainy = np.load('trainy.npy')
	testy = np.load('testy.npy')

	# ind = int(sys.argv[1]) 
	print(trainx.shape)
	print(testx.shape)

	data = ColouredData(trainx,trainy,testx,testy)
	# data.to(dev)
	numY = data.numY
	numS = numY

	show_samples(range(3))

	PS = torch.zeros(numS).to(dev)

	for s in data.trains:
		PS[s.item()]+=1

	PS /= len(data.trains)

	model_name = "lenet"


	if model_name == "mlp":
		data.to_linear()

	print(data.trainx.shape)
	print(data.testx.shape)
	print(data.trainy.shape)


	S = data.trains
	X = data.trainx
	Y = data.trainy



