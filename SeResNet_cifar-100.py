import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import visdom

viz = visdom.Visdom()
def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):  # SeNet Block.
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)  # The Squeeze of SeNet.
		self.fc = nn.Sequential(
				nn.Linear(channel, channel // reduction),
				nn.ReLU(inplace=True),
				nn.Linear(channel // reduction, channel),
				nn.Sigmoid()
		)  # Excitation of SeNet.

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y

class CifarSEBasicBlock(nn.Module):
	def __init__(self, inplanes, planes, stride=1, reduction=16):
		super(CifarSEBasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.se = SELayer(planes, reduction)
		if inplanes != planes:
			self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
		else:
			self.downsample = lambda x: x
		self.stride = stride

	def forward(self, x):
		residual = self.downsample(x)
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.se(out)

		out += residual
		out = self.relu(out)

		return out
class CifarSEResNet(nn.Module):
	def __init__(self, block, n_size, num_classes=10, reduction=16):
		super(CifarSEResNet, self).__init__()
		self.inplane = 16
		self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.inplane)
		self.relu = nn.ReLU(inplace=True)
		self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1, reduction=reduction)
		self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2, reduction=reduction)
		self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2, reduction=reduction)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(64, num_classes)
		self.initialize()

	def initialize(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride, reduction):
		strides = [stride] + [1] * (blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.inplane, planes, stride, reduction))
			self.inplane = planes

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

def se_resnet56(**kwargs):
	"""Constructs a ResNet-34 model.
	"""
	model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
	return model


# Data loading
to_normalized_tensor = [transforms.ToTensor(), transforms.Normalize((0.5070754, 0.48655024, 0.44091907),
																	(0.26733398, 0.25643876, 0.2761503))]
data_augmentation = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]

train_data = torchvision.datasets.CIFAR100(root='/home/cz/', train=True, download=True, transform=transforms.Compose(data_augmentation + to_normalized_tensor))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR100(root='/home/cz/', train=False, download=True, transform=transforms.Compose(data_augmentation + to_normalized_tensor))

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

SeResnet = se_resnet56(num_classes=100)
print(SeResnet)
lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(SeResnet.parameters(), lr=0.0007)

lossdata = []
accdata = []
loss = 0
accuracy = 0
for epoch in range(20):
	for i, data in enumerate(train_loader, 0):
		images, labels = data
		optimizer.zero_grad()
		outputs = SeResnet(images)
		loss = lossfunc(outputs, labels)
		loss.backward()
		optimizer.step()

		if i % 100 == 99:
			correct = 0
			test_times = 0
			accuracy = 0
			for j, (test_images, test_labels) in enumerate(test_loader, 0):
				test_output = SeResnet(test_images)
				_, prediction = torch.max(test_output.data, 1)
				correct += (prediction == test_labels).sum()
				test_times += test_labels.size(0)
				accuracy = float(correct) / float(test_times)
			print('[%d,%5d]   Loss:%.3f' % (epoch + 1, i + 1, loss), '  Accuracy %.4f' % accuracy)
	lossdata.append(loss)
	accdata.append(accuracy)


torch.save(SeResnet.state_dict(), '/home/cz/SeResnet.pkl')
lossdata = torch.Tensor(lossdata)
accdata = torch.Tensor(accdata)
x = torch.range(1, 20)

viz.line(lossdata, x, opts=dict(title='loss'))
viz.line(accdata, x, opts=dict(title='acc'))
