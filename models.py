
import torch 
from torchvision import models
import math
import numpy as np


def model_norm(model_1, model_2):
	squared_sum = 0
	for name, layer in model_1.named_parameters():
	#	print(torch.mean(layer.data), torch.mean(model_2.state_dict()[name].data))
		squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
	return math.sqrt(squared_sum)
def encrypt_vector(public_key, x):
	return [public_key.encrypt(i) for i in x]


def encrypt_matrix(public_key, x):
	ret = []
	for r in x:
		ret.append(encrypt_vector(public_key, r))
	return ret


def decrypt_vector(private_key, x):
	return [private_key.decrypt(i) for i in x]


def decrypt_matrix(private_key, x):
	ret = []
	for r in x:
		ret.append(decrypt_vector(private_key, r))
	return ret


class LR_Model(object):
	def __init__(self, public_key, w_size=None, w=None, encrypted=False):
		"""
		w_size: 权重参数数量
		w: 是否直接传递已有权重，w和w_size只需要传递一个即可
		encrypted: 是明文还是加密的形式
		"""
		self.public_key = public_key
		if w is not None:
			self.weights = w
		else:
			limit = -1.0 / w_size
			self.weights = np.random.uniform(-0.5, 0.5, (w_size,))

		if encrypted == False:
			self.encrypt_weights = encrypt_vector(public_key, self.weights)
		else:
			self.encrypt_weights = self.weights

	def set_encrypt_weights(self, w):
		for id, e in enumerate(w):
			self.encrypt_weights[id] = e

	def set_raw_weights(self, w):
		for id, e in enumerate(w):
			self.weights[id] = e

def get_model(name="vgg16", pretrained=True):
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)	
	elif name == "densenet121":
		model = models.densenet121(pretrained=pretrained)		
	elif name == "alexnet":
		model = models.alexnet(pretrained=pretrained)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)
	elif name == "vgg19":
		model = models.vgg19(pretrained=pretrained)
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=pretrained)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
		
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 