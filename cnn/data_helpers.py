import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
from TextDatasetFileParser import Instance

import os, sys
sys.path.insert(0, os.path.abspath(".."))

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

def load_data_and_labels_from_file(dataAddress):
	#Loads data from csv,format is two text columns followed by location and class
	d = pd.read_csv(dataAddress, sep='\t')
	d = d[d['class'] != -1]
	# Load data from files
	# Split by words
	x = d['text']
	x = [clean_str(sent) for sent in x]
	# Generate labels
	class_labels = d['class']
	labels = [[0, 1] if l == 1 else [1, 0] for l in class_labels]
	y = np.concatenate([labels], 0)
	return [x, y]

def load_data_and_labels_from_instances(instances, classes):
	x = []
	labels = []
	for i in instances:
		if i.class_value != -1:
			labels.append(i.class_value)
			x.append(i.text)

	def getLabelVector(label, class_names):
		vector = [0] * len(class_names)
		for i in range(len(class_names)):
			if label == class_names[i]:
				vector[i] = 1
			return vector
		return vector

	# Split by words
	x = [clean_str(sent) for sent in x]
	# Generate labels
	labels = [getLabelVector(l, classes) for l in labels]
	y = np.concatenate([labels], 0) #why is this?
	return [x, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(len(data)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]
