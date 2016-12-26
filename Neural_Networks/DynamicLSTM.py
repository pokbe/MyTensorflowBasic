import tensorflow as tf
import numpy as np
import random

class SequenceDataGeneration(object):
	def __init__(self, num_examples, max_seq_len, min_seq_len, max_value):
		self.features = []
		self.labels = []
		self.lengths = []
		for example in range(num_examples):
			example_length = random.randint(min_seq_len,max_seq_len)  #Return a random integer N such that a <= N <= b
			self.lengths.append(example_length)
			if random.random()<0.5 : #Return the next random floating point number in the range [0.0, 1.0)
				start_rand = random.randint(0, max_value-example_length)
				sequence = [x/max_value for x in range(start_rand, start_rand+example_length)]
				sequence += [0.0 for _ in range(max_seq_len - example_length)]
				self.features.append(sequence)
				self.labels.append([1.0, 0.0])
			else:
				sequence = [ random.randint(0, max_value)/max_value for _ in range(example_length) ]
				sequence += [0.0 for _ in range(max_seq_len - example_length)]
				self.features.append(sequence)
				self.labels.append([0.0, 1.0])
		self.batch_index = 0
	def next_batch(self, batch_size):
		if self.batch_index = num_examples-1:
			self.batch_index = 0
		batch_features = self.features[self.batch_index, min(self.batch_index+batch_size , num_examples-1)]
		batch_labels = self.labels[self.batch_index, min(self.batch_index+batch_size , num_examples-1)]
		batch_lengths = self.lengths[self.batch_index, min(self.batch_index+batch_size , num_examples-1)]
		return batch_features, batch_labels, batch_lengths