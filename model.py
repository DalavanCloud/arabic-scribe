import numpy as np
import numpy.matlib
import math
import random
import os
import xml.etree.ElementTree as ET

import tensorflow as tf
from utils import *
from realModel import Real_Model
class Model():
	def __init__(self, args, logger):
		self.logger = logger

		# ----- transfer some of the args params over to the model

		# model params
		self.rnn_size = args.rnn_size
		self.train = args.train
		self.nmixtures = args.nmixtures
		self.kmixtures = args.kmixtures
		self.batch_size = args.batch_size if self.train else 1 # training/sampling specific
		self.tsteps = args.tsteps if self.train else 1 # training/sampling specific
		self.alphabet = args.alphabet
		# training params
		self.dropout = args.dropout
		self.grad_clip = args.grad_clip
		# misc
		self.tsteps_per_ascii = args.tsteps_per_ascii
		self.data_dir = args.data_dir
		# Creates an initializer for the model variables
		self.graves_initializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
		self.window_b_initializer = tf.truncated_normal_initializer(mean=-3.0, stddev=.25, seed=None, dtype=tf.float32) # hacky initialization

		self.logger.write('\tusing alphabet{}'.format(self.alphabet))
		# UNK Token means Unknown token, for all the words not in the vocabularly, example names
		self.char_vec_len = len(self.alphabet) + 1 #plus one for <UNK> token # Alphabets small 26 + Alphabet caps 26 + space + UKN Token
		self.ascii_steps = args.tsteps/args.tsteps_per_ascii


		# Distribution
		self.ps_hosts = args.ps_hosts.split(",")
		self.worker_hosts = args.worker_hosts.split(",")
		self.job_name = args.job_name
		self.task_index = args.task_index

		cluster = args.cluster
		server = args.server

		# with tf.device("/job:worker/task:0/gpu:0"):
		with tf.device("/job:worker/task:0/gpu:0"):
			with tf.variable_scope('worker',reuse=False):
				self.worker_model = Real_Model(args, logger)
			worker_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='worker')
			worker_vars_grad = worker_vars[len(worker_vars)/2:]
			worker_gradients = tf.gradients(self.worker_model.cost, worker_vars_grad)
		with tf.variable_scope('master',reuse=False):
			self.ps_model = Real_Model(args, logger)
		ps_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='master')
		ps_vars_grad = ps_vars[0 : len(ps_vars)/2]
		ps_gradients = tf.gradients(self.ps_model.cost, ps_vars_grad)
		# ----- bring together all variables and prepare for training
		
		self.learning_rate = tf.Variable(0.0, trainable=False)
		self.decay = tf.Variable(0.0, trainable=False)
		self.momentum = tf.Variable(0.0, trainable=False)
			
		
		gradients = ps_gradients + worker_gradients

		grads, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
		if args.optimizer == 'adam':
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		elif args.optimizer == 'rmsprop':
			self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay, momentum=self.momentum)
		else:
			raise ValueError("Optimizer type not recognized")
		self.train_op = self.optimizer.apply_gradients(zip(grads, ps_vars))

		with tf.device("/job:worker/task:0/gpu:0"):
			gradients2 = ps_gradients + worker_gradients
			grads2, _ = tf.clip_by_global_norm(gradients2, self.grad_clip)
			if args.optimizer == 'adam':
				self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			elif args.optimizer == 'rmsprop':
				self.optimizer2 = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay, momentum=self.momentum)
			else:
				raise ValueError("Optimizer type not recognized")
			self.train_op2 = self.optimizer2.apply_gradients(zip(grads2, worker_vars))
		# ----- some TensorFlow I/O
		# Uncomment the following line to know the device used by each operation (GPU or CPU for debugging)
		# self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
		self.assign_decay = tf.assign(self.decay, args.decay )
		self.assign_momentum = tf.assign(self.momentum, args.momentum )
		self.assign_learning_rate = tf.assign(self.learning_rate, args.learning_rate)
		self.saver = tf.train.Saver(tf.global_variables())
		# sv = tf.train.Supervisor(is_chief=(self.task_index == 0), init_op=tf.global_variables_initializer())
		config = tf.ConfigProto(allow_soft_placement = True)
		# self.sess = sv.prepare_or_wait_for_session(server.target,config=config)
		self.sess = tf.InteractiveSession(server.target, config=config)
		# self.sess = tf.InteractiveSession(config=config)
		self.sess.run(tf.global_variables_initializer())

	# ----- for restoring previous models
	def try_load_model(self, save_path):
		load_was_success = True # yes, I'm being optimistic
		global_step = 0
		try:
			save_dir = '/'.join(save_path.split('/')[:-1])
			ckpt = tf.train.get_checkpoint_state(save_dir)
			load_path = ckpt.model_checkpoint_path
			self.saver.restore(self.sess, load_path)
		except:
			self.logger.write("no saved model to load. starting new session")
			load_was_success = False
		else:
			self.logger.write("loaded model: {}".format(load_path))
			self.saver = tf.train.Saver(tf.global_variables())
			global_step = int(load_path.split('-')[-1])
		return load_was_success, global_step
