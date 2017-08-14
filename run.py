import numpy as np
import tensorflow as tf

import argparse
import time
import os
import random
import requests
import base64

from model import Model
from utils import *
from sample import *

def main():
	parser = argparse.ArgumentParser()

	#general model params
	parser.add_argument('--train', dest='train', action='store_true', help='train the model')
	parser.add_argument('--sample', dest='train', action='store_false', help='sample from the model')
	parser.add_argument('--validation', dest='validation', action='store_true', help='validation generation from the model')
	parser.add_argument('--rnn_size', type=int, default=400, help='size of RNN hidden state')
	parser.add_argument('--tsteps', type=int, default=360, help='RNN time steps (for backprop)')
	parser.add_argument('--nmixtures', type=int, default=20, help='number of gaussian mixtures')

	# window params
	parser.add_argument('--kmixtures', type=int, default=10, help='number of gaussian mixtures for character window')
	parser.add_argument('--alphabet', type=str, default=' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', \
						help='default is a-z, A-Z and <UNK> tag')
	parser.add_argument('--filter', type=str, default='\r\t\n', help='remove this from ascii before training')
	parser.add_argument('--tsteps_per_ascii', type=int, default=24, help='expected number of pen points per character')

	# training params
	parser.add_argument('--batch_size', type=int, default=32, help='batch size for each gradient step')
	parser.add_argument('--nbatches', type=int, default=500, help='number of batches per epoch')
	parser.add_argument('--nepochs', type=int, default=250, help='number of epochs')
	parser.add_argument('--dropout', type=float, default=0.85, help='probability of keeping neuron during dropout')

	parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients to this magnitude')
	parser.add_argument('--optimizer', type=str, default='rmsprop', help="ctype of optimizer: 'rmsprop' 'adam'")
	parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--lr_decay', type=float, default=1.0, help='decay rate for learning rate')
	parser.add_argument('--decay', type=float, default=0.95, help='decay rate for rmsprop')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum for rmsprop')

	#book-keeping
	parser.add_argument('--data_scale', type=int, default=50, help='amount to scale data down before training')
	parser.add_argument('--log_dir', type=str, default='./logs/', help='location, relative to execution, of log files')
	parser.add_argument('--valid_dir', type=str, default='./valid/', help='location, relative to execution, of validation output files')
	parser.add_argument('--data_dir', type=str, default='./data', help='location, relative to execution, of data')
	parser.add_argument('--save_path', type=str, default='saved/model.ckpt', help='location to save model')
	parser.add_argument('--save_every', type=int, default=500, help='number of batches between each save')

	#sampling
	parser.add_argument('--text', type=str, default='', help='string for sampling model (defaults to test cases)')
	parser.add_argument('--style', type=int, default=-1, help='optionally condition model on a preset style (using data in styles.p)')
	parser.add_argument('--bias', type=float, default=1.0, help='higher bias means neater, lower means more diverse (range is 0-5)')
	parser.add_argument('--sleep_time', type=int, default=60*5, help='time to sleep between running sampler')
	parser.add_argument('--repeat', dest='repeat', action='store_true', help='repeat sampling infinitly')
	parser.add_argument('--no_info', dest='add_info', action='store_false', help='adds additional info')

	parser.set_defaults(repeat=False)
	parser.set_defaults(add_info=True)
	parser.set_defaults(train=True)
	parser.set_defaults(validation=False)
	args = parser.parse_args()
	if (args.validation):
		validation_run(args)
	else:
		train_model(args) if args.train else sample_model(args, add_info=args.add_info)

def train_model(args):
	logger = Logger(args) # make logging utility
	logger.write("\nTRAINING MODE...")
	logger.write("{}\n".format(args))
	logger.write("loading data...")
	data_loader = DataLoader(args, logger=logger)
	# Preprocessing complete, created a validation set and training set , and got the number of batches.

	logger.write("building model...")
	model = Model(args, logger=logger)

	logger.write("attempt to load saved model...")
	load_was_success, global_step = model.try_load_model(args.save_path)

	# Validates data once, which validates only 32 lines out of the entire validation set
	v_x, v_y, v_s, v_c = data_loader.validation_data()
	# INPUTS data to the model
	# V_X ==> x_batch 
	# v_y ==> y_batch (Which is the next point after the x_batch)
	# v_c ==> One_hot sequence
	# Target_data ==> This is the next point to be predicted
	valid_inputs = {model.input_data: v_x, model.target_data: v_y, model.char_seq: v_c}

	logger.write("training...")
	model.sess.run(tf.assign(model.decay, args.decay ))
	model.sess.run(tf.assign(model.momentum, args.momentum ))
	running_average = 0.0 ; remember_rate = 0.99

	# Global_steps is the number indented at the end of the file
	# Nepochs is the number the training occurs 
	# nBatches is the number of the batches
	for e in range(global_step/args.nbatches, args.nepochs):
		model.sess.run(tf.assign(model.learning_rate, args.learning_rate * (args.lr_decay ** e)))
		logger.write("learning rate: {}".format(model.learning_rate.eval()))

		c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
		h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()
		kappa = np.zeros((args.batch_size, args.kmixtures, 1))

		for b in range(global_step%args.nbatches, args.nbatches):
			i = e * args.nbatches + b
			if global_step is not 0 : i+=1 ; global_step = 0

			if i % args.save_every == 0 and (i > 0):
				model.saver.save(model.sess, args.save_path, global_step = i) ; logger.write('SAVED MODEL')
				data_loader.save_pointer()

			start = time.time()
			x, y, s, c = data_loader.next_batch()

			feed = {model.input_data: x, model.target_data: y, model.char_seq: c, model.init_kappa: kappa, \
					model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2, \
					model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}

			[train_loss, _] = model.sess.run([model.cost, model.train_op], feed)
			feed.update(valid_inputs)
			feed[model.init_kappa] = np.zeros((args.batch_size, args.kmixtures, 1))
			[valid_loss] = model.sess.run([model.cost], feed)
			
			running_average = running_average*remember_rate + train_loss*(1-remember_rate)

			end = time.time()
			if i % 10 is 0: logger.write("{}/{}, loss = {:.3f}, regloss = {:.5f}, valid_loss = {:.3f}, time = {:.3f}" \
				.format(i, args.nepochs * args.nbatches, train_loss, running_average, valid_loss, end - start) )
	model.saver.save(model.sess, args.save_path, global_step = args.nepochs * args.nbatches) ; logger.write('SAVED MODEL')
	data_loader.save_pointer()

def sample_model(args, logger=None, add_info=True, model=None, save_path=None):
	if args.text == '':
		strings = ['call me ishmael some years ago', 'A project by Sam Greydanus', 'mmm mmm mmm mmm mmm mmm mmm', \
			'What I cannot create I do not understand', 'You know nothing Jon Snow'] # test strings
	else:
		strings = [args.text]

	logger = Logger(args) if logger is None else logger # instantiate logger, if None
	logger.write("\nSAMPLING MODE...")

	if (model is None):
		logger.write("building model...")
		model = Model(args, logger)

		logger.write("attempt to load saved model...")
		load_was_success, global_step = model.try_load_model(args.save_path)
	else:
		model = model
		load_was_success, global_step = model.try_load_model(args.save_path)
	if (save_path is None):
		save_path = args.log_dir
	else:
		save_path = save_path

	if load_was_success:
		for s in strings:
			words = s.split(" ")
			strokes, phis, windows, kappas = [], [], [], []
			prev_x = 0
			for word in words:
				strokes_temp, phis_temp, windows_temp, kappas_temp = sample(word, model, args)
				mod_strokes = np.asarray(strokes_temp, dtype = np.float32)
				mod_strokes[:,0] += prev_x
				mod_strokes[len(mod_strokes) - 1, 5] = 1
				strokes.append(mod_strokes)
				phis = combine_image_matrixes(phis, phis_temp)
				windows = combine_image_matrixes(windows, windows_temp)
				kappas.append(kappas_temp)
				prev_x = mod_strokes[:,0].max() + random.uniform(2,5)
			windows = np.vstack(windows)	
			phis = np.vstack(phis)
			kappas = np.vstack(kappas)
			strokes = np.vstack(strokes)
			if (add_info):
				w_save_path = '{}figures/iter-{}-w-{}.png'.format(save_path, global_step, s[:10].replace(' ', '_'))
				g_save_path = '{}figures/iter-{}-g-{}.png'.format(save_path, global_step, s[:10].replace(' ', '_'))
				l_save_path = '{}figures/iter-{}-l-{}.png'.format(save_path, global_step, s[:10].replace(' ', '_'))
			else:
				l_save_path = '{}figures/{}.png'.format(save_path, s)
			if (add_info):
				window_plots(phis, windows, save_path=w_save_path)
				gauss_plot(strokes, 'Heatmap for "{}"'.format(s), figsize = (2*len(s),4), save_path=g_save_path)
				logger.write( "kappas: \n{}".format(str(kappas[min(kappas.shape[0]-1, args.tsteps_per_ascii),:])) )
			line_plot(strokes, 'Line plot for "{}"'.format(s), figsize = (len(s),2), save_path=l_save_path, add_info=add_info)
			
	else:
		logger.write("load failed, sampling canceled")

	if args.repeat:
		tf.reset_default_graph()
		time.sleep(args.sleep_time)
		sample_model(args, logger=logger)

def validation_run(args, logger=None):
	args.train = False
	args.repeat = False
	logger = Logger(args) if logger is None else logger # instantiate logger, if None
	logger.write("\nValidation MODE...")
	logger.write("loading data...")
	data_loader = DataLoader(args, logger=logger)
	logger.write("building model...")
	model = Model(args, logger)
	logger.write("attempt to load saved model...")
	load_was_success, global_step = model.try_load_model(args.save_path)
	if (load_was_success):
		logger.write("Load successfull...")
		for i in range(len(data_loader.valid_ascii_data)):
			logger.write("Sampling {} validation data".format(i + 1))
			args.text = data_loader.valid_ascii_data[i]
			sample_model(args, logger, add_info=False, model=model, save_path = args.valid_dir)
			logger.write("Finished sampling ...")
			logger.write("Saving ascii representation for sample {}".format(i + 1))
			f = open("{}ascii/{}.txt".format(args.valid_dir, args.text[:10].replace(' ', '_')), 'w')
			f.write(args.text)
			f.close()
	else:
		logger.write("No saved model.....Validating cancelled !")


if __name__ == '__main__':
	main()
