import tensorflow as tf
tf.enable_eager_execution()
#print(tf.executing_eagerly())
from data_loader.data_loader import DataLoader
from models.NLSN import NLSN
from utility import psnr_train, psnr_test, ssim_test, lr_scheduler, LossAndMetricCallback
import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def load_config(config_file):
	with open(config_file, 'r') as f:
		config = json.load(f)
	return config

def main():
	# Load configuration
	parser = argparse.ArgumentParser(description='Train or test your model')
	parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Specify whether to train or test the model')
	parser.add_argument('--config', required=True, help='Path to the configuration file')
	args = parser.parse_args()
	config = load_config(args.config)
	
	# Create the results folder
	results_dir = "results"
	os.makedirs(results_dir, exist_ok=True)
	
	# Define the data path
	data_dir = os.path.join(os.getcwd(), "dataset")
	
	# Define scale and hyperparameters
	patch_size = config['patch_size']
	batch_size = config['batch_size']
	num_epochs = config['num_epochs']
	learning_rate = config['learning_rate']
	b1 = config['b1']
	b2 = config['b2']
	epsilon = config['epsilon']	
	num_iter_per_epoch = config['num_iter_per_epoch']
	scale = config['scale']
	
	#print(tf.executing_eagerly())
	
    # Train mode
	if args.mode == 'train':
		# Initialize model and data loader
		data_loader = DataLoader(data_dir, scale, patch_size, batch_size)
		"""model_ = NLSN(scale, config, train = True)
		inputs = tf.keras.Input(shape=(48, 48, 3))
		outputs = model_(inputs)
		model = tf.keras.Model(inputs=inputs, outputs=outputs)"""
		model = NLSN(config, train = True)
		#model.build(input_shape=(4,48, 48, 3))
		print(model.summary())
		#print(model.summary())
		
		
		# Adam optimizer
		optimizer = tf.keras.optimizers.Adam(
				learning_rate=learning_rate,
				beta_1=b1,
				beta_2=b2,
				epsilon=epsilon)
		
		# L1 reconstruction loss
		loss_fn = tf.keras.losses.MeanAbsoluteError()
		
		# Peak signal to noise ratio
		metrics = [psnr_train]
			
		model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
		#model.run_eagerly=True
		lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
		loss_and_metrics_callback = LossAndMetricCallback()
		
		model.fit_generator(generator = data_loader.generate_batch_train(), epochs = 5, steps_per_epoch = 1, callbacks = [lr_callback, loss_and_metrics_callback], workers = 1)
		print(model.summary())
		model.save_weights("model_weights.h5")
	
		# Plot the Loss 
		plt.figure(figsize=(12, 4))		
		plt.plot(loss_and_metrics_callback.losses)
		plt.title('Training Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')			
		plot_path = os.path.join(results_dir, "results_plot.png")
		plt.savefig(plot_path)
		
	# Test mode
	elif args.mode == 'test':
        # Initialize model and data loader 
		data_loader = DataLoader(data_dir, scale, patch_size, batch_size = 1)		
		model = NLSN(config, train = False)
        
		# Peak signal to noise ratio
		metrics = [psnr_test, ssim_test]
		optimizer = tf.keras.optimizers.Adam(
				learning_rate=learning_rate,
				beta_1=b1,
				beta_2=b2,
				epsilon=epsilon
			)
		
		# L1 reconstruction loss
		loss_fn = tf.keras.losses.MeanAbsoluteError()
		
		model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
		model.build(input_shape=(None,48, 48, 3))
		#model.fit(x = data_loader.generate_batch_train(), epochs = 1, steps_per_epoch = 1,workers = 1)
		model.load_weights("model_weights.h5")
		
		evaluation = model.evaluate(data_loader.generate_batch_test(), callbacks =[] , steps = 50, workers = 1)
		
		print(evaluation)
		

if __name__ == '__main__':
	main()


#ssim, testarlo, dataloader per validation, controllare la pipeline