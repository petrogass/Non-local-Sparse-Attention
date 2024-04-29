import tensorflow as tf
import cv2
import numpy as np
import threading
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def psnr_train(y_true, y_pred):
	# Calculate PSNR 
	
	return tf.image.psnr(y_true, y_pred, max_val=1.0)

def psnr_test(y_true, y_pred):
	
	# Convert RGB to YCbCr
	y_true_ycbcr = tf.image.rgb_to_yuv(y_true)
	y_pred_ycbcr = tf.image.rgb_to_yuv(y_pred)

	# Extract the Y channel
	y_true_y = y_true_ycbcr[..., 0]
	
	y_pred_y = y_pred_ycbcr[..., 0]
	

	# Calculate PSNR for the Y channel
	return tf.image.psnr(y_true_y, y_pred_y, max_val=1.0)

def ssim_test(y_true, y_pred):
	
	# Convert RGB to YCbCr
	y_true_ycbcr = tf.image.rgb_to_yuv(y_true)
	y_pred_ycbcr = tf.image.rgb_to_yuv(y_pred)

	# Extract the Y channel
	y_true_y = y_true_ycbcr[..., 0]
	y_true_y = tf.expand_dims(y_true_y, axis=3)	 
	
	
	y_pred_y = y_pred_ycbcr[..., 0]
	y_pred_y = tf.expand_dims(y_pred_y, axis=3)
	

	# Calculate SSIM for the Y channel
	return tf.image.ssim(y_true_y, y_pred_y, max_val=1.0)

	
def lr_scheduler(epoch):
	print('callback')
	initial_lr = 1e-7
	drop_every = 200
	drop_factor = 0.5
	learning_rate = initial_lr * (drop_factor ** (epoch // drop_every))
	return learning_rate

class ThreadsafeIter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self
	def generate_batch_train(self):
		with self.lock:
			return self.it.generate_batch_train()
			
	def __next__(self):
		with self.lock:
			return self.it.__next__()

def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return ThreadsafeIter(f(*a, **kw))
	return g
	
class LossAndMetricCallback(tf.keras.callbacks.Callback):
	def __init__(self):
		super(LossAndMetricCallback, self).__init__()
		self.losses = []
		self.psnr = []
		

	def on_epoch_end(self, epoch, logs=None):
		print(logs)
		self.losses.append(logs['loss'])
		self.psnr.append(logs['psnr_train'])
		
