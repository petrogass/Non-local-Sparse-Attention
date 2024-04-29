import tensorflow as tf
import tensorflow.keras.layers as layers 
		
class MeanShift(layers.Layer):
	def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), sign = -1):
		super(MeanShift, self).__init__()
		self.rgb_mean = sign*tf.constant(rgb_mean, dtype=tf.float32)
		
	def call(self, x):		  
		res = x + self.rgb_mean		   
		return res
	
	

def torch_gather(x, indices, gather_axis):
	all_indices = tf.where(tf.fill(indices.shape, True))
	gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

	gather_indices = []
	for axis in range(len(indices.shape)):
		if axis == gather_axis:
			gather_indices.append(tf.cast(gather_locations, dtype=tf.int64))
		else:
			gather_indices.append(tf.cast(all_indices[:, axis], dtype=tf.int64))

	gather_indices = tf.stack(gather_indices, axis=-1)
	gathered = tf.gather_nd(x, gather_indices)
	reshaped = tf.reshape(gathered, indices.shape)
	return reshaped

def batched_index_select(values, indices, last_dim):
	#last_dim = values.shape[-1]	
	# Add a new axis to indices and expand it
	expanded_indices = tf.expand_dims(indices, axis=-1)
	expanded_indices = tf.tile(expanded_indices, [1, 1, last_dim])	 
	# Gather elements from values based on the expanded indices
	selected_elements = torch_gather(values, expanded_indices, gather_axis=1)	 
	return selected_elements	
   

class ResBlock(layers.Layer):
	def __init__(self, n_feats, kernel_size, act, res_scale):
		super(ResBlock, self).__init__()
		m = []
		for i in range(2):
			m.append(layers.Conv2D(n_feats, kernel_size, padding = 'same'))					   
			if i == 0:				
				m.append(act)
		self.body = tf.keras.Sequential(m)
		self.res_scale = res_scale

	def call(self, x):
		res = self.body(x) * self.res_scale
		res += x
		return res
	"""def compute_output_shape(self, input_shape):
		return input_shape"""
	

class Upsampler(layers.Layer):
	def __init__(self, scale, n_feats, kernel_size):
		super(Upsampler, self).__init__()
		m = []
		if scale == 2:
			m.append(layers.Conv2D(4*n_feats, kernel_size, padding='same'))
			m.append(layers.Lambda(lambda x:tf.nn.depth_to_space(x,2)))
		if scale == 3:
			m.append(layers.Conv2D(9*n_feats, kernel_size, padding='same'))
			m.append(layers.Lambda(lambda x:tf.nn.depth_to_space(x,3)))
			
		self.body = tf.keras.Sequential(m)
		
	def call(self, x):
		return self.body(x)
			
	"""def compute_output_shape(self, input_shape):
		return (None, 96,96, 256)"""