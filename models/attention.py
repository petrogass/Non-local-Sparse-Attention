import tensorflow as tf
import tensorflow.keras.layers as layers 
from models import common

class NonLocalSparseAttention(tf.keras.layers.Layer):
	def __init__(self, channels, config, n_hashes=4,  k_size=3, reduction=4, chunk_size=144, res_scale=0.1, train = True):
		super(NonLocalSparseAttention, self).__init__()
		self.N = 1
		if train:
			self.N = config['batch_size']
		self.H = config['patch_size']
		self.W = config['patch_size']
		self.L = config['patch_size']*config['patch_size']
		self.C = channels//reduction
		self.chunk_size = chunk_size
		self.n_hashes = n_hashes
		self.reduction = reduction
		self.res_scale = res_scale		
		self.conv_match = layers.Conv2D(filters = channels // reduction,strides = 1, kernel_size = k_size, padding='same')		
		self.conv_assembly = layers.Conv2D(filters = channels, kernel_size = k_size, padding='same')
		

	def LSH(self, hash_buckets, x):
		#N = tf.shape(x)[0]
		
		rotations_shape = (1, self.C, self.n_hashes, hash_buckets // 2)
		
		random_rotations = tf.random.normal(rotations_shape, dtype=x.dtype)
		random_rotations = tf.tile(random_rotations, (self.N, 1, 1, 1))
		
		

		rotated_vecs = tf.einsum('btf,bfhi->bhti', x, random_rotations)
		rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)

		hash_codes = tf.argmax(rotated_vecs, axis=-1, output_type = tf.dtypes.int32)
		offsets = tf.range(self.n_hashes)
		offsets = offsets * hash_buckets
		offsets = tf.reshape(offsets, (1, -1, 1))		
		hash_codes = tf.reshape(hash_codes + offsets, (self.N, -1))

		return hash_codes
		

	def add_adjacent_buckets(self, x):
		x_extra_back = tf.concat([x[:, :, -1:, ...], x[:, :, :-1, ...]], axis=2)
		x_extra_forward = tf.concat([x[:, :, 1:, ...], x[:, :, :1, ...]], axis=2)
		return tf.concat([x, x_extra_back, x_extra_forward], axis=3)
		
	
	def call(self, input):
		#N, H, W, _ = input.shape #sappiamo cosa sono, togliamo// sempre un problema che non legge N, rendiamo tutto statico tanto è 16 a train e 1 a test, H w e c 
		#pure sono statici
		
		x_embed = self.conv_match(input)		
		x_embed = tf.reshape(x_embed, (self.N, -1,  self.H * self.W))
		x_embed = tf.transpose(x_embed, (0, 2, 1))
		
		
		y_embed = self.conv_assembly(input)
		y_embed = tf.reshape(y_embed, (self.N, -1,  self.H * self.W))
		y_embed = tf.transpose(y_embed, (0, 2, 1))

		
		
		

		hash_buckets = min(self.L // self.chunk_size + (self.L // self.chunk_size) % 2, 128)
		
		hash_codes = self.LSH(hash_buckets, x_embed)
		# Detach the tensor
		hash_codes = tf.stop_gradient(hash_codes)		
		# Group elements with the same hash code by sorting
		indices = tf.argsort(hash_codes, axis=-1) #use python
		#_, indices = hash_codes.sort(dim=-1)
		undo_sort = tf.argsort(indices, axis=-1)
		#_, undo_sort = indices.sort(dim=-1)
		mod_indices = indices % self.L
		
		x_embed_sorted = common.batched_index_select(x_embed, mod_indices, self.C)
		y_embed_sorted = common.batched_index_select(y_embed, mod_indices, self.C*self.reduction)		
		
		padding = self.chunk_size - self.L % self.chunk_size if self.L % self.chunk_size != 0 else 0
		x_att_buckets = tf.reshape(x_embed_sorted, (self.N, self.n_hashes, -1, self.C))
		y_att_buckets = tf.reshape(y_embed_sorted, (self.N, self.n_hashes, -1, self.C * self.reduction))
		
		if padding:
			pad_x = x_att_buckets[:, :, -padding:, ...]
			pad_y = y_att_buckets[:, :, -padding:, ...]
			x_att_buckets = tf.concat([x_att_buckets, pad_x], axis=2)
			y_att_buckets = tf.concat([y_att_buckets, pad_y], axis=2)

		x_att_buckets = tf.reshape(x_att_buckets, (self.N, self.n_hashes, -1, self.chunk_size, self.C))
		y_att_buckets = tf.reshape(y_att_buckets, (self.N, self.n_hashes, -1, self.chunk_size, self.C * self.reduction))

		x_match = tf.math.l2_normalize(x_att_buckets, axis=-1, epsilon=5e-5)

		x_match = self.add_adjacent_buckets(x_match)
		y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

		raw_score = tf.einsum('bhkie,bhkje->bhkij', x_att_buckets, x_match)
		

		bucket_score = tf.reduce_logsumexp(raw_score, axis=-1, keepdims=True)
		score = tf.exp(raw_score - bucket_score)
		bucket_score = tf.reshape(bucket_score, (self.N, self.n_hashes, -1))

		ret = tf.einsum('bukij,bukje->bukie', score, y_att_buckets)
		ret = tf.reshape(ret, (self.N, self.n_hashes, -1, self.C * self.reduction))

		if padding:
			ret = ret[:, :, :-padding, :].copy()
			bucket_score = bucket_score[:, :, :-padding].copy()

		ret = tf.reshape(ret, (self.N, -1, self.C * self.reduction))
		bucket_score = tf.reshape(bucket_score, (self.N, -1,))
		
		ret = common.batched_index_select(ret, undo_sort, self.C*self.reduction)
		
		bucket_score = common.torch_gather(bucket_score, undo_sort, gather_axis=1)
		#print(bucket_score.shape)
		ret = tf.reshape(ret, (self.N, self.n_hashes, self.L, self.C * self.reduction))
		
		bucket_score = tf.reshape(bucket_score, (self.N, self.n_hashes, self.L, 1))
		probs = tf.nn.softmax(bucket_score, axis=1)
		ret = tf.reduce_sum(ret * probs, axis=1)

		ret = tf.transpose(ret, (0, 2, 1))
		ret = tf.reshape(ret, (self.N, self.H, self.W,-1,)) * self.res_scale + input
		
		
		return ret
		
	"""def compute_output_shape(self, input_shape):
		print(input_shape)
		return(input_shape)"""
		

		


