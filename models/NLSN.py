import tensorflow as tf
import tensorflow.keras.layers as layers
from models import attention
from models import common


class NLSN(tf.keras.Model):
	def __init__(self, config, train):
		super(NLSN, self).__init__()
		self.train = train
		self.n_resblock = config['n_resblock']
		self.n_feats = config['n_feats']
		self.kernel_size = config['kernel_size']
		self.n_hashes = config['n_hashes']
		self.reduction = config['reduction']
		self.chunk_size = config['chunk_size']
		self.res_scale = config['res_scale']
		self.scale = config['scale']
		self.act = tf.keras.layers.Activation('relu')
		
		self.sub_mean = common.MeanShift()
		self.conv1 = layers.Conv2D(self.n_feats, self.kernel_size, padding='same')
		b = []
		b.append(attention.NonLocalSparseAttention(channels=self.n_feats, n_hashes=self.n_hashes, k_size=self.kernel_size,
												reduction=self.reduction, chunk_size=self.chunk_size, res_scale=self.res_scale,
												config = config, train = train))
		for i in range(self.n_resblock):
			b.append(common.ResBlock(n_feats=self.n_feats, kernel_size=self.kernel_size, act=self.act, res_scale=self.res_scale))
			if (i+1%8==0):				
				b.append(attention.NonLocalSparseAttention(channels=self.n_feats, n_hashes=self.n_hashes, k_size=self.kernel_size,
														reduction=self.reduction, chunk_size=self.chunk_size, res_scale=self.res_scale,
														config = config, train = train))
		b.append(layers.Conv2D(self.n_feats, self.kernel_size, padding='same'))	
		
		t = []
		t.append(common.Upsampler(scale=self.scale, n_feats=self.n_feats, kernel_size=self.kernel_size))
		t.append(layers.Conv2D(3, self.kernel_size, padding = 'same'))		
		
		self.body = tf.keras.Sequential(b)
		self.tail= tf.keras.Sequential(t)
		self.add_mean = common.MeanShift(sign = 1 )
		
	def call(self, x):		
		x = self.sub_mean(x)
		x = self.conv1(x)		
		res = self.body(x)
		res = res + x		
		x = self.tail(res)
		x = self.add_mean(x)		
		return x		
		
	def summary(self):
		x = tf.keras.Input(shape=( 48, 48, 3))
		model = tf.keras.Model(inputs=[x], outputs=self.call(x))
		return model.summary()

  
