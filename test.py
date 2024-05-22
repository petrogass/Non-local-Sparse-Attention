import tensorflow as tf
from models import attention
class MeanShift(tf.keras.layers.Layer):
    def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), sign = -1):
        super(MeanShift, self).__init__()
        self.rgb_mean = sign*tf.constant(rgb_mean, dtype=tf.float32)
		
    def call(self, x):
        # Subtract the mean value at the beginning
        res = x + self.rgb_mean        
        return res
		

"""lr_image = tf.io.read_file("C:/Users/petro/Desktop/NLSA/dataset/DIV2K_TRAIN_LR/X2/0001x2.png")		 
lr_image = tf.image.decode_image(lr_image, channels=3)
lr_image /= 255
print(lr_image)

mf= MeanShift()
x = mf.call(lr_image)
print(x)"""

#salvare le immagini in test mode, mettere possibilitá di caricare dataset diversi in test mode