import os
import random
import os
import random
import threading
import tensorflow as tf
from tensorflow import image as tf_image
from utility import threadsafe_generator

class DataLoader:
    def __init__(self, data_dir, scale, patch_size=48, batch_size=16):
        self.lock = threading.Lock()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.scale = scale
        self.lr_image_paths = self.load_image_paths('DIV2K_TRAIN_LR/X2')
        self.hr_image_paths = self.load_image_paths('DIV2K_TRAIN_HR')
        self.lr_test_image_paths = self.load_image_paths('DIV2K_TEST_LR/X2')
        self.hr_test_image_paths = self.load_image_paths('DIV2K_TEST_HR')

    def load_image_paths(self, folder_name):
        folder_path = os.path.join(self.data_dir, folder_name)
        image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return image_paths

    def load_and_augment_image(self, lr_image_path, hr_image_path):
        #print(lr_image_path)
        lr_image = tf.io.read_file(lr_image_path)
        #print(tf.shape(lr_image))      
        lr_image = tf.image.decode_image(lr_image, channels=3)  
        hr_image = tf.io.read_file(hr_image_path)
        hr_image = tf.image.decode_image(hr_image, channels=3)  
        
        # Get random top-left coordinates for cropping
        #print(tf.shape(lr_image))
        top = random.randint(0, lr_image.shape[0] - self.patch_size)
        left = random.randint(0, lr_image.shape[1] - self.patch_size)

        # Crop both LR and HR images
        lr_image = lr_image[top:top+self.patch_size, left:left+self.patch_size, :]
        hr_image = hr_image[top*self.scale:(top+self.patch_size)*self.scale, left*self.scale:(left+self.patch_size)*self.scale, :]
        
        # Random horizontal flip
        if random.random() > 0.5:
            lr_image = tf.image.flip_left_right(lr_image)
            hr_image = tf.image.flip_left_right(hr_image)
            
        # Random 90-degree rotation
        if random.random() > 0.5:
            lr_image = tf.image.rot90(lr_image, k=random.randint(1, 3))
            hr_image = tf.image.rot90(hr_image, k=random.randint(1, 3))

        # Normalize to [0, 1]
        lr_image = tf.cast(lr_image, tf.float32) / 255.0
        hr_image = tf.cast(hr_image, tf.float32) / 255.0
        return lr_image, hr_image


    def generate_batch_test(self):
        while True:
            batch_lr_images = []
            batch_hr_images = []            
            for _ in range(self.batch_size):
                random_index = random.randint(0, len(self.lr_test_image_paths) - 1)
                lr_image, hr_image = self.load_and_augment_image(self.lr_test_image_paths[random_index], self.hr_test_image_paths[random_index])                
                batch_lr_images.append(lr_image)
                batch_hr_images.append(hr_image)
            yield tf.stack(batch_lr_images), tf.stack(batch_hr_images)  
            
            
    def generate_batch_train(self):        
        while True:
            with self.lock: 
                batch_lr_images = []
                batch_hr_images = []
                for _ in range(self.batch_size):
                    random_index = random.randint(0, len(self.lr_image_paths) - 1)
                    lr_image, hr_image = self.load_and_augment_image(self.lr_image_paths[random_index], self.hr_image_paths[random_index])
                    batch_lr_images.append(lr_image)
                    batch_hr_images.append(hr_image) 
                yield tf.stack(batch_lr_images), tf.stack(batch_hr_images)
