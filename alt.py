import tensorflow as tf
tf.compat.v1.enable_eager_execution()
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

def create_model(config, train=True):
    model = NLSN(config, train=train)
    return model

def compile_model(model, config):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
        beta_1=config['b1'],
        beta_2=config['b2'],
        epsilon=config['epsilon']
    )
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    metrics = [psnr_train if config['mode'] == 'train' else psnr_test, ssim_test]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model

def train_model(model, data_loader, config):
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    loss_and_metrics_callback = LossAndMetricCallback()

    model.fit_generator(
        generator=data_loader.generate_batch_train(),
        epochs=config['num_epochs'],
        steps_per_epoch=config['num_iter_per_epoch'],
        callbacks=[lr_callback, loss_and_metrics_callback],
        workers=1
    )
    model.save_weights("model_weights.h5")

    # Plot the Loss 
    plt.figure(figsize=(12, 4))
    plt.plot(loss_and_metrics_callback.losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plot_path = os.path.join("results", "results_plot.png")
    plt.savefig(plot_path)

def test_model(model, data_loader):
    model.build(input_shape=(None,48, 48, 3))
    model.load_weights("model_weights.h5")
    evaluation = model.evaluate(data_loader.generate_batch_test(), steps=50, workers=1)
    print(evaluation)

def main():
    # Load configuration
    parser = argparse.ArgumentParser(description='Train or test your model')
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Specify whether to train or test the model')
    parser.add_argument('--config', required=True, help='Path to the configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    config['mode'] = args.mode
    
    # Create the results folder
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define the data path
    data_dir = os.path.join(os.getcwd(), "dataset")
    
    # Initialize model and data loader
    
    model = create_model(config, train=(args.mode == 'train'))
    compile_model(model, config)
    
    if args.mode == 'train':
        data_loader = DataLoader(data_dir, config['scale'], config['patch_size'], config['batch_size'])
        train_model(model, data_loader, config)
    elif args.mode == 'test':
        data_loader = DataLoader(data_dir, config['scale'], config['patch_size'], batch_size = 1)
        test_model(model, data_loader)

if __name__ == '__main__':
    main()
