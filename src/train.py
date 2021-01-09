import time
import logging
from os import path
from itertools import cycle
from datetime import datetime
from json import dump, dumps

import tensorflow as tf
from matplotlib import pyplot as plt

from utils import plot_single_image, plot_multiple_images, logger
from losses import l1_image_loss, multiscale_ssim_loss
from core.pan import PixelAttentionSRNetwork
from data import LR_IMAGE_SIZE, HR_IMAGE_SIZE, CLEAR_IMAGES_DIR, get_dataset


batch_size = 24
input_lr_shape = (None, *LR_IMAGE_SIZE, 3) # HxWxC
input_hr_shape = (None, *HR_IMAGE_SIZE, 3) # HxWxC
print_freq = 100

n_epochs = 30
# every 10 epochs (bs=24) decay current lr to lr_new = prev_lr * 0.1
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=3333,
    decay_rate=.1,
)

alpha = tf.constant(0.84)

cpu = '/CPU:0'
gpu = tf.test.gpu_device_name()

device = gpu or cpu
logger.info(f'Running on device {repr(device)}')


max_ckpt_to_keep = 3
checkpoint_dir = './model/checkpoints'
checkpoint_name = 'pix-att-net-ckpt'


logger = logging.getLogger()


@tf.function
def train_step(lr_batch_images, hr_batch_images):
    with tf.GradientTape() as tape:
        enhanced_images = sr_net(lr_batch_images, training=True)
        loss = alpha * multiscale_ssim_loss(hr_batch_images, enhanced_images) + (1 - alpha) * 0.1333 * l1_image_loss(hr_batch_images, enhanced_images)
    
    grads = tape.gradient(loss, sr_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, sr_net.trainable_variables))

    return enhanced_images, float(loss)


def train():
    with tf.device(device):
        losses = []
        train_psnrs, val_psnrs, test_psnrs = [], [], []
        total_training_time = 0.0
        global_iter_count = 0
        for i_epoch in range(n_epochs):       
            tick = time.time()
            for lr_images, hr_images in train_dataset:
                enhanced_imgs, loss = train_step(lr_images, hr_images)
                losses.append(float(loss))
                train_psnr = float(tf.reduce_mean(tf.image.psnr(hr_images, tf.clip_by_value(enhanced_imgs, 0, 255), 255)))
                train_psnrs.append(train_psnr)
                print(train_psnr)
                global_iter_count += 1
                if global_iter_count % print_freq == 0:
                    fig_file_path_tmpl = './out/visualization/epoch_{i_epoch}_iter_{iter_cnt}_{data_clf}_results.png'
                    fig_title_tmpl = 'Epoch {i_epoch}. Iteration {iter_cnt}. {data_clf_label} PSNR: {psnr}'
                    
                    plot_multiple_images((enhanced_imgs, hr_images), fig_title_tmpl.format(i_epoch=i_epoch, iter_cnt=global_iter_count, data_clf_label='Train', psnr=train_psnr), figsize=(8, 8))
                    plt.savefig(fig_file_path_tmpl.format(i_epoch=i_epoch, iter_cnt=global_iter_count, data_clf='train'))
                    plt.close()

                    lr_images, hr_images = next(val_dataset)
                    enhanced_imgs = tf.clip_by_value(sr_net(lr_images, training=False), 0, 255)
                    val_psnr = float(tf.reduce_mean(tf.image.psnr(hr_images, enhanced_imgs, 255)))
                    plot_multiple_images((enhanced_imgs, hr_images), fig_title_tmpl.format(i_epoch=i_epoch, iter_cnt=global_iter_count, data_clf_label='Validation', psnr=val_psnr), figsize=(8, 8))
                    plt.savefig(fig_file_path_tmpl.format(i_epoch=i_epoch, iter_cnt=global_iter_count, data_clf='val'))
                    plt.close()
                    val_psnrs.append(val_psnr)

                    lr_images, hr_images = next(test_dataset)
                    enhanced_imgs = tf.clip_by_value(sr_net(lr_images, training=False), 0, 255)
                    test_psnr = float(tf.reduce_mean(tf.image.psnr(hr_images, enhanced_imgs, 255)))
                    plot_multiple_images((enhanced_imgs, hr_images), fig_title_tmpl.format(i_epoch=i_epoch, iter_cnt=global_iter_count, data_clf_label='Test', psnr=test_psnr), figsize=(8, 8))
                    plt.savefig(fig_file_path_tmpl.format(i_epoch=i_epoch, iter_cnt=global_iter_count, data_clf='test'))
                    plt.close()
                    test_psnrs.append(test_psnr)

                    logger.info('Epoch {0}. Iteration {1}. Loss: {2:.4f}. Train PSNR: {3:.4f}. Val PSNR: {4:.4f}. Test PSNR: {5:.4f}.'.format(
                        i_epoch, global_iter_count, loss, float(train_psnr), float(val_psnr), float(test_psnr)
                    ))  
                    
                    ckpt_manager.save()
                    print('Saved model')
                    
                else:
                    logger.info('Epoch {0}. Batch {1}. Loss: {2:.4f}. Train PSNR: {3:.4f}.'.format(
                        i_epoch, global_iter_count, loss, train_psnr
                    ))
              
            tock = time.time()
            epoch_duration = round(tock - tick, 3)
            total_training_time += epoch_duration 
            logger.info(f'Run epoch #{i_epoch} in {epoch_duration} seconds')

    with open(f'./out/{int(datetime.now().timestamp())}_train_results.json', 'w') as fid:
        dump(dict(
            batch_size=batch_size, 
            optimizer_conf={k:str(v) for k, v in optimizer.get_config().items()},
            device=device,
            n_epochs=n_epochs, 
            lr_image_size=LR_IMAGE_SIZE,
            hr_image_size=HR_IMAGE_SIZE,
            total_training_time=total_training_time,
            losses=losses,
            train_psnrs=train_psnrs,
            val_psnrs=val_psnrs,
            test_psnrs=test_psnrs,
        ), fid, indent=4)


if __name__ == '__main__':
    with tf.device(device):
        train_dataset = get_dataset(path.join(CLEAR_IMAGES_DIR, 'train')).batch(batch_size)
        val_dataset = cycle(get_dataset(path.join(CLEAR_IMAGES_DIR, 'val')).batch(batch_size))
        test_dataset = cycle(get_dataset(path.join(CLEAR_IMAGES_DIR, 'test')).batch(batch_size))

        optimizer = tf.keras.optimizers.Adam(learning_rate)
        sr_net = PixelAttentionSRNetwork(feat_extr_n_filters=30, upsamp_n_filters=20, n_blocks=16, scale=2, input_shape=input_lr_shape)
        
        checkpoint = tf.train.Checkpoint(
            sr_net=sr_net,
            optimizer=optimizer,
        )

        sr_net.build(input_lr_shape)
        
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_ckpt_to_keep, checkpoint_name=checkpoint_name)
        
        # try restoring previous checkpoint or initialize a new one
        restored_ckpt_path = ckpt_manager.restore_or_initialize()
        if restored_ckpt_path:
            logger.info('Restored state from checkpoint {}'.format(repr(restored_ckpt_path)))

        train()