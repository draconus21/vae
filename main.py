import tensorflow as tf
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
import re
from datetime import datetime

from cvae_net import CVAE

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2*np.pi)
    return tf.reduce_sum(
        -.5 * ((sample-mean) **2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz   = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return - tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(model, epoch, test_input, figdir):
    predictions = model.sample(test_input)
    nr, nc = 4, 4
    figsize = [4, 4]
    fig, ax = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True, squeeze=False,
                           figsize=figsize)
    for i in range(predictions.shape[0]):
        ax[i//nc, i%nc].matshow(predictions[i, :, :, 0], cmap='gray')

    plt.savefig(os.path.join(figdir, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close('all')


def generateGIF(figdir):
    anim_fname = os.path.join(figdir, 'cvae.gif')
    regEx = re.compile('image*.png')
    with imageio.get_writer(anim_fname, mode='I') as writer:
        fnames = [f for f in os.listdir(figdir) if re.match(f)]

        fnames = sorted(fnames)
        last = -1

        for i, fname in enumerate(fnames):
            frame = 2*(i**.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(os.path.join(figdir, fname))
            writer.append_data(image)

        #image = imageio.imread(os.path.join(figdir, fname))
        #writer.append_data(image)

if __name__=='__main__':
    now = datetime.now()
    rootdir = os.curdir
    figdir = os.path.join(rootdir, 'figs', now.strftime('%y%m%d_%H%M%S'))
    os.makedirs(figdir)
    input_shape = [28, 28, 1]
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], *input_shape).astype(np.float32)
    test_images = test_images.reshape(test_images.shape[0], *input_shape).astype(np.float32)

    # Normalize to range [0, 1]
    train_images = train_images/255.
    test_images = test_images/255.

    # Binarization
    thr = .5
    train_images[train_images >= thr] = 1.
    train_images[train_images < thr]  = 0.
    test_images[test_images >= thr]   = 1.
    test_images[test_images < thr]    = 0.

    TRAIN_BUF = 60000
    BATCH_SIZE = 100

    TEST_BUF = 10000

    # create batches
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset  = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)


    optimizer = tf.keras.optimizers.Adam(1e-4)
    epochs = 100
    latent_dim = 50
    num_examples_to_generate = 16

    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim)

    generate_and_save_images(model, 0, random_vector_for_generation, figdir)

    for epoch in range(1, epochs+1):
        start_time = time.time()
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x, optimizer)
        end_time = time.time()

        if epoch % 10 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
                elbo = -loss.result()
                print('Epoch: {}, test set ELBO: {}, '
                      'time elapse for current epoch {}'.format(epoch, elbo, end_time-start_time))
        generate_and_save_images(model, epoch, random_vector_for_generation, figdir)

    generateGIF(figdir)

