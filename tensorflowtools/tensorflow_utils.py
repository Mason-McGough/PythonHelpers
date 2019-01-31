import os
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.layers import LeakyReLU as LReLU

# for python2
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def save_model(sess, saver, checkpoint_dir, model_name='', step=None):
    """
    Save the current model checkpoint.

    Inputs:
        sess - The current tensorflow session.
        saver - The instance of tensorflow Saver class managing the model.
        checkpoint_dir - The directory to save the checkpoint.
        model_name - The name of the current model to prefix checkpoint filenames.
            (Default: '')
        step - The current value of the global step used for training. (Default: None)
    Outputs:
        None
    """

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

def load_model(sess, saver, checkpoint_dir, strict=False):
    """
    Load a model checkpoint from directory.

    Inputs:
        sess - The current tensorflow session.
        saver - The instance of tensorflow Saver class managing the model.
        checkpoint_dir - The directory where the saved checkpoint is stored.
        strict - If True, the function returns a FileNotFoundError (IOError in 
            Python2) if a valid checkpoint file is not found.
    Outputs:
        success - True if the checkpoint is successfully loaded from memory.
    """

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        if strict:
            raise FileNotFoundError('Checkpoint file not found in: {}'.format(checkpoint_dir))
        else:
            return False

def build_unet(inputs, n_layers=9, n_filters=32, conv2d_size=(3, 3), pool_size=(2, 2), 
               n_outputs=1, lrelu_alpha=0.0):
    """
    Construct a standard UNet model.

    Inputs:
        inputs - The input tensor to the network. 
        n_layers - The number of network layers. Each layer consists of two 
            2D-convolution and LReLU activations, followed by either max pooling 
            if before the midpoint (if l < (n_layers - 1)/2) or beginning with an 
            upsampling and concatenation if after the midpoint 
            (if l > (n_layers - 1)/2). (Default: 9)
        n_filters - The number of filters in the first layer. The number of filters
            in the following layers are multiples of powers of 2 of the current layer
            (Default: 32)
        conv2d_size - The dimensions of convolutional layers. (Default: (3, 3))
        pool_size - The dimensions of max-pooling layers. (Default: (2, 2))
        n_outputs - The number of channels in the output layer. (Default: 1)
        lrelu_alpha - The alpha value of LReLU layers. (Default: 0.0)
    Outputs:
        outputs - The output layer of the network.
        hidden - Dict holding the outputs of every activation and pooling layer.
    """

    n_f = n_filters
    
    hidden = {}
    assert(n_layers % 2 == 1), 'Number of layers must be an odd number (You '\
        'entered: {}'.format(n_layers)
    midpoint = (n_layers - 1) / 2.0
    h = inputs
    cat_layers = []
    for i in range(n_layers):
        if i < midpoint:
            h = Conv2D(n_f * 2**i, conv2d_size, activation='linear', padding='same', 
                       name='conv-{}-0'.format(i))(h)
            h = LReLU(alpha=lrelu_alpha, name='lrelu-{}-0'.format(i))(h)
            hidden[h.name] = h
            cat_layers.append(h)
            
            h = Conv2D(n_f * 2**i, conv2d_size, activation='linear', padding='same', 
                       name='conv-{}-1'.format(i))(h)
            h = LReLU(alpha=lrelu_alpha, name='lrelu-{}-1'.format(i))(h)
            hidden[h.name] = h

            h = MaxPooling2D(pool_size=pool_size, name='pool-{}'.format(i))(h)
            hidden[h.name] = h
        elif i > midpoint:
            h = concatenate([UpSampling2D(size=pool_size)(h), cat_layers[n_layers - i - 1]], 
                            axis=3, name='cat-{}'.format(i))
            hidden[h.name] = h
            
            h = Conv2D(n_f * 2**(n_layers - i - 1), conv2d_size, activation='linear', padding='same', 
                       name='conv-{}-0'.format(i))(h)
            h = LReLU(alpha=lrelu_alpha, name='lrelu-{}-0'.format(i))(h)
            hidden[h.name] = h
            
            h = Conv2D(n_f * 2**(n_layers - i - 1), conv2d_size, activation='linear', padding='same', 
                       name='conv-{}-1'.format(i))(h)
            h = LReLU(alpha=lrelu_alpha, name='lrelu-{}-1'.format(i))(h)
            hidden[h.name] = h
        else:
            h = Conv2D(n_f * 2**i, conv2d_size, activation='relu', padding='same', 
                       name='conv-{}-0'.format(i))(h)
            h = LReLU(alpha=lrelu_alpha, name='lrelu-{}-0'.format(i))(h)
            hidden[h.name] = h
            
            h = Conv2D(n_f * 2**i, conv2d_size, activation='relu', padding='same', 
                       name='conv-{}-1'.format(i))(h)
            h = LReLU(alpha=lrelu_alpha, name='lrelu-{}-1'.format(i))(h)
            hidden[h.name] = h

    outputs = Conv2D(n_outputs, [1, 1], activation='linear', name='output')(h)
    return outputs, hidden
