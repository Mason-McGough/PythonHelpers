import os
import tensorflow as tf

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
