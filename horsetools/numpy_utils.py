import numpy as np

def subsample(x, n_samples, return_choices=False, replace=False):
    """
    Randomly sample a subset of x.

    Inputs:
        x - The array to sample from.
        n_samples - The number of samples to select from x.
        return_choices - If True, returns an array of the random indices used to
            construct output x. (Default: False)
        replace - If True, samples are taken with replacement. (Default: False)
    Outputs:
        x - The subsampled array.
    """

    choices = np.random.choice(range(x.shape[0]), size=[n_samples], replace=replace)
    
    if return_choices:
        return x[choices], choices
    else:
        return x[choices]

def describe(x):
    """
    Print the shape, min, max, and datatype of array.

    Inputs:
        x - Numpy-like array to be described.
    Outputs:
        None
    """

    print('{}, {}, {}'.format(x.shape, [np.min(x), np.max(x)], x.dtype))
