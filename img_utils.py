import warnings
import numpy as np
from PIL import Image

def gridCrop(img, crop_dims, stride_size=None, include_excess=True):
    """
    Split the image into tiles of smaller images.

    Since the image dimensions may not be perfect multiples of the crop_dims, the 
    edge portions of the image are included as crops positioned at img.size - 
    crop_size. This results in the edge images overlapping slightly by default. This
    behavior can be overridden by setting 'include_excess' to False.

    Inputs:
        img - PIL Image.
        crop_dims - The dimensions of each cropped image.
        stride_size - The offset between each cropped image, in pixels. (Default: 
                    crop_dims)
        include_excess - When true, includes extra crops to include edges that would
                         exceed the largest multiple of crop_dims within the image.
                         (Default: True)
    Outputs:
        crop_imgs - List of crop dicts containing img and other keys.
    """

    img_dims = img.size # NOTE: Image.size is (width, height)

    crop_corners = _gridCropCorners(img_dims, crop_dims, stride_size, include_excess)

    # loop through crop_corners and create crop for each
    crop_imgs = []
    for corner in crop_corners:
        box = (corner[0], 
               corner[1], 
               corner[0] + crop_dims[0], 
               corner[1] + crop_dims[1])
        crop = {'img': img.crop(box),
                'corner': corner}
        crop_imgs.append(crop)

    return crop_imgs

def _gridCropCorners(im_dims, crop_size, stride_size=None, include_excess=True):
    if stride_size is None:
        stride_size = crop_size

    assert(len(im_dims) == 2 
           and len(crop_size) == 2 
           and len(stride_size) == 2)

    c_indices = range(0, im_dims[0] - crop_size[0], stride_size[0])
    r_indices = range(0, im_dims[1] - crop_size[1], stride_size[1])
    if include_excess:
        c_indices.append(im_dims[0] - crop_size[0])
        r_indices.append(im_dims[1] - crop_size[1])

    crop_corners = []
    crop_ctr = 0
    for c in c_indices:
        for r in r_indices:
            crop_corners.append((c, r))

    return crop_corners

def stitchCrops(crop_imgs, method='average'):
    """
    Merge a list of regularly-spaced cropped images into one single image.

    Inputs:
        crop_imgs - List of crop dicts containing the following keys:
                        'img' - Numpy array with image.
                        'corner' - Tuple specifying the location of the upper-left
                                   corner of the image.
        method - Blend method to combine two images. Options are:
                    'average' (Default)
                    'and'
                    'or'
    Outputs:
        img - Numpy array with stitched image.
    """

    # determine type of crops
    mode = crop_imgs[0]['img'].dtype

    # check that all crops are of same type
    for crop in crop_imgs:
        if crop['img'].dtype != mode:
            warnings.warn("Data types of images are not consistent. May produce \
                           unpredictable results.")
            break

    # find dimensions of original image
    max_dims = [0, 0]
    max_corner = [0, 0]
    for crop in crop_imgs:
        if crop['img'].size[0] > max_dims[0]:
            max_dims[0] = crop['img'].size[0]

        if crop['img'].size[1] > max_dims[1]:
            max_dims[1] = crop['img'].size[1]

        if crop['corner'][0] > max_corner[0]:
            max_corner[0] = crop['corner'][0]

        if crop['corner'][1] > max_corner[1]:
            max_corner[1] = crop['corner'][1]
    img_dims = (max_corner[0] + max_dims[0], max_corner[1] + max_dims[1])

    # create numpy array to hold crops
    img = np.zeros((img_dims[1], img_dims[0]), dtype=int)

    # stitch image into numpy array
    for crop in crop_imgs:
        idxs = (crop['corner'][1], crop['corner'][1] + crop['img'].size[1], 
                crop['corner'][0], crop['corner'][0] + crop['img'].size[0])
        img_section = img_np[idxs[0]:idxs[1], idxs[2]:idxs[3]]

        # blend method
        if method == 'or':
            crop_merged = np.bitwise_or(img_section, crop['img'])
        elif method == 'and':
            crop_merged = np.bitwise_and(img_section, crop['img'])
        elif method == 'xor':
            crop_merged = np.bitwise_xor(img_section, crop['img'])
        elif method == 'average':
            crop_merged = (img_section + crop['img']) / 2.0
        else:
            warnings.warn("Invalid method. Reverting to 'average'. Your method: %s" 
                          % method, UserWarning)
            crop_merged = (img_section + crop['img']) / 2.0

        img[idxs[0]:idxs[1], idxs[2]:idxs[3]] = crop_merged

    return img
