import os, warnings, errno
import numpy as np
from imageio import imread, imwrite
from skimage import color
from .file_utils import get_nested_dirs, list_files

def grid_crop(img, crop_dims, stride_size=None, include_excess=True):
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
    Outputs:
        crop_imgs - List of crop dicts containing img and other keys.
    """

    img_dims = img.shape # NOTE: (rows, cols)

    crop_corners = _grid_crop_corners(img_dims, crop_dims, stride_size, include_excess)

    # loop through crop_corners and create crop for each
    crop_imgs = []
    for corner in crop_corners:
        idxs = (corner[0], corner[0] + crop_dims[0],
                corner[1], corner[1] + crop_dims[1])
        try:
            crop_img = img[idxs[0]:idxs[1], idxs[2]:idxs[3], :]
        except IndexError:
            crop_img = img[idxs[0]:idxs[1], idxs[2]:idxs[3]]
        crop = {'img': crop_img,
                'corner': corner}
        crop_imgs.append(crop)

    return crop_imgs

def _grid_crop_corners(im_dims, crop_size, stride_size=None, include_excess=True):
    if stride_size is None:
        stride_size = crop_size

    assert(len(crop_size) == 2 
           and len(stride_size) == 2)

    r_indices = range(0, im_dims[0] - crop_size[0], stride_size[0])
    c_indices = range(0, im_dims[1] - crop_size[1], stride_size[1])
    if include_excess:
        r_indices.append(im_dims[0] - crop_size[0])
        c_indices.append(im_dims[1] - crop_size[1])

    crop_corners = []
    crop_ctr = 0
    for r in r_indices:
        for c in c_indices:
            crop_corners.append((r, c))

    return crop_corners

def stitch_crops(crop_imgs, method='average'):
    """
    Merge a list of regularly-spaced cropped images into one single image.

    Inputs:
        crop_imgs - List of crop dicts containing the following keys:
                        'img' - Numpy array with image.
                        'corner' - Tuple specifying the location of the upper-left
                                   corner of the image.
        method - Blend method to combine two images. Options are:
                    'average'
                    'and'
                    'or'
    Outputs:
        img - Numpy array with stitched image.
    """

    # determine type and number of channel of crops
    mode = crop_imgs[0]['img'].dtype
    try:
        n_channels = crop_imgs[0]['img'].shape[2]
    except IndexError:
        n_channels = 1

    # check that all crops have same number of channels
    for crop in crop_imgs:
        try:
            crop_n_channel = crop['img'].shape[2]
        except IndexError:
            crop_n_channel = 1

        if crop_n_channel != n_channels:
            raise Exception("Number of channels is not consistent between images.")

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
        if crop['img'].shape[0] > max_dims[0]:
            max_dims[0] = crop['img'].shape[0]

        if crop['img'].shape[1] > max_dims[1]:
            max_dims[1] = crop['img'].shape[1]

        if crop['corner'][0] > max_corner[0]:
            max_corner[0] = crop['corner'][0]

        if crop['corner'][1] > max_corner[1]:
            max_corner[1] = crop['corner'][1]
    img_dims = (max_corner[0] + max_dims[0], max_corner[1] + max_dims[1])

    # create numpy array to hold crops
    img = np.zeros((img_dims[0], img_dims[1], n_channels), dtype=int)

    # stitch image into numpy array
    for crop in crop_imgs:
        idxs = (crop['corner'][0], crop['corner'][0] + crop['img'].shape[0], 
                crop['corner'][1], crop['corner'][1] + crop['img'].shape[1])
        img_section = img[idxs[0]:idxs[1], idxs[2]:idxs[3], :]

        if len(crop['img'].shape) < 3:
            crop['img'] = np.expand_dims(crop['img'], 2)

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

        img[idxs[0]:idxs[1], idxs[2]:idxs[3], :] = crop_merged

    return img

from PIL import Image
from openslide import open_slide
def convert_image(img_path, output_path, write_kwargs={}):
    """
    Load image in img_path and rewrite it to output_path, converting type if needed.

    Inputs:
        img_path - Path to image.
        output_path - Path to write image to (extension determines format).
        write_kwargs - kwargs to pass to the write function
    Outputs:
        None
    """

    openslide_formats = ('.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn',
                         '.mrxs', '.svslide', '.bif')
    image_formats = ('.jpg', '.png', '.tif', '.bmp')

    _, ext = os.path.splitext(output_path)
    if ext in openslide_formats:
        slide = open_slide(img_path)
        try:
            img = slide.read_region((0, 0), 0, slide.dimensions)
        except:
            print("Loading image failed: " + output_path)
    elif ext in image_formats:
        img = Image.open(img_path)
    else:
        raise Exception('Unsupported format: ' + ext)

    if output_path != img_path:
        try: 
            img.save(output_path, **write_kwargs)
            print("Image saved: " + output_path)
        except IOError:
            print("Conversion failed: " + img_path)

def grid_crop_images(src_dir, dest_dir, crop_dims, recursive=True, stride_size=None, 
                   output_ext=".jpg", write_kwargs={}, verbose=False):
    """
    Apply grid_crop to all images within a given list.

    Inputs:
        src_dir - Directory where images are stored.
        dest_dir - Directory to output folders of subcropped images (preserves 
                   structure of source directory).
        crop_dims - The dimensions of each cropped image.
        recursive - If true, searches src_dir and all child folders.
        stride_size - The offset between each cropped image, in pixels. (Default: 
                      crop_dims)
        output_ext - Extension to save output files in.
        write_kwargs - kwargs for imageio.imwrite.
        verbose - If true, prints status updates.
    Outputs:
        None
    """

    # get all files
    file_list = list_files(src_dir, recursive=recursive)

    if verbose:
        print("number of imgs: %d" % len(file_list))

    # subcrop images to directory
    len_src_dir = len(src_dir)
    for img_path in file_list:

        # create destination if does not exist
        full_d = os.path.join(dest_dir, img_path[len_src_dir:])
        try: 
            os.makedirs(full_d)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if verbose:
            print("Reading image: %s" % img_path)
        img = imread(img_path)

        crops = grid_crop(img, crop_dims, stride_size)

        if verbose:
            print("Saving %d subcrops in: %s" % (len(crops), full_d))
        for crop in crops:
            r = crop['corner'][0]
            c = crop['corner'][1]
            filename = "crop-" + str(r) + '-' + str(c) + output_ext
            filepath = os.path.join(full_d, filename)

            try:
                imwrite(filepath, crop['img'], **write_kwargs)
            except IOError:
                print("cannot convert", crop['img'])

def stitch_images(src_dir, dest_dir, recursive=True, output_ext='.png', method='or', verbose=False):
    """
    Apply stitch_crops to all images grouped within a set of directories.

    Inputs:
        src_dir - Directory where image directories (one directory -> one image).
        dest_dir - Directory to output stitched images (preserves structure of 
                   source directory).
        recursive - If true, searches src_dir and all child folders.
        output_ext - Extension to save output files in.
        method - Stitching method (see stitch_crops).
        verbose - If true, prints status updates.
    Outputs:
        None
    """

    # get all directories within src_dir
    dir_list = list_files(src_dir, 
                          return_dirs=True, 
                          return_files=False, 
                          recursive=recursive)

    if verbose:
        print("number of dirs: %d" % len(dir_list))

    # subcrop images to directory
    len_output_ext = len(output_ext)
    len_src_dir = len(src_dir)
    for d in dir_list:
        if verbose:
            print("Current dir: %s" % d)

        # create destination if does not exist
        filepath = dest_dir + d[len_src_dir:]
        try: 
            os.makedirs(os.path.dirname(filepath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        img_list = list_files(d)
        if verbose:
            print("Number of imgs: %d" % len(img_list))

        if len(img_list) == 0:
            continue

        crop_list = []
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            r = int(img_name.split('-')[1])
            c = int(img_name.split('-')[2][:-len_output_ext])

            img = color.rgb2gray(imread(img_path)) > (65535 / 2.0)
            crop_dict = {'img': img,
                         'corner': (r, c)}
            crop_list.append(crop_dict)

        img = stitch_crops(crop_list, method=method)
        
        out_path = filepath + output_ext
        try:
            imwrite(out_path, img)
        except IOError:
            print("cannot convert", img)