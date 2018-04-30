from __future__ import division
from math import sqrt
from os.path import basename, splitext, join
from PIL import Image
import deepzoom
from openslide import open_slide

try:
    from functools import reduce
except ImportError:
    pass

def get_level_with_max_pixels(src, max_pixels=50000000):
    """
    Return the largest level of Openslide-compatible image src with dimensions
    smaller than max_pixels.

    Inputs:
        src - Path to image.
        max_pixels - Maximum number of pixels in level. (default: 50000000)
    Outputs:
        opt_level - Largest level with fewer than max_pixels.
    """

    slide_dims = open_slide(src).level_dimensions

    opt_level = -1
    for i, dims in enumerate(slide_dims):
        if reduce(lambda a, b: a * b, dims) < max_pixels:
            opt_level = i
            break

    return opt_level

def convert_to_dzi(src, dest='./', level=None, max_pixels=50000000, tile_size=254,
                   tile_overlap=1, tile_format='jpg', image_quality=0.75,
                   resize_filter='bicubic', verbose=False):
    """
    Convert the image src to a dzi file and save in dest.

    Inputs:
        src - Source to the image to be converted to dzi format.
        dest - Location to save the new dzi file. (default: './')
        level - The Openslide level at which to save the dzi file. If set to none,
                it is set to the value which maximizes the number of pixels subject
                to the max_pixels size. (default: None)
        max_pixels - The maximum number of pixels allowed if level is not specified.
                     if level is specified, has no effect. (default: 50000000)

    Outputs:
        name_dzi - The full path of the new dzi file created.
    """

    if level is None:
        level = get_level_with_max_pixels(src, max_pixels)
        if level is -1:
            level = 0
            orig_dims = open_slide(src).dimensions
            k = orig_dims[0] / orig_dims[1]
            c = sqrt(max_pixels / k)
            opt_dims = (int(k * c), int(c))
        else:
            slide_dims = open_slide(src).level_dimensions
            opt_dims = slide_dims[level]
    else:
        slide_dims = open_slide(src).level_dimensions
        opt_dims = slide_dims[level]

    if verbose:
        print("Image Level: " + str(level))
        print("Dimensions: " + str(opt_dims))

    slide = open_slide(src).read_region((0, 0), level, opt_dims)

    # Create Deep Zoom Image creator with weird parameters
    creator = deepzoom.ImageCreator(tile_size=tile_size,
                                    tile_overlap=tile_overlap,
                                    tile_format=tile_format,
                                    image_quality=image_quality,
                                    resize_filter=resize_filter)

    # Create Deep Zoom image pyramid from source
    name, _ = splitext(basename(src))
    name_dzi = join(dest, name + '.dzi')
    creator.create(slide, name_dzi)

    return name_dzi

def convert_slide_image(img_path, output_path, write_kwargs={}):
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

    _, ext = splitext(output_path)
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
