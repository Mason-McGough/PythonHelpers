import deepzoom
from os.path import basename, splitext, join
from openslide import open_slide

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

    slide_dims = open_slide(SOURCE).level_dimensions

    opt_level = -1
    for i, dims in enumerate(slide_dims):
        if reduce(lambda a, b: a*b, dims) < max_pixels:
            opt_level = i
            break

    return opt_level

def convert_to_dzi(src, dest='./', level=None, max_pixels=50000000, tile_size=254,
                   tile_overlap=1, tile_format='jpg', image_quality=0.75,
                   resize_filter='bicubic'):
    """
    Convert the image src to a dzi file and save in dest.
    """

    if level is None:
        level = get_level_with_max_pixels(src, max_pixels)

    slide_dims = open_slide(src).level_dimensions
    slide = open_slide(src).read_region((0, 0), level, slide_dims[level])

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
