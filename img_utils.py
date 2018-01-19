def gridcrop(img, crop_dims):
    """
    Split the image into tiles of smaller images.

    Inputs:
        img - PIL Image.
        crop_dims - The dimensions of each cropped image. (Default: 
                    [img.size[0]/2, img.size[1]/2])
    Outputs:
        crops - List of crop dicts containing img and other keys.
    """

    dimensions = img.size # NOTE: Image.size is (width, height)

    crop_corners = _gridCropCorners(dimensions, crop_dims)

    # loop through crop_corners and create crop for each

    return crop_corners

def _gridCropCorners(im_dims, crop_size, stride_size=None):
    if stride_size is None:
        stride_size = crop_size

    assert(len(im_dims) == 2 
           and len(crop_size) == 2 
           and len(stride_size) == 2)

    c_indices = range(0, im_dims[0] - crop_size[0], stride_size[0])
    c_indices.append(im_dims[0] - crop_size[0])
    r_indices = range(0, im_dims[1] - crop_size[1], stride_size[1])
    r_indices.append(im_dims[1] - crop_size[1])

    crop_corners = []
    crop_ctr = 0
    for c in c_indices:
        for r in r_indices:
            crop_corners.append((c, r))

    return crop_corners