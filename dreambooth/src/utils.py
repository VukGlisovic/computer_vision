import numpy as np
from PIL import Image


def show_images(imgs, resize=512):
    """Combines the list of images into one image to show.

    Args:
        imgs (list[PIL.Image.Image]):
        resize (int):

    Returns:
        PIL.Image.Image
    """
    n_imgs = len(imgs)
    n_cols = int(np.ceil(n_imgs ** 0.5))
    n_rows = int(np.ceil(n_imgs / n_cols))

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]

    w, h = imgs[0].size
    grid_w, grid_h = n_cols * w, n_rows * h
    grid = Image.new("RGB", size=(grid_w, grid_h))

    for i, img in enumerate(imgs):
        x = i % n_cols * w
        y = i // n_cols * h
        grid.paste(img, box=(x, y))

    return grid