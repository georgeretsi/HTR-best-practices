# basic preprocessing for loading a text image.
import skimage.io as img_io
import skimage.color as img_color
from skimage.transform import resize
import numpy as np

def load_image(image_path):

    # read the image
    image = img_io.imread(image_path)

    # convert to grayscale skimage
    if len(image.shape) == 3:
        image = img_color.rgb2gray(image)
    
    # normalize the image
    image = 1 - image / 255.

    return image


def preprocess(img, input_size, border_size=8):
    
    h_target, w_target = input_size

    n_height = min(h_target - 2 * border_size, img.shape[0])
    
    scale = n_height / img.shape[0]
    n_width = min(w_target - 2 * border_size, int(scale * img.shape[1]))

    img = resize(image=img, output_shape=(n_height, n_width)).astype(np.float32)

    # right pad image to input_size
    img  = np.pad(img, ((border_size, h_target - n_height - border_size), (border_size, w_target - n_width - border_size),),
                                                    mode='median')

    return img



