from torchvision import transforms
from torchvision.transforms.functional import adjust_gamma
import numpy as np
from PIL import Image


class MyGammaSquare:
    def __init__(self, gamma1, gamma2, thr):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.thr = thr

    def __call__(self, PIL_img):
        light_img = transforms.functional.adjust_gamma(PIL_img, self.gamma1, self.thr)
        enhanced_img = transforms.functional.adjust_gamma(PIL_img, self.gamma2, self.thr)
        mask = np.zeros((PIL_img.height, PIL_img.width), dtype=np.uint8) + 255
        bw = 0.25  # identify border width and height as fraction of image size
        bx = int(mask.shape[1] * bw)  # get the x dimension border width
        by = int(mask.shape[0] * bw)  # get the y dimension border height
        mask[bx:-bx, by:-by] = 0  # create a mask with 255 for border and 0 for inside
        out = np.where(mask == (255, 255, 255), enhanced_img, light_img)
        return out
