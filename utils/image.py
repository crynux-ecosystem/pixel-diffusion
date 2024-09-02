import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random

# Returns a float array of shape C x H x W, normalize to 0..1
def load_image(path : str, w: int = -1, h: int = -1, show=False):
    data = Image.open(path).convert("RGBA")
    if w > 0 and h > 0:
        data = data.resize((w, h), Image.Resampling.LANCZOS)
    background = Image.new("RGBA", data.size, (255, 255, 255))
    data = Image.alpha_composite(background, data)
    if show:
        data.show()
    data = np.asarray(data).astype(np.float32) / 255
    data = data[:, :, :3]

    # (H, W, C) -> (C, H, W)
    data = np.transpose(data, (2, 0, 1))
    return data



def shift(data, x, y):
    # (C, H, W) -> (H, W, C)
    data = np.transpose(data, (1, 2, 0))
    data = np.roll(data, x, axis=1)
    data = np.roll(data, y, axis=0)
    fill = np.array([1,1,1])

    # Fill the shifted edge in white
    if x > 0:
        data[:,:x,:]=fill
    if x < 0:
        data[:,x:,:]=fill
    if y > 0:
        data[:y,:,:]=fill
    if y < 0:
        data[y:,:,:]=fill

    # (H, W, C) -> (C, H, W)
    return np.transpose(data, (2, 0, 1))

def perturb(data):
    if random.random()<0.5:
        data = np.flip(data, axis=2)
    else:
        data = np.flip(data, axis=1)
    shifts = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
    return shift(data, *random.choice(shifts))


# Creates a spritesheet from images with size 'size'
def make_spritesheet(ims, fix_width=None, fix_height=None):
    assert fix_width is None or fix_height is None
    
    size, size_, _ = ims[0].shape
    assert size == size_, "Spritesheet: input images must be square"

    # Calculate (width, height) of spritesheet
    if fix_width is not None:
        X = fix_width
        Y = len(ims)//X
        while X*Y < len(ims): Y+=1
    elif fix_height is not None:
        Y = fix_height
        X = len(ims)//Y
        while X*Y < len(ims): X+=1
    else:
        Y = int(len(ims) ** 0.5)
        X = len(ims) // Y
        while X*Y < len(ims): X+=1

    final = np.zeros((Y*size, X*size, 3), dtype=np.float32)

    ind = 0
    for im in ims:
        x = ind // X
        y = ind % X
        fx = size * x
        fy = size * y
        ind += 1
        final[fx:fx+size, fy:fy+size, :] = im
    
    return final


def show(data):
    # Convert
    if type(data) == torch.Tensor:
        data = data.detach().cpu().numpy()
    data = [np.transpose(i, (1, 2, 0)) for i in data]
    sheet = make_spritesheet(data, fix_width=None, fix_height=None)
    sheet = np.clip(sheet, 0, 1)

    # Render
    plt.grid(False)
    plt.axis('off')
    plt.imshow(sheet)
    plt.show(block=False)
    plt.pause(0.1)
    return sheet
