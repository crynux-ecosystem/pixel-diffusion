import torch
from tqdm import tqdm
import noise
import random
from utils import platform


SIZE = 16
NUM_CHANNELS = 3
ACTUAL_STEPS = 490
DEVICE = platform.get_accelerator()


# Goes from x(t) -> x(t-1)
#  noise_mul: Multiplier on top of getσ(t). Higher values lead to more chaotic images
#  classifier_func: Output of clip_grad_func or classifier_to_grad_func
#  classifier_mul_func: Classifier multiplier function
def sample_step(model, im, t, noise_mul = 8, classifier_func = None, classifier_mul_func = None):
    with torch.no_grad():
        z = torch.normal(torch.zeros_like(im))
        if t == 1:
            z *= 0

        ts = torch.Tensor([t] * im.shape[0]).to(DEVICE)
        noise_err = model(im, ts.to("cpu"))
        alpha = 1 - noise.beta(t)
        new_mean = (alpha**-0.5) * (im - (1 - alpha) / ((1 - noise.ALPHA[t]) ** 0.5) * noise_err)

        # if t==1 don't attempt to use classifier guidance
        if classifier_func != None and t != 1:
            grad = classifier_func(im, ts)
            if classifier_mul_func is None:
                new_mean += grad * noise.beta(t)
            else:
                new_mean += grad * classifier_mul_func(t) # * getσ_classifier(t)

        add_noise = noise.beta(t) * z * noise_mul
        im = new_mean + add_noise      # add random noise on
        return im

# Sample N images from the model.
#  display_count: number of times to display intermediate result
def sample(model, num, display_count = 4, noise_mul = 6):
    with torch.no_grad():
        # Initial samples
        size = (num, NUM_CHANNELS, SIZE, SIZE)
        h = torch.normal(torch.zeros(size), 1).float().to(DEVICE)
        s = ACTUAL_STEPS // display_count if display_count != 0 else ACTUAL_STEPS*5

        for t in tqdm(range(ACTUAL_STEPS, 0, -1)):
            if t == 1:
                seed = torch.seed()
                state = random.getstate()
                verify = sample_step(model, h, t, noise_mul)

                torch.manual_seed(seed)
                random.setstate(state)
                h = sample_step(model, h, t, noise_mul)
                print(torch.sum(verify - h))
            else:
                h = sample_step(model, h, t, noise_mul)
        # -1..1 -> 0..1
        return (h+1)/2

if __name__=="__main__":
    import argparse
    from utils import ckpt
    from utils import image as image_util
    from model import diffusion
    import numpy as np
    from matplotlib import image

    parser = argparse.ArgumentParser("sample.py")
    parser.add_argument("-model", help="Path to the model.")
    parser.add_argument("-num", help="Number of samples.", type=int, default=10)
    parser.add_argument("-output", help="Output PNG with all candidates.", default="", nargs='?')
    parser.add_argument("-noise_mul", help="Standard deviation during sampling. Larger values lead to more chaotic samples. Default: 8.0", default=8, nargs='?', type=float)
    args = parser.parse_args()

    model = diffusion.UNet().to(DEVICE).eval()
    epoch = ckpt.load_model(model, args.model, DEVICE)
    xs = sample(model, args.num, display_count=0, noise_mul=args.noise_mul)

    sheet = image_util.show(xs)
    if args.output != "":
        image.imsave(args.output, sheet)
    input("Press enter...")
