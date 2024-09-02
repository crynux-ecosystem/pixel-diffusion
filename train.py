
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import matplotlib.pyplot as plt

import noise
from model import diffusion
from utils import dataset as data_utils
from utils import ckpt as ckpt_utils
from utils import platform


BATCH_SIZE = 128
LEARNING_RATE = 1e-4
SAVE_EVERY_EPOCH = 100
DEVICE = platform.get_accelerator()

STEPS = 500
ACTUAL_STEPS = 490
SIZE = 16


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    iters = size // BATCH_SIZE
    assert iters > 0

    model.train()
    tot_loss = 0
    for _, item in enumerate(dataloader):
        item = item * 2 - 1
        item = item.to(DEVICE)
        t = torch.rand(item.shape[:1])
        t = (t * ACTUAL_STEPS).long()

        noise_err, noised_image = noise.noise(item, t)
        pred = model(noised_image, t)
        loss = loss_fn(pred, noise_err, t)
        tot_loss += float(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return tot_loss / iters


def run():
    print("Loading dataset...")
    data = data_utils.ImageDataset(args.data_path, SIZE, SIZE)
    print("Loaded dataset of size", len(data))
    train_loader = data.dataloader(BATCH_SIZE)

    model = diffusion.UNet().to(DEVICE)
    epoch = 1
    if args.load_path != "":
        epoch = ckpt_utils.load_model(model, args.load_path, DEVICE)
    
    mse = nn.MSELoss()
    def loss_fn(ims, xs, ts):
        return mse(ims, xs)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # betas=(0.5, 0.999)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)  # High gamma for small dataset

    while True:
        print(f"Epoch {epoch}   lr = {scheduler.get_last_lr()}")
        loss = train(train_loader, model, loss_fn, optimizer)
        print("Average loss:", loss)
        if epoch % SAVE_EVERY_EPOCH == 0: 
            print("SAVING MODEL")
            ckpt_utils.save_model(model, epoch, f"{args.save_path}@{epoch}.pt")
        if scheduler.get_last_lr()[0] > 1e-5:
            scheduler.step()
        epoch += 1


if __name__=="__main__":
    # Handle command line arguments
    import argparse

    parser = argparse.ArgumentParser("train.py")
    parser.add_argument("-data_path", help="in format of `path/to/*.png` for all training data")
    parser.add_argument("-load_path", help="`path/to/model.pt` for the checkpoint to load. Leave empty to start from scratch.", default="", nargs='?')
    parser.add_argument("-save_path", help="`path/to/model.pt` for the checkpoint to write.", default="", nargs='?')
    args = parser.parse_args()
    assert args.save_path
    run()