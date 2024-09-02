import torch
import os

def save_model(model, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }, path)

def load_model(model, path, device):
    if not os.path.exists(path):
        print(f"Error loading model: path \"{path}\" does not exist. Starting from new model instead.")
        return 1

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    
    model.train()
    print("Loaded model from epoch", epoch)

    return epoch
