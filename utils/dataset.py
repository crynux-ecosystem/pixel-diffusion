import torch
from torch.utils.data import Dataset
import torch.utils.data as dutils
from glob import glob
from utils import image

class ImageDataset(Dataset):
    def __init__(self, data_path, w, h):
        """
        Read all images under data_path, and store in a list of W x H x C numpy array
        """

        self.data = []
        
        print("Reading images... ", end="")
        cnt = 0
        for path in glob(data_path):
            self.data.append(image.load_image(path, w, h, cnt < 3))
            cnt += 1

        print(f"Done: {len(self.data)} data")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        res = image.perturb(self.data[idx])
        res = torch.Tensor(res)
        return res


    def dataloader(self, batch_size):
        return dutils.DataLoader(self, batch_size=batch_size, shuffle=True)