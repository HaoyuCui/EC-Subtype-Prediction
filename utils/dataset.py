import torch

from torch.utils.data import Dataset
from torchvision.transforms import v2
import pandas as pd

from PIL import Image


transform_train = v2.Compose([
     v2.PILToTensor(),
     v2.RandomHorizontalFlip(),
     v2.RandomVerticalFlip(),
     # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
     v2.ToDtype(torch.float32, scale=True),
     v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
 ])


transform_val = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class PathologyDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.data = pd.read_csv(f'{mode}.csv')

        self.name = self.data['name']
        self.path = self.data['path']
        self.y_data = self.data['label']
        self.transform = transform_val

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # load tile
        name = self.name.iloc[index]
        path = self.path.iloc[index]

        tile = self.transform(Image.open(path))

        # load label
        label = self.y_data.iloc[index]
        label = torch.as_tensor(label, dtype=torch.long)

        if self.mode == 'train':
            return tile, label
        elif self.mode == 'val':
            return name, tile, label
        return None


class PathologyDatasetKFold(Dataset):
    def __init__(self,  mode, fold):
        self.mode = mode
        self.data = pd.read_csv(f'kf/{fold}_{mode}.csv')

        self.name = self.data['name']
        self.path = self.data['path']
        self.y_data = self.data['label']
        self.transform = transform_val

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # load tile
        name = self.name.iloc[index]
        path = self.path.iloc[index]

        tile = self.transform(Image.open(path))

        # load label
        label = self.y_data.iloc[index]
        label = torch.as_tensor(label, dtype=torch.long)

        if self.mode == 'train':
            return tile, label
        elif self.mode == 'val':
            return name, tile, label
        return None
