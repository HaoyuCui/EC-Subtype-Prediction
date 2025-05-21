"""
In the im4MEC, it used atmost 2048 tiles in each slide
"""

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class MoCoDataset(Dataset):
    def __init__(self, data_dir, max_tiles=2048):
        self.data_dir = data_dir
        self.transform = transform
        self.max_tiles = max_tiles

        self.queue = []

        for slide_dir in os.listdir(data_dir):
            slide_path = os.path.join(data_dir, slide_dir)
            if os.path.isdir(slide_path):
                count = 0
                for tile in os.listdir(slide_path):
                    tile_path = os.path.join(slide_path, tile)
                    if os.path.isfile(tile_path):
                        self.queue.append(tile_path)
                        count += 1
                        if count >= self.max_tiles:
                            break

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, idx):
        tile_path = self.queue[idx]
        img = self.transform(Image.open(tile_path).convert('RGB'))
        return img



