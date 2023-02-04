import torch
from PIL import Image
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import os
from utils.arguments import args
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class all_Dataset(data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, all=False):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.all = all

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.all:
            if self.transform:
                image = self.transform(image)
                image = image.to(torch.float32)
            if self.target_transform:
                label = self.target_transform(label)
                label = label.to(torch.float32)
        return image, img_path, label

def _convert_image_to_rgb(image):
    return image.convert("RGB")


transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                _convert_image_to_rgb,
                                transforms.ToTensor(),
                                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                               ])


def split_dataset(all_data):
    num_train = len(all_data)
    indices = list(range(num_train))
    split1 = int(np.floor(0.8 * num_train))
    split2 = int(np.floor(0.9 * num_train))
    # shuffle the index
    np.random.seed(args.seed)
    np.random.shuffle(indices)

    train_idx, valid_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:]
    return train_idx, valid_idx, test_idx


