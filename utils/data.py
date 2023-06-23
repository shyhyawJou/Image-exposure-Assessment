import random
from PIL import Image
import numpy as np
from copy import deepcopy
from os.path import join
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF



class Dataset:
    def __init__(self, data_dir, label_txt, size, seed=None):
        pairs = []
        with open(label_txt) as f:
            for line in f.readlines():
                name, label = line.split()
                pairs.append((join(data_dir, name), float(label)))
        
        if seed is not None:
            random.seed(seed)
            random.shuffle(pairs)

        paths, labels = [], []
        for path, label in pairs:
            paths.append(path)
            labels.append(label)
        self.paths = paths
        self.labels = np.asarray(labels, 'float32')
        self.size = size

    def __getitem__(self, idx):
        path, label = self.paths[idx], self.labels[idx]
        img = TF.resize(Image.open(path), (self.size, self.size))
        img = TF.to_tensor(img)
        return img, label[None]

    def __len__(self):
        return len(self.paths)

    def split(self, val_ratio):
        n = int(len(self) * val_ratio)
        val_ds = deepcopy(self)
        val_ds.paths = self.paths[:n]
        val_ds.labels = self.labels[:n]
        self.paths = self.paths[n:]
        self.labels = self.labels[n:]
        return self, val_ds
    

def load_data(train_dir, 
              train_txt, 
              test_dir, 
              test_txt, 
              img_size, 
              batch_size, 
              seed):
    
    # load
    tr_ds = Dataset(train_dir, train_txt, img_size, seed)
    tr_ds, val_ds = tr_ds.split(0.2)
    test_ds = Dataset(test_dir, test_txt, img_size, seed=None)
    #from pathlib import Path as p
    #for i, path in enumerate(test_ds.paths):
    #    test_ds.paths[i] = str(p(path).with_suffix('.bmp'))

    # dataloader
    tr_ds = DataLoader(tr_ds, batch_size, True, num_workers=4)
    val_ds = DataLoader(val_ds, batch_size, False, num_workers=4)
    test_ds = DataLoader(test_ds, batch_size, False, num_workers=4)

    return tr_ds, val_ds, test_ds