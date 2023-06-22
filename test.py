import argparse
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional as TF



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', help='image path')
    parser.add_argument('-model', help='model path')
    args = parser.parse_args()
    return args


def test():
    args = get_args()

    model = torch.load(args.model, 'cpu').eval()
    img = Image.open(args.img)
    x = TF.resize(img, (224, 224))
    x = TF.to_tensor(x)[None]
    exposure = np.clip(round(model(x).item(), 4), -1, 1)
    print(f'exposure:', exposure)



if __name__ == '__main__':
    test()