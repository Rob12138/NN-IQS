import argparse
import os
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict
import random

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def resize_fn(img, size):
    x_list = random.sample(range(0, img.shape[-2]), size)
    y_list = random.sample(range(0, img.shape[-1]), size)
    x_list.sort()
    y_list.sort()
    resized_img = torch.empty((3, size, size))
    for i in range(size):
        for j in range(size):
            resized_img[0][i][j] = img[0][x_list[i]][y_list[j]]
    for i in range(size):
        for j in range(size):
            resized_img[1][i][j] = 0.000001
            resized_img[2][i][j] = 0.000001
    return resized_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--multiple', default=1)
    parser.add_argument('--orires')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(np.moveaxis(np.load(args.input), 0, -1))

    h = (int(args.multiple))*(img.shape[-1])
    w = (int(args.multiple))*(img.shape[-2])
    w_inp = img.shape[-1]

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()


    split = torch.split(pred, split_size_or_sections = 1, dim = 0)
    pred_split = split[0]
    pred_split = np.reshape(pred_split, (h, w))
    one_array = np.ones((w, h))
  
    pred_split_antisigma = -np.log(np.divide(one_array, pred_split) - one_array)

    #pred_path = 'C:\\Users\\robwa\\liif-main\\datasets\\Schwinger\\test\\low_resolution_upscale_demo\\'
    pred_path = 'C:\\Users\\robwa\\liif-main\\datasets\\Schwinger\\test\\rainbow\\'
    pred_name = args.output
    pred_full = pred_path + pred_name
    np.save(pred_full, pred_split_antisigma)
