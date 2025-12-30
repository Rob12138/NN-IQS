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
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--multiple', default=1)
    parser.add_argument('--inpres')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    directory = args.dataset

    for filename in os.listdir(directory):

        input = directory + filename

        img = transforms.ToTensor()(np.moveaxis(np.load(input), 0, -1))

        h = int(args.inpres) * int(args.multiple)
        w = int(args.inpres) * int(args.multiple)
        w_inp = int(args.inpres)

        img = resize_fn(img, h)
        inp = resize_fn(img, w_inp)

        model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        pred = batched_predict(model, ((inp - 0.5) / 0.5).cuda().unsqueeze(0),
            coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()

        split_img = torch.split(img, split_size_or_sections = 1, dim = 0)
        img_split = np.reshape(split_img[0], (img.shape[-2], img.shape[-1]))

        split_pred = torch.split(pred, split_size_or_sections = 1, dim = 0)
        pred_split = np.reshape(split_pred[0], (img.shape[-2], img.shape[-1]))

        one_array = np.ones((img.shape[-2], img.shape[-1]))
        pred_antisigma = -np.log(np.divide(one_array, pred_split) - one_array)
        img_antisigma = -np.log(np.divide(one_array, img_split) - one_array)
        diff_abs = np.abs(pred_antisigma - img_antisigma)
        diff = np.divide(diff_abs, img_antisigma)


        diffpath = 'datasets\\Schwinger\\test\\partial_w\\' + str(args.multiple) + '_times\\'
        #diffpath = 'datasets\\Schwinger\\test\\N12\\'
        #diffpath = 'datasets\\Schwinger\\test\\diff_outside_w\\' + str(args.multiple) + '_times\\'
        diffname = filename
        #difffull = diffpath + diffname
        difffull = diffpath + diffname[:-4] + '_diff'

        np.save(difffull, diff)
