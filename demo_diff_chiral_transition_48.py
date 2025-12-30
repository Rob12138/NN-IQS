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
import math
from scipy import interpolate


def interpo(a, b, c, d, x1, x2, y1, y2, x, y):
    first_term = 1/((y2-y1)*(x2-x1))
    second_term = b*(x-x1)*(y-y1)
    third_term = a*(x2-x)*(y-y1)
    fourth_term = d*(x-x1)*(y2-y)
    fifth_term = c*(x2-x)*(y2-y)
    return first_term*(second_term + third_term + fourth_term + fifth_term)

def find_parameter(x, y, chiral_mat, size):
    x_width = (g*1.4 - 0)/(size - 1)
    y_width = (g*2.5 - 0.1)/(size - 1)
    x_count = math.trunc((x - 0)/x_width)
    y_count = math.trunc((y - 0.1)/y_width)
    array = np.array(chiral_mat)
    if x_count != (size-1) and y_count != (size-1):
        x1 = x_width*x_count
        x2 = x1 + x_width
        y1 = 0.1 + y_width*y_count
        y2 = y1 + y_width
        a = array[x_count][y_count + 1]
        b = array[x_count + 1][y_count + 1]
        c = array[x_count][y_count]
        d = array[x_count + 1][y_count]
    elif x_count == (size-1) and y_count == (size-1):
        x1 = x_width*(x_count - 1)
        x2 = x1 + x_width
        y1 = 0.1 + y_width*(y_count - 1)
        y2 = y1 + y_width
        a = array[x_count - 1][y_count]
        b = array[x_count][y_count]
        c = array[x_count - 1][y_count - 1]
        d = array[x_count][y_count - 1]
    elif x_count == (size-1) and y_count != (size-1):
        x1 = x_width*(x_count - 1)
        x2 = x1 + x_width
        y1 = 0.1 + y_width*y_count
        y2 = y1 + y_width
        a = array[x_count - 1][y_count + 1]
        b = array[x_count][y_count + 1]
        c = array[x_count - 1][y_count]
        d = array[x_count][y_count]
    else:
        x1 = x_width*x_count
        x2 = x1 + x_width
        y1 = 0.1 + y_width*(y_count - 1)
        y2 = y1 + y_width
        a = array[x_count][y_count]
        b = array[x_count + 1][y_count]
        c = array[x_count][y_count - 1]
        d = array[x_count + 1][y_count - 1]

    return a, b, c, d, x1, x2, y1, y2

def get_coordinate(i, j, orires):
    x = ((g*1.4 - 0)/(orires - 1))*i
    y = ((g*2.5 - 0.1)/(orires - 1))*j + 0.1
    return x, y


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

        
        transition_error = []
        input = directory + filename

        img = transforms.ToTensor()(np.moveaxis(np.load(input), 0, -1))

        h = int(args.inpres) * int(args.multiple)
        w = int(args.inpres) * int(args.multiple)
        w_inp = int(args.inpres)

        img = resize_fn(img, h)
        inp = resize_fn(img, w_inp)

        '''
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
        max_value = np.max(np.array(img_antisigma))
        min_value = np.min(np.array(img_antisigma))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (img_antisigma[i][j] - min_value)/(max_value - min_value) > 0.4 and (img_antisigma[i][j] - min_value)/(max_value - min_value) <0.6:
                    transition_error.append(diff[i][j])


        #diffpath = 'C:\\Users\\robwa\\liif-main\\datasets\\Schwinger\\test\\diff_input48\\' + str(args.multiple) + '_times_transition\\'
        diffpath = 'C:\\Users\\robwa\\liif-main\\datasets\\Schwinger\\test\\partial_w\\' + str(args.multiple) + '_times_transition\\'
        diffname = filename
        difffull = diffpath + diffname
        np.save(difffull, np.array(transition_error))

   
        interpo_transition_error = []
        g = 3
    
        interpo_array = np.empty((h, w))

        inp_split = np.reshape(inp[0], (inp.shape[-2], inp.shape[-1]))

        one_array_inp = np.ones((w_inp, w_inp))

        inp_antisigma = -np.log(np.divide(one_array_inp, inp_split) - one_array_inp)

        for i in range(0, h):
            for j in range(0, w):
                x, y = get_coordinate(i, j, h)
                a, b, c, d, x1, x2, y1, y2 = find_parameter(x, y, inp_antisigma, w_inp)
                interpo_array[i][j] = interpo(a, b, c, d, x1, x2, y1, y2, x, y)

        pred = interpo_array

        split_img = torch.split(img, split_size_or_sections = 1, dim = 0)
        img_split = np.reshape(split_img[0], (img.shape[-2], img.shape[-1]))


        one_array = np.ones((img.shape[-2], img.shape[-1]))
        img_antisigma = -np.log(np.divide(one_array, img_split) - one_array)
        diff_abs = np.abs(pred - np.array(img_antisigma))
        diff_interpo = np.divide(diff_abs, np.array(img_antisigma))

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (img_antisigma[i][j] - min_value)/(max_value - min_value) > 0.4 and (img_antisigma[i][j] - min_value)/(max_value - min_value) <0.6:
                    interpo_transition_error.append(diff_interpo[i][j])

        '''
        
        g = 3
        
        inp_split = np.reshape(inp[0], (inp.shape[-2], inp.shape[-1]))

        one_array_inp = np.ones((w_inp, w_inp))

        inp_antisigma = -np.log(np.divide(one_array_inp, inp_split) - one_array_inp)

        x_inp = []
        y_inp = []

        for i in range(0, w_inp):
            for j in range(0, w_inp):
                x, y = get_coordinate(i, j, w_inp)
                if x not in x_inp:
                    x_inp.append(x)
                if y not in y_inp:
                    y_inp.append(y)


        f_cubic = interpolate.interp2d(x_inp, y_inp, inp_antisigma, kind='cubic')
        #f_cubic = interpolate.RectBivariateSpline(x_inp, y_inp, inp_antisigma)

        x_cubic = []
        y_cubic = []

        for i in range(0, h):
            for j in range(0, w):
                x, y = get_coordinate(i, j, h)
                if x not in x_cubic:
                    x_cubic.append(x)
                if y not in y_cubic:
                    y_cubic.append(y)
                        

        pred = f_cubic(x_cubic, y_cubic)

        split_img = torch.split(img, split_size_or_sections = 1, dim = 0)
        img_split = np.reshape(split_img[0], (img.shape[-2], img.shape[-1]))


        one_array = np.ones((img.shape[-2], img.shape[-1]))
        img_antisigma = -np.log(np.divide(one_array, img_split) - one_array)
        diff_abs = np.abs(np.array(pred) - np.array(img_antisigma))
        diff_cubic = np.divide(diff_abs, np.array(img_antisigma))
    
        cubic_transition_error = []

        max_value = np.max(np.array(img_antisigma))
        min_value = np.min(np.array(img_antisigma))

        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (img_antisigma[i][j] - min_value)/(max_value - min_value) > 0.4 and (img_antisigma[i][j] - min_value)/(max_value - min_value) <0.6:
                    cubic_transition_error.append(diff_cubic[i][j])
        
        
        #diffpath = 'datasets\\Schwinger\\test\\diff_input48\\' + str(args.multiple) + '_times_transition_interpo\\'
        diffpath = 'datasets\\Schwinger\\test\\partial_w\\' + str(args.multiple) + '_times_transition_cubic\\'
        diffname = filename
        difffull = diffpath + diffname

        np.save(difffull, np.array(cubic_transition_error))
