# encoding utf-8
import cv2
import numpy
import os


def find_bars(img):
    black_bar = []
    white_bar = []
    w_b = 0
    for i, p in enumerate(img):
        if p > 0 and w_b == 0:
            w_start = i
            b_end = i
            w_b = 1
            if not i == 0:
                black_bar.append(int((b_start + b_end) / 2))
        elif p == 0 and w_b == 1:
            b_start = i
            w_end = i
            w_b = 0
            white_bar.append([w_start,w_end])
        if i == img.shape[0] - 1:
            white_bar.append([w_start,i])
    return numpy.asarray(white_bar), numpy.asarray(black_bar)


def Adjusting_length(img, new_img, img_Contrast):
    constrast_length = img_Contrast.shape[0]
    img_length = img.shape[0]
    if constrast_length == img_length:
        return img, new_img
    elif img_length > constrast_length:
        sub_length = img_length - constrast_length
        for i in range(sub_length):
            sub_position = int((sub_length - i - 1) * img_length / (img_length - constrast_length))
            while new_img[sub_position] == 0:
                sub_position += 1
            img = numpy.delete(img, sub_position)
            new_img = numpy.delete(new_img, sub_position)
    elif img_length < constrast_length:
        add_length = constrast_length - img_length
        for i in range(add_length):
            add_position = int(i * img_length / (constrast_length - img_length))
            img = numpy.insert(img, add_position, img[add_position])
            new_img = numpy.insert(new_img, add_position, new_img[add_position])
    return img, new_img


def Adjusting_bar(img, new_img, black_bar, white_bar, black_bar_Contrast, img_Contrast):
    if not len(black_bar) == len(black_bar_Contrast):
        raise ("BAR Detective ERROR")
    adjust_list = []
    for i in range(len(black_bar)):
        bar = black_bar[i]
        bar_contrast = black_bar_Contrast[i]
        if bar == bar_contrast:
            pass
        elif bar > bar_contrast:
            for i in range(len(white_bar)):
                if white_bar[i] > bar:
                    add_position = white_bar[i - 1]
            for i in range(bar - bar_contrast):
                img = numpy.insert(img, add_position, img[add_position])
                new_img = numpy.insert(new_img, add_position, new_img[add_position])
                adjust_list.append(add_position)
        elif bar < bar_contrast:
            for i in range(len(white_bar)):
                if white_bar[i] > bar:
                    sub_position = white_bar[i - 1]
            for i in range(bar_contrast - bar):
                img = numpy.delete(img, sub_position)
                new_img = numpy.delete(new_img, sub_position)
                adjust_list.append(-1 * sub_position)
    adjust_ending = img.shape[0] - img_Contrast.shape[0]
    if adjust_ending > 0:
        img = img[:img.shape[0] + adjust_ending]
        new_img = new_img[:img.shape[0] + adjust_ending]
    else:
        while adjust_ending < 0:
            img = numpy.insert(img, -1, img[-1])
            new_img = numpy.insert(new_img, -1, new_img[-1])
            adjust_ending += 1

    adjust_position = adjust_list, adjust_ending
    return img, new_img, adjust_position







def transfer(img,new_img, adjust_position):
    adjust_list, adjust_ending = adjust_position
    for i in adjust_list:
        if i > 0:
            if new_img[i]==0:
                pass
            img = numpy.insert(img, i, img[i])
        if i < 0:
            if new_img[i] == 0:
                pass
            new_img = numpy.delete(new_img, -1 * i)
    if adjust_ending > 0:
        img = img[:img.shape[0] + adjust_ending]
    elif adjust_ending==0:
        pass
    else:
        while adjust_ending < 0:
            img = numpy.insert(img, -1, img[-1])
            adjust_ending += 1
    return img


def find_range(img, low_threshold=0.977, high_threshold=1.0249):
    img_use = numpy.where(img[1:] / img[:-1] > high_threshold, 1, 0)
    img_use += numpy.where(img[1:] / img[:-1] < low_threshold, -1, 0)
    points = []
    state = 1
    for i, u in enumerate(img_use):
        if u == 0:
            pass
        elif not u == state:
            state = u
            points.append(i)
    points.append(img.shape[0])

    return points


def padding(img, points):
    range_begin = 0
    new_img = numpy.ones(img.shape) * 255.
    img_range = []
    for i, point in enumerate(points):
        range_end = point
        img_range.append([range_begin, range_end])
        range_begin = range_end

    for i in range(len(img_range)):
        if i % 2 == 1:
            new_img[img_range[i][0]:img_range[i][1]] = 0

    return new_img


img_path = 'cjl/1'
img_name = 'in.jpg'
output_path = 'output'
img = cv2.imread(os.path.join(img_path, img_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = numpy.mean(img, axis=1)
points = find_range(img)
new_img = padding(img, points)
white_bar, black_bar = find_bars(new_img)
img_Contrast = img
new_img = img.reshape(new_img.shape[0], 1)
new_img = numpy.repeat(new_img, 35, axis=1)
cv2.imwrite(img_name, new_img)

img_path = 'cjl/1'
img_name = 'ad.jpg'
img = cv2.imread(os.path.join(img_path, img_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = numpy.mean(img, axis=1)
points = find_range(img)
new_img = padding(img, points)
white_bar_, black_bar_ = find_bars(new_img)
img, new_img = Adjusting_length(img, new_img, img_Contrast)
img, new_img, adjust_list = Adjusting_bar(img, new_img, black_bar_, white_bar_, black_bar, img_Contrast)
new_img = img.reshape(new_img.shape[0], 1)
new_img = numpy.repeat(new_img, 35, axis=1)
cv2.imwrite(img_name, new_img)
