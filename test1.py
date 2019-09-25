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
            white_bar.append([w_start, w_end])
        if i == img.shape[0] - 1:
            white_bar.append([w_start, i])
    return numpy.asarray(white_bar), numpy.asarray(black_bar)


# def Adjusting_length(img, new_img, img_Contrast):
#     constrast_length = img_Contrast.shape[0]
#     img_length = img.shape[0]
#     if constrast_length == img_length:
#         return img, new_img
#     elif img_length > constrast_length:
#         sub_length = img_length - constrast_length
#         for i in range(sub_length):
#             sub_position = int((sub_length - i - 1) * img_length / (img_length - constrast_length))
#             while new_img[sub_position] == 0:
#                 sub_position += 1
#             img = numpy.delete(img, sub_position)
#             new_img = numpy.delete(new_img, sub_position)
#     elif img_length < constrast_length:
#         add_length = constrast_length - img_length
#         for i in range(add_length):
#             add_position = int(i * img_length / (constrast_length - img_length))
#             img = numpy.insert(img, add_position, img[add_position])
#             new_img = numpy.insert(new_img, add_position, new_img[add_position])
#     return img, new_img


def Adjusting_bar(black_bar, white_bar, black_bar_Contrast):
    if not len(black_bar) == len(black_bar_Contrast):
        raise ("BAR Detective ERROR")
    adjust_list = []
    for b_i in range(len(black_bar)):
        bar = black_bar[b_i]
        bar_contrast = black_bar_Contrast[b_i]
        for w_i in range(len(white_bar)):
            if white_bar[w_i][0] < bar:
                adjust_position = white_bar[w_i - 1]
        adjust_list.append([adjust_position,bar - bar_contrast])
    return adjust_list


def transfer(img, new_img, white_bar, adjust_list, img_contrast):
    for ad_i in range(len(adjust_list)):
        adjust_num = adjust_list[ad_i][1]
        ad_list = []
        for w_i in white_bar:
            if white_bar[w_i][0] > adjust_list[ad_i][0][1] or white_bar[w_i][1] < adjust_list[ad_i][0][0]:
                ad_list.append(white_bar[w_i])
        if len(ad_list) > 0:
            if adjust_num < 0:
                # add_pxl
                for n_add_pxl in range(-adjust_num):
                    x_white_bar = n_add_pxl % len(ad_list)
                    position = int((ad_list[x_white_bar][0] + ad_list[x_white_bar][1]) / 2)
                    numpy.insert(img, position, img[position])
                    numpy.insert(new_img, position, new_img[position])
            if adjust_num > 0:
                for n_sub_pxl in range(adjust_num):
                    x_white_bar = n_sub_pxl % len(ad_list)
                    position = int((ad_list[x_white_bar][0] + ad_list[x_white_bar][1]) / 2)
                    numpy.delete(img, position)
                    numpy.delete(new_img, position)
                    if new_img[position + 1] > 0 and new_img[position - 1] > 0:
                        ad_list = numpy.asarray(ad_list)
                        numpy.delete(ad_list, x_white_bar)
                        if len(ad_list) == 0 and ad_i + 1 < len(adjust_list):
                            adjust_list[ad_i + 1][0][2] = (adjust_num + n_sub_pxl - 1)
                            break
        elif ad_i + 1 < len(adjust_list):
            adjust_list[ad_i + 1][0][2] = (adjust_num + adjust_num)

    if img_contrast.shape[0] - img.shape[0] > 0:
        for i in range(img_contrast.shape[0] - img.shape[0]):
            numpy.insert(img, -1, img[-1])
    else:
        for i in range(img.shape[0] - img_contrast.shape[0]):
            numpy.delete(img, -1)

    return img, new_img


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

adjust_list=Adjusting_bar(black_bar, white_bar, black_bar_)
img, new_img=transfer(img, new_img, white_bar, adjust_list, img_Contrast)
new_img = img.reshape(new_img.shape[0], 1)
new_img = numpy.repeat(new_img, 35, axis=1)
cv2.imwrite(img_name, new_img)
