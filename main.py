# encoding utf-8
import cv2
import numpy
import os
import copy
import argparse


def find_bars(img):
    '''
    根据已有的黑白binary_img寻找黑条
    :param img: binary_img
    :return:
    '''
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


def Adjusting_bar(black_bar, white_bar, black_bar_Contrast):
    '''
    根据内标外标黑条位置对比生成调整列表
    :param black_bar: 内标黑条
    :param white_bar: 内标白条
    :param black_bar_Contrast:外标黑条
    :return: adjust_list需调整的列表
    '''
    if not len(black_bar) == len(black_bar_Contrast):
        raise ("BAR Detective ERROR")
    adjust_list = []
    distance_acc = 0
    for b_i in range(len(black_bar)):
        bar = black_bar[b_i]
        bar_contrast = black_bar_Contrast[b_i]
        adjust_position = []
        for w_i in range(len(white_bar)):
            if white_bar[w_i][1] < bar_contrast:
                adjust_position = white_bar[w_i]
        distance = bar_contrast - bar - distance_acc
        distance_acc += distance
        adjust_list.append([adjust_position, distance])
    return adjust_list


def transfer(img, new_img, adjust_list_save, length):
    '''
    根据adjust_list，将图标调整变形
    :param img: 输入的图片
    :param new_img: 即黑白图片binary_img
    :param adjust_list_save: 保存的调整列表
    :param length: 图片的长度
    :return: img调整好的图片，new_img调整好的黑白图片binary_img
    '''
    adjust_list = copy.deepcopy(adjust_list_save)
    for ad_i in range(len(adjust_list)):
        adjust_num = adjust_list[ad_i][1]
        ad_list = []
        white_bar, black_bar = find_bars(new_img)
        for w_i in range(len(white_bar)):
            if (white_bar[w_i][0] >= adjust_list[ad_i][0][0] and white_bar[w_i][0] <= adjust_list[ad_i][0][1]) or (
                    white_bar[w_i][1] >= adjust_list[ad_i][0][0] and white_bar[w_i][1] <= adjust_list[ad_i][0][1]):
                ad_list.append(white_bar[w_i])
        if len(ad_list) > 0:
            ad_list = numpy.asarray(ad_list)
            if adjust_num < 0:
                # add_pxl
                for n_add_pxl in range(-adjust_num):
                    x_white_bar = n_add_pxl % len(ad_list)
                    position = int((ad_list[x_white_bar][0] + ad_list[x_white_bar][1]) / 2)
                    img = numpy.insert(img, position, img[position])
                    new_img = numpy.insert(new_img, position, new_img[position])
                    for i_ad_list in range(x_white_bar, ad_list.shape[0]):
                        if i_ad_list == x_white_bar:
                            ad_list[i_ad_list][1] += 1
                        elif i_ad_list > x_white_bar:
                            ad_list[i_ad_list][0] += 1
                            ad_list[i_ad_list][1] += 1

            elif adjust_num > 0:
                # delete_pxl
                for n_sub_pxl in range(adjust_num):
                    x_white_bar = n_sub_pxl % len(ad_list)
                    position = int((ad_list[x_white_bar][0] + ad_list[x_white_bar][1]) / 2)
                    img = numpy.delete(img, position)
                    new_img = numpy.delete(new_img, position)
                    for i_ad_list in range(x_white_bar, ad_list.shape[0]):
                        if i_ad_list == x_white_bar:
                            ad_list[i_ad_list][1] -= 1
                            if ad_list[i_ad_list][0] + 1 == ad_list[i_ad_list][1]:
                                ad_list = numpy.delete(ad_list, x_white_bar, axis=0)
                                if len(ad_list) == 0 and ad_i + 1 < len(adjust_list):
                                    adjust_list[ad_i + 1][0][2] = (adjust_num + n_sub_pxl - 1)
                                    break
                        elif i_ad_list > x_white_bar and i_ad_list < ad_list.shape[0]:
                            ad_list[i_ad_list][0] -= 1
                            ad_list[i_ad_list][1] -= 1
                        ad_list = numpy.where(ad_list < 0, 0, ad_list)
            for adjust_list_i in range(len(adjust_list)):
                adjust_list[adjust_list_i][0][0] -= adjust_num
                if adjust_list[adjust_list_i][0][0] < 0:
                    adjust_list[adjust_list_i][0][0] = 0
                adjust_list[adjust_list_i][0][1] -= adjust_num
                if adjust_list[adjust_list_i][0][1] < 0:
                    adjust_list[adjust_list_i][0][1] = 0
        elif ad_i + 1 < len(adjust_list) and len(adjust_list) < ad_i + 1:
            adjust_list[ad_i + 1][0][2] = (adjust_num + adjust_num)

    if length - img.shape[0] > 0:
        for i in range(length - img.shape[0]):
            img = numpy.insert(img, -1, img[-1])
            new_img = numpy.insert(new_img, -1, img[-1])
    else:
        del_sub_white_num = 0
        white_bar, black_bar = find_bars(new_img)
        for i in range(img.shape[0] - length):
            if new_img[-1] > 0:
                img = numpy.delete(img, -1)
                new_img = numpy.delete(new_img, -1)
            else:
                for white_bar_i in range(white_bar.shape[0]):
                    if white_bar[white_bar_i].shape[0] == 1:
                        white_bar = numpy.delete(white_bar, white_bar_i, axis=0)
                x_white_bar = del_sub_white_num % len(white_bar)
                position = int((white_bar[x_white_bar][0] + white_bar[x_white_bar][1]) / 2)
                img = numpy.delete(img, position)
                new_img = numpy.delete(new_img, position)
                for i_white_bar in range(x_white_bar, white_bar.shape[0]):
                    if i_white_bar == x_white_bar:
                        white_bar[i_white_bar][1] -= 1
                    elif i_white_bar > x_white_bar and i_white_bar < white_bar.shape[0]:
                        white_bar[i_white_bar][0] -= 1
                        white_bar[i_white_bar][1] -= 1

    return img, new_img


def find_range(img, low_threshold=0.977, high_threshold=1.0249):
    '''
    识别图片的黑白的边缘
    :param img: 图片
    :param low_threshold:低阈值
    :param high_threshold: 高阈值
    :return: 黑边变换的边缘节点
    '''
    img = numpy.where(img == 0, 1, img)
    img_use = numpy.where(img[1:] / (img[:-1]) > high_threshold, 1, 0)
    img_use += numpy.where(img[1:] / (img[:-1]) < low_threshold, -1, 0)
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
    '''
    根据边缘节点points，填充黑白色
    :param img: 图片
    :param points: 边缘节点
    :return:
    '''
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


def load_pic(img_path, img_name):
    '''
    读取图片
    :param img_path:图片路径
    :param img_name: 图片文件名
    :return:
    '''
    img = cv2.imread(os.path.join(img_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = img.shape[1]
    length = img.shape[0]
    img = numpy.mean(img, axis=1)
    return img, length, width


def save_pic(img, output_path, width, img_name):
    '''
    保存图片
    :param img:图片矩阵
    :param output_path:保存路径
    :param width: 图片宽度
    :param img_name: 保存文件名
    :return:
    '''
    img_name=img_name.split('/')[-1]
    new_img = img.reshape(img.shape[0], 1)
    new_img = numpy.repeat(new_img, width, axis=1)
    cv2.imwrite(os.path.join(output_path, img_name), new_img)


def compare_in_out(mark_in_name, mark_out_name, low_threshold, high_threshold, img_path='pic'):
    '''
    对比内外表图片
    :param mark_in_name:内标
    :param mark_out_name: 外标
    :param low_threshold: 低阈值
    :param high_threshold: 高阈值
    :param img_path: 图片路径
    :return: 调整列表，图片跨度，长度
    '''
    img, _, width = load_pic(img_path, mark_out_name)
    points_out = find_range(img, low_threshold, high_threshold)
    binary_img_out = padding(img, points_out)
    _, black_bar_out = find_bars(binary_img_out)

    img, length, width = load_pic(img_path, mark_in_name)
    points_in = find_range(img, low_threshold, high_threshold)
    binary_img_in = padding(img, points_in)
    white_bar_in, black_bar_in = find_bars(binary_img_in)
    adjust_list_save = Adjusting_bar(black_bar_out, white_bar_in, black_bar_in)
    return adjust_list_save, length, width


def process_img(img_name, img_path='pic', output_path='output', adjust_list_save=None, mark_in_name=None,
                mark_out_name=None, low_threshold=0.977, high_threshold=1.0249):
    '''
    根据调整列表对图片进行处理并保存处理好的图片
    :param img_name: 图片文件名
    :param img_path: 图片路径
    :param output_path: 保存路径
    :param adjust_list_save:调整列表
    :param mark_in_name: 内标文件名
    :param mark_out_name: 外标文件名
    :param low_threshold: 低阈值
    :param high_threshold: 高阈值
    :return:
    '''
    if adjust_list_save is None:
        adjust_list_save, length, width = compare_in_out(mark_in_name, mark_out_name, img_path=img_path,
                                                         low_threshold=low_threshold, high_threshold=high_threshold)
    adjust_list = copy.deepcopy(adjust_list_save)
    img, length, width = load_pic(img_path, img_name)
    points = find_range(img, low_threshold, high_threshold)
    binary_img = padding(img, points)
    img, binary_img = transfer(img, binary_img, adjust_list, length)
    save_pic(img, output_path, width, img_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'please enter seven parameter'
    parser.add_argument("-low", help="this is parameter low_threshold", type=float, default="0.977")
    parser.add_argument("-high", help="this is parameter high_threshold", type=float, default="1.0249")
    parser.add_argument("-img_name", help="this is parameter img_name", type=str, default="pic/1.jpg")
    parser.add_argument("-img_path", help="this is parameter img_path", type=str, default="")
    parser.add_argument("-output_path", help="this is parameter output_path", type=str, default="output")
    parser.add_argument("-mark_in_name", help="this is parameter mark_in_name", type=str, default="pic/in.jpg")
    parser.add_argument("-mark_out_name", help="this is parameter mark_out_name", type=str, default="pic/out.jpg")
    args = parser.parse_args()
    img_name = args.img_name
    img_path = args.img_path
    output_path = args.output_path
    mark_in_name = args.mark_in_name
    mark_out_name = args.mark_out_name
    low_threshold = args.low
    high_threshold = args.high
    process_img(img_name=img_name, img_path=img_path, output_path=output_path, mark_in_name=mark_in_name,
                mark_out_name=mark_out_name,
                low_threshold=low_threshold, high_threshold=high_threshold)
