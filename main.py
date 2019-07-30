# encoding utf-8
import cv2
import numpy
import os


def find_range(img, threshold=0.085):
    sub_img = numpy.abs(img[:-1] - img[1:]) / (img[1:] + 1)
    img_use = numpy.where(sub_img > threshold, sub_img, 0)
    points = []
    state = 0
    edge_start = 0
    for i, u in enumerate(img_use):
        if u > 0 and state == 0:
            edge_start = i
            state = 1
        if u == 0 and state == 1:
            edge_end = i
            point = int((edge_end + edge_start) / 2) + 1
            points.append(point)
            if edge_end == edge_start + 1:
                points.append(point + 2)
            state = 0
    return points


def padding(img, points):
    range_begin = 0
    new_img = numpy.zeros(img.shape)
    img_range = []
    for i, point in enumerate(points):
        range_end = point
        img_range.append([range_begin, range_end])
        range_begin = range_end

    for i in range(len(img_range) - 2):
        adjacent_color = (numpy.mean(img[img_range[i][0]:img_range[i][1]]) + numpy.mean(
            img[img_range[i + 2][0]:img_range[i + 2][1]])) / 2
        this_color = numpy.mean(img[img_range[i + 1][0]:img_range[i + 1][1]])
        if this_color > adjacent_color:
            new_img[img_range[i + 1][0]:img_range[i + 1][1]] = 255
        else:
            new_img[img_range[i + 1][0]:img_range[i + 1][1]] = 0

    return new_img


def bar_code(img_path, img_name, output_path):
    img = cv2.imread(os.path.join(img_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = numpy.mean(img, axis=1)
    points = find_range(img)
    img_point = numpy.zeros(img.shape[0])
    for i in points:
        img_point[i] = 1
    new_img = padding(img, points)
    new_img = new_img.reshape(new_img.shape[0], 1)
    new_img = numpy.repeat(new_img, 35, axis=1)
    output_name = os.path.join(output_path, img_name)
    cv2.imwrite(output_name, new_img)

if __name__ == '__main__':
    img_path = '01lane'
    output_path = 'output'
    for i in range(15):
       bar_code(img_path,'01lane_%i.bmp'%(i+1),output_path)