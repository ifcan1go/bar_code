# encoding utf-8
import cv2
import numpy
import os


def find_range(img, low_threshold=0.97, high_threshold=1.034):
    img_use = numpy.where(img[1:] / img[:-1] > high_threshold, 1, 0)
    img_use += numpy.where(img[1:] / img[:-1] < low_threshold, -1, 0)
    points = []
    state = 0

    for i, u in enumerate(img_use):
        if u == 0:
            pass
        elif not u == state:
            state = u
            points.append(i)
    points.append(img.shape[0]-1)
    return points

def padding(img, points):
    range_begin = 0
    new_img = numpy.ones(img.shape) * 255.
    img_range = []
    white_bar = []
    for i, point in enumerate(points):
        range_end = point
        img_range.append([range_begin, range_end])
        range_begin = range_end

    for i in range(len(img_range)):
        if i % 2 == 1:
            new_img[img_range[i][0]:img_range[i][1]] = 0

    return new_img, white_bar


def bar_code(img_path, img_name, output_path, low_threshold, high_threshold):
    img = cv2.imread(os.path.join(img_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = numpy.mean(img, axis=1)
    points = find_range(img, low_threshold, high_threshold)
    img_point = numpy.zeros(img.shape[0])
    for i in points:
        img_point[i] = 1
    new_img, white_bar = padding(img, points)
    new_img = new_img.reshape(new_img.shape[0], 1)
    new_img = numpy.repeat(new_img, 35, axis=1)
    output_name = os.path.join(output_path, img_name)
    cv2.imwrite(output_name, new_img)
    return white_bar


def output_file(output_path, white_bar, filename='white_bar_location.txt'):
    with open(os.path.join(output_path, filename), 'wb+') as f:
        f.write(('01lane_01:' + str(white_bar) + '\n').encode("utf-8"))


if __name__ == '__main__':
    img_path = 'cjl'
    output_path = 'output'
    low_threshold = 0.97
    high_threshold = 1.034
    white_bar = bar_code(img_path, 'out.jpg', output_path, low_threshold, high_threshold)
