 # -*- coding:utf-8 -*-
__author__ = "Wang Hewen"

import platform
import os
import pandas
import cv2
import matplotlib.pyplot
import numpy
import scipy.misc
import CommonModules as CM
import multiprocessing as mp

base_folder = "H:/Kaggle/masking"

#abspath_dataset_dir_train_1 = os.path.join(base_folder, 'train/Type_1')
#abspath_dataset_dir_train_2 = os.path.join(base_folder, 'train/Type_2')
#abspath_dataset_dir_train_3 = os.path.join(base_folder, 'train/Type_3')
#abspath_dataset_dir_test    = os.path.join(base_folder, 'test')
#abspath_dataset_dir_add_1   = os.path.join(base_folder, 'additional/Type_1')
#abspath_dataset_dir_add_2   = os.path.join(base_folder, 'additional/Type_2')
#abspath_dataset_dir_add_3   = os.path.join(base_folder, 'additional/Type_3')

    
#def get_list_abspath_img(abspath_dataset_dir):
#    list_abspath_img = []
#    for str_name_file_or_dir in os.listdir(abspath_dataset_dir):
#        if ('.jpg' in str_name_file_or_dir) == True:
#            list_abspath_img.append(os.path.join(abspath_dataset_dir, str_name_file_or_dir))
#    list_abspath_img.sort()
#    return list_abspath_img


#list_abspath_img_train_1 = get_list_abspath_img(abspath_dataset_dir_train_1)
#list_abspath_img_train_2 = get_list_abspath_img(abspath_dataset_dir_train_2)
#list_abspath_img_train_3 = get_list_abspath_img(abspath_dataset_dir_train_3)
#list_abspath_img_train   = list_abspath_img_train_1 + list_abspath_img_train_2 + list_abspath_img_train_3

#list_abspath_img_test    = get_list_abspath_img(abspath_dataset_dir_test)

#list_abspath_img_add_1   = get_list_abspath_img(abspath_dataset_dir_add_1)
#list_abspath_img_add_2   = get_list_abspath_img(abspath_dataset_dir_add_2)
#list_abspath_img_add_3   = get_list_abspath_img(abspath_dataset_dir_add_3)
#list_abspath_img_add     = list_abspath_img_add_1   + list_abspath_img_add_2   + list_abspath_img_add_3

## 0: Type_1, 1: Type_2, 2: Type_3
#list_answer_train        = [0] * len(list_abspath_img_train_1) + [1] * len(list_abspath_img_train_2) + [2] * len(list_abspath_img_train_3)
#list_answer_add          = [0] * len(list_abspath_img_add_1) + [1] * len(list_abspath_img_add_2) + [2] * len(list_abspath_img_add_3)

def sub_func_load_img(abspath_img):
    img_rgb = cv2.cvtColor(cv2.imread(abspath_img), cv2.COLOR_BGR2RGB)
    return img_rgb

def show_img(abspath_img):
    matplotlib.pyplot.imshow(sub_func_load_img(abspath_img))
    matplotlib.pyplot.show()

def sub_func_rotate_img_if_need(img_rgb):
    if img_rgb.shape[0] >= img_rgb.shape[1]:
        return img_rgb
    else:
        return numpy.rot90(img_rgb)

def sub_func_resize_img_same_ratio(img_rgb):
    if img_rgb.shape[0] / 640.0 >= img_rgb.shape[1] / 480.0:
        img_resized_rgb = cv2.resize(img_rgb, (int(640.0 * img_rgb.shape[1] / img_rgb.shape[0]), 640)) # (640, *, 3)
    else:
        img_resized_rgb = cv2.resize(img_rgb, (480, int(480.0 * img_rgb.shape[0] / img_rgb.shape[1]))) # (*, 480, 3)
    return img_resized_rgb

def sub_func_fill_img(img_rgb):
    if img_rgb.shape[0] == 640:
        int_resize_1    = img_rgb.shape[1]
        int_fill_1      = (480 - int_resize_1 ) // 2
        int_fill_2      =  480 - int_resize_1 - int_fill_1
        numpy_fill_1    =  numpy.zeros((640, int_fill_1, 3), dtype=numpy.uint8)
        numpy_fill_2    =  numpy.zeros((640, int_fill_2, 3), dtype=numpy.uint8)
        img_filled_rgb = numpy.concatenate((numpy_fill_1, img_rgb, numpy_fill_1), axis=1)
    elif img_rgb.shape[1] == 480:
        int_resize_0    = img_rgb.shape[0]
        int_fill_1      = (640 - int_resize_0 ) // 2
        int_fill_2      =  640 - int_resize_0 - int_fill_1
        numpy_fill_1 =  numpy.zeros((int_fill_1, 480, 3), dtype=numpy.uint8)
        numpy_fill_2 =  numpy.zeros((int_fill_2, 480, 3), dtype=numpy.uint8)
        img_filled_rgb = numpy.concatenate((numpy_fill_1, img_rgb, numpy_fill_1), axis=0)
    else:
        raise ValueError
    return img_filled_rgb

def sub_func_rotate_img_if_need(img_rgb):
    if img_rgb.shape[0] >= img_rgb.shape[1]:
        return img_rgb
    else:
        return numpy.rot90(img_rgb)

def sub_func_resize_img_same_ratio(img_rgb):
    if img_rgb.shape[0] / 640.0 >= img_rgb.shape[1] / 480.0:
        img_resized_rgb = cv2.resize(img_rgb, (int(640.0 * img_rgb.shape[1] / img_rgb.shape[0]), 640)) # (640, *, 3)
    else:
        img_resized_rgb = cv2.resize(img_rgb, (480, int(480.0 * img_rgb.shape[0] / img_rgb.shape[1]))) # (*, 480, 3)
    return img_resized_rgb

def sub_func_fill_img(img_rgb):
    if img_rgb.shape[0] == 640:
        int_resize_1    = img_rgb.shape[1]
        int_fill_1      = (480 - int_resize_1 ) // 2
        int_fill_2      =  480 - int_resize_1 - int_fill_1
        numpy_fill_1    =  numpy.zeros((640, int_fill_1, 3), dtype=numpy.uint8)
        numpy_fill_2    =  numpy.zeros((640, int_fill_2, 3), dtype=numpy.uint8)
        img_filled_rgb = numpy.concatenate((numpy_fill_1, img_rgb, numpy_fill_1), axis=1)
    elif img_rgb.shape[1] == 480:
        int_resize_0    = img_rgb.shape[0]
        int_fill_1      = (640 - int_resize_0 ) // 2
        int_fill_2      =  640 - int_resize_0 - int_fill_1
        numpy_fill_1 =  numpy.zeros((int_fill_1, 480, 3), dtype=numpy.uint8)
        numpy_fill_2 =  numpy.zeros((int_fill_2, 480, 3), dtype=numpy.uint8)
        img_filled_rgb = numpy.concatenate((numpy_fill_1, img_rgb, numpy_fill_1), axis=0)
    else:
        raise ValueError
    return img_filled_rgb

def sub_func_resample_img(abspath_img):
    print(".", end = "", flush = True)
    img = sub_func_load_img(abspath_img)
    img = sub_func_rotate_img_if_need(img)
    img = sub_func_resize_img_same_ratio(img)
    img = sub_func_fill_img(img)    
    return img

def show_resample_img(abspath_img):
    matplotlib.pyplot.imshow(sub_func_resample_img(abspath_img))
    matplotlib.pyplot.show()

def processing_function(resample_img):
    try:
        path = resample_img
        preprocessed_folder = "H:/Kaggle/masking_preprocessed"
        new_path = os.path.join(preprocessed_folder, os.path.relpath(resample_img, base_folder))
        #matplotlib.pyplot.imshow(resample_img)
        #matplotlib.pyplot.show()
        #if "Type_1" in path:
        #    scipy.misc.imsave(os.path.join("./kaggle/train/Type_1", os.path.split(path)[-1]), resample_img)
        #if "Type_2" in path:
        #    scipy.misc.imsave(os.path.join("./kaggle/train/Type_2", os.path.split(path)[-1]), resample_img)
        #if "Type_3" in path:
        #    scipy.misc.imsave(os.path.join("./kaggle/train/Type_3", os.path.split(path)[-1]), resample_img)
        scipy.misc.imsave(new_path, sub_func_resample_img(resample_img))
    except:
        print(resample_img)
    return

def main():
    #print(len(list_answer_train), len(list_answer_add))
    #list_img_train = list_abspath_img_train    
    pool = mp.Pool(3)
    pool.map(processing_function, CM.IO.ListFiles(base_folder, ".jpg", All = True))
    return

if __name__ == "__main__":
    mp.freeze_support()
    main()