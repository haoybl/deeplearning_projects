 # -*- coding:utf-8 -*-
__author__ = "Wang Hewen"

import platform
import os
import pandas
import cv2
import matplotlib.pyplot
import numpy as np
import scipy.misc
import traceback
import csv
import CommonModules as CM
import multiprocessing as mp

base_folder = "H:/Kaggle"

def processing_function(resample_img, x, y, width, height):
    try:
        preprocessed_folder = "H:/Kaggle/masking"
        old_path = os.path.join(base_folder, os.path.relpath(resample_img, base_folder))
        new_path = os.path.join(preprocessed_folder, os.path.relpath(resample_img, base_folder))

        img_rgb = cv2.cvtColor(cv2.imread(resample_img), cv2.COLOR_BGR2RGB)
        #new_img = img_rgb
        new_img = np.zeros_like(img_rgb)
        new_img[x: x + width, y: y + height] = 255
        #matplotlib.pyplot.imshow(resample_img)
        #matplotlib.pyplot.show()
        #if "Type_1" in path:
        #    scipy.misc.imsave(os.path.join("./kaggle/train/Type_1", os.path.split(path)[-1]), resample_img)
        #if "Type_2" in path:
        #    scipy.misc.imsave(os.path.join("./kaggle/train/Type_2", os.path.split(path)[-1]), resample_img)
        #if "Type_3" in path:
        #    scipy.misc.imsave(os.path.join("./kaggle/train/Type_3", os.path.split(path)[-1]), resample_img)
        scipy.misc.imsave(new_path, new_img)
    except:
        print(resample_img)
        traceback.print_exc()
    return

def main():  
    #pool = mp.Pool(3)
    #pool.map(processing_function, CM.IO.ListFiles(base_folder, ".jpg", All = True))
    for tsv in CM.IO.ListFiles("H:/Kaggle/masking", "tsv"):
        with open(tsv,'r') as tsvin:
            tsvin = csv.reader(tsvin, delimiter=' ')
            for row in tsvin:
                name = row[0]
                type = int(row[1])
                x, y, width, height = int(row[2]), int(row[3]), int(row[4]), int(row[5])
                processing_function(os.path.join(base_folder, "train", name), x, y, width, height)
                print(".", end = "", flush = True)

        #break
    return

if __name__ == "__main__":
    mp.freeze_support()
    main()