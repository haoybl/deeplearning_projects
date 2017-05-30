import os
from os.path import join
fpa = {}
for root, dirs, files in os.walk('H:/Kaggle/preprocessed/train/Type_3'):
   for name in files:
    fpa[name] = 1


fpb = {}
for root, dirs, files in os.walk('H:/Kaggle/masking_preprocessed/train/Type_3'):
   for name in files:
    fpb[name] = 1

print("files only in a")
for name in fpa.keys():
    if not(name in fpb.keys()):
        print(name,"\n")

print("files only in b")
for name in fpb.keys():
    if not(name in fpa.keys()):
        print(name,"\n")