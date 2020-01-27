import os
import glob
import random
import shutil

IMAGES_PER_CLASS = 10000


ds1_path = 'F:\\FACE_DATASETS\\VGG_FACE_2\\byid_alignedlib_0.3_train\\'
ds2_path = 'F:\\FACE_DATASETS\\CASIA-WebFace\\byid_alignedlib_0.3\\'
out_path = 'real2real\\'

ds1_files = [f for f in glob.glob(ds1_path + "**/*.jpg", recursive=True)]
ds2_files = [f for f in glob.glob(ds2_path + "**/*.jpg", recursive=True)]
print("Found " + str(len(ds1_files))  + " in dataset 1")
print("Found " + str(len(ds2_files))  + " in dataset 2")

#imgs_per_class = min(len(ds1_files),len(ds2_files))
imgs_per_class = IMAGES_PER_CLASS
print("The new dataset will have " + str(imgs_per_class) + " images on each class")

# random.shuffle(ds1_files)
# random.shuffle(ds2_files)

# SELECT A SUBSET OF 'imgs_per_class' IMAGES FROM EACH DATASET

ds1_files = ds1_files[:imgs_per_class]
ds2_files = ds2_files[:imgs_per_class]

# ds1_files = ds1_files + ds2_files
# ds2_files = ds1_files
# random.shuffle(ds1_files)
# random.shuffle(ds2_files)
# random.shuffle(ds2_files)
# ds1_files = ds1_files[:imgs_per_class]
# ds2_files = ds2_files[:imgs_per_class]

# WRITE IMAGES TO DISK

try:
    os.makedirs(out_path + '0')
    os.makedirs(out_path + '1')
except OSError:
    print ("Creation of the directory %s failed" % out_path)
else:
    print ("Successfully created the directory %s " % out_path)

# CLASS 0

idx = 0
for f in ds1_files:
    img_name = 'img_{:06d}.jpg'.format(idx)
    shutil.copyfile(f,out_path + '0/' + img_name)
    idx = idx + 1

# CLASS 1

idx = 0
for f in ds2_files:
    img_name = 'img_{:06d}.jpg'.format(idx)
    shutil.copyfile(f,out_path + '1/' + img_name)
    idx = idx + 1
