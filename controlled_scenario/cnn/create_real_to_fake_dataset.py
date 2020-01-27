import os
import sys
import glob
import random
import shutil
import math as m
from sklearn.model_selection import train_test_split


def createdataset_byimg(ds_files_subsets, subsets, classname, out_path):

    """
    ds_files_subsets - dictionary with the image paths of each training split of a specific class of the dataset
         example:
         ds_files_subsets['train'] - list of training images paths
         ds_files_subsets['val'] - list of training images paths
         ds_files_subsets['test'] - list of training images paths

    subsets - list of the training splits that we want to include in the dataset

    classname - name of the class (str)

    out_path - the path where the images will be written

    """

    for s in subsets:

        try:
            folderpath = os.path.join(out_path, s, classname)
            os.makedirs(folderpath)
        except OSError:
            print("Creation of the directory %s failed" % out_path)
        else:
            print("Successfully created the directory %s " % out_path)

        idx = 0
        for i in range(len(ds1_files_subsets[s])):

            img_file = ds_files_subsets[s][i]

            img_name = 'img_{:06d}.jpg'.format(idx)
            shutil.copyfile(img_file, os.path.join(out_path, s, classname, img_name))
            idx = idx + 1

            if idx >= n_img[s]:
                break


def createdataset_byid(ds_files_subsets, subsets, classname, out_path):

    """
    ds_files_subsets - dictionary with the id paths of each training split of a specific class of the dataset
         example:
         ds_files_subsets['train'] - list of training images paths
         ds_files_subsets['val'] - list of training images paths
         ds_files_subsets['test'] - list of training images paths

    subsets - list of the training splits that we want to include in the dataset

    classname - name of the class (str)

    out_path - the path where the images will be written

    """

    for s in subsets:

        try:
            folderpath = os.path.join(out_path, s, classname)
            os.makedirs(folderpath)
        except OSError:
            print("Creation of the directory %s failed" % out_path)
        else:
            print("Successfully created the directory %s " % out_path)

        idx = 0
        for i in range(len(ds_files_subsets[s])):

            id_path = ds_files_subsets[s][i]
            img_files = [ff for ff in glob.glob(id_path + "/*.jpg")]

            for f in img_files:

                img_name = 'img_{:06d}.jpg'.format(idx)
                shutil.copyfile(f, os.path.join(out_path, s, classname, img_name))
                idx = idx + 1

                if idx >= n_img[s]:
                    break

            if idx >= n_img[s]:
                break



if __name__ == "__main__":

    IMAGES_PER_CLASS = 10000
    subsets = ['train', 'val', 'test']
    subsets_prop = [0.5, 0.2, 0.3]

    ds1_path = 'G:\\FACE_DATASETS\\VGG_FACE_2\\byid_alignedlib_0.3_train\\'
    ds2_path = 'G:\\FACE_DATASETS\\100K_FAKE\\byimg_alignedlib_0.3\\'
    out_path = 'G:\\gan_fingerprint_removal\\data\\real2fake_VF2_100F'

    if len(sys.argv) > 1:
        ds1_path = sys.argv[1]
        ds2_path = sys.argv[2]
        out_path = sys.argv[3]

    ds1_files = [f for f in glob.glob(ds1_path + "*", recursive=True)]
    ds2_files = [f for f in glob.glob(ds2_path + "*", recursive=True)]
    print("Found " + str(len(ds1_files)) + " in dataset 1")
    print("Found " + str(len(ds2_files)) + " in dataset 2")

    imgs_per_id = max(m.ceil(IMAGES_PER_CLASS/len(ds1_files)), m.ceil(IMAGES_PER_CLASS/len(ds2_files)))
    ids_to_use = m.floor(IMAGES_PER_CLASS/imgs_per_id)

    # DIVIDE INTO SUBSETS BY IDENTITY

    ds1_files_subsets = dict()
    ds1_files_subsets[subsets[0]], ds1_files_subsets[subsets[2]] = \
        train_test_split(ds1_files, test_size=subsets_prop[2], shuffle=False, random_state=42)
    ds1_files_subsets[subsets[0]], ds1_files_subsets[subsets[1]] = \
        train_test_split(ds1_files_subsets[subsets[0]], test_size=subsets_prop[1]/(subsets_prop[0]+subsets_prop[1]), shuffle=False, random_state=42)

    ds2_files_subsets = dict()
    ds2_files_subsets[subsets[0]], ds2_files_subsets[subsets[2]] = \
        train_test_split(ds2_files, test_size=subsets_prop[2], shuffle=False, random_state=42)
    ds2_files_subsets[subsets[0]], ds2_files_subsets[subsets[1]] = \
        train_test_split(ds2_files_subsets[subsets[0]], test_size=subsets_prop[1]/(subsets_prop[0]+subsets_prop[1]), shuffle=False, random_state=42)

    n_img = dict()
    n_img[subsets[0]] = IMAGES_PER_CLASS*subsets_prop[0]
    n_img[subsets[1]] = IMAGES_PER_CLASS*subsets_prop[1]
    n_img[subsets[2]] = IMAGES_PER_CLASS*subsets_prop[2]

    # CLASS 0

    classname = '0'
    if ds1_path.find('byimg_') >= 0:
        createdataset_byimg(ds1_files_subsets, subsets, classname, out_path)
    else:
        createdataset_byid(ds1_files_subsets, subsets, classname, out_path)

    # CLASS 1

    classname = '1'
    if ds2_path.find('byimg_') >= 0:
        createdataset_byimg(ds2_files_subsets, subsets, classname, out_path)
    else:
        createdataset_byid(ds2_files_subsets, subsets, classname, out_path)

