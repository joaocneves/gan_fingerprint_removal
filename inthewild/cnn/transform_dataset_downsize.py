
import os
import cv2
import sys
import glob



img_size = 64
original_dataset_dir = 'G:\\deepfakepaper\\exp2\\real2fake_100F_VF2_train_NFF_VF2_test'
out_path = 'real2fake_downsize_{0}_100F_VF2_train_NFF_VF2_test'.format(img_size)

if len(sys.argv) > 1:
    img_size = int(sys.argv[1])
    original_dataset_dir = sys.argv[2]
    out_path = sys.argv[3]

subset_class_to_modify = [('test','1')]


# ----------------------- LOAD DATA ---------------------------- #

subsets = os.listdir(original_dataset_dir)

data = dict()
for s in subsets:

    data[s] = dict()
    classes = os.listdir(original_dataset_dir + '\\' + s)

    for c in classes:

        images = glob.glob(original_dataset_dir + '\\' + s + '\\' + c + '\\*.jpg')
        data[s][c] = images

# ----------------------- CREATE FOLDERS ---------------------------- #

for s in subsets:

    classes = list(data[s].keys())

    for c in classes:

        try:
            folderpath = os.path.join(out_path, s, c)
            os.makedirs(folderpath)
        except OSError:
            print("Creation of the directory %s failed" % folderpath)
        else:
            print("Successfully created the directory %s " % folderpath)

# ----------------------- TRANSFORM IMAGES ---------------------------- #

for s in subsets:

    classes = list(data[s].keys())

    for c in classes:

        for image_path in data[s][c]:

            original_image_path = image_path
            output_image_path = original_image_path.replace(original_dataset_dir, out_path)

            im = cv2.imread(original_image_path)
            if (s,c) in subset_class_to_modify:
                im = cv2.resize(im, (img_size, img_size))
                im = cv2.resize(im, (224, 224))
            cv2.imwrite(output_image_path, im)





