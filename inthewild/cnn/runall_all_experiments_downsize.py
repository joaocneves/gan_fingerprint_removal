import os

subexp = ['real2fake_VF2_TPDNE_train_VF2_100F_test',
          'real2fake_VF2_TPDNE_train_CASIA_100F_test',
          'real2fake_VF2_100F_train_VF2_TPDNE_test',
          'real2fake_VF2_100F_train_CASIA_TPDNE_test',
          'real2fake_CASIA_TPDNE_train_VF2_100F_test',
          'real2fake_CASIA_TPDNE_train_CASIA_100F_test',
          'real2fake_CASIA_100F_train_VF2_TPDNE_test',
          'real2fake_CASIA_100F_train_CASIA_TPDNE_test']

bat_file = open('run_all_experiments_downsize.bat','w')
for s in subexp:

    original_dataset_dir = 'F:\\gan_fingerprint_removal\\data\\' + s
    model_dir = 'F:\\gan_fingerprint_removal\\data\\' + s[:s.find('_train')]

    for img_size in range(32,224+32,32):

        out_path = '{0}_R{1}'.format(original_dataset_dir, img_size)
        run_command = "python transform_dataset_downsize.py {0} {1} {2}".format(img_size,
            original_dataset_dir, out_path)

        print(run_command, file=bat_file)

        data_dir = '{0}_R{1}'.format(original_dataset_dir, img_size)
        run_command = "python ..//..//test_pytorch_gpu.py {0} {1}".format(data_dir, model_dir)

        print(run_command, file=bat_file)
