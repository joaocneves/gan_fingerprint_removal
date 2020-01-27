import os

ae_epochs = 100
latent_code_size = 32

# subexp = ['100F_CASIA_train_NFF_VF2_test',
#           '100F_VF2_train_NFF_CASIA_test',
#           'NFF_CASIA_train_100F_VF2_test',
#           'NFF_VF2_train_100F_CASIA_test']

subexp = ['real2fake_VF2_TPDNE_train_VF2_100F_test',
          'real2fake_VF2_TPDNE_train_CASIA_100F_test',
          'real2fake_VF2_100F_train_VF2_TPDNE_test',
          'real2fake_VF2_100F_train_CASIA_TPDNE_test',
          'real2fake_CASIA_TPDNE_train_VF2_100F_test',
          'real2fake_CASIA_TPDNE_train_CASIA_100F_test',
          'real2fake_CASIA_100F_train_VF2_TPDNE_test',
          'real2fake_CASIA_100F_train_CASIA_TPDNE_test']

bat_file = open('run_all_experiments_ae.bat','w')
for s in subexp:

    original_dataset_dir = 'F:\\gan_fingerprint_removal\\data\\' + s

    out_path = 'real2fake_ae{0}_{1}_{2}'.format(latent_code_size, ae_epochs, s)
    command = "python transform_dataset_ae.py {0} {1} {2} {3}".format(ae_epochs,
            latent_code_size, original_dataset_dir, out_path)
    print(command, file=bat_file)

    model_dir = 'F:\\gan_fingerprint_removal\\data\\' + s[:s.find('_train')]
    data_dir = 'real2fake_ae{0}_{1}_{2}'.format(latent_code_size, ae_epochs, s)
    command = "python ..\\test_pytorch_gpu.py {0} {1}".format(data_dir, model_dir)
    print(command, file=bat_file)
