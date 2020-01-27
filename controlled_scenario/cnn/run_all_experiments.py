
import os

dataset_root = 'E:\\FACE_DATASETS\\'
data_root = 'E:\\gan_fingerprint_removal\\data\\'

real_sets_path = ['VGG_FACE_2\\byid_alignedlib_0.3_train\\', 'CASIA-WebFace\\byid_alignedlib_0.3\\']
real_sets_name = ['VF2', 'CASIA']

fake_sets_path = ['NVIDIA_FakeFace\\byimg_alignedlib_0.3\\', '100K_FAKE\\byimg_alignedlib_0.3\\']
fake_sets_name = ['TPDNE', '100F']

bat_file = open('run_all_experiments.bat','w')

for i, rsp in enumerate(real_sets_path):
    for j, fsp in enumerate(fake_sets_path):

        """ Create Dataset """

        out_path = '{droot}real2fake_{rname}_{fname}'.format(droot=data_root, rname=real_sets_name[i],
                                                                   fname=fake_sets_name[j])
        run_command = 'python create_real_to_fake_dataset.py {droot}{rpath} {droot}{fpath} {opath}'.\
            format(droot=dataset_root, rpath=rsp, fpath=fsp, opath=out_path)
        print(run_command, file=bat_file)

        """ Train Network """

        out_path = '{droot}real2fake_{rname}_{fname}'.format(droot=data_root, rname=real_sets_name[i],
                                                                   fname=fake_sets_name[j])
        run_command = 'python ../../train_pytorch_gpu.py {opath}'.format(opath=out_path)
        print(run_command, file=bat_file)

        """ Test Network """

        out_path = '{droot}real2fake_{rname}_{fname}'.format(droot=data_root, rname=real_sets_name[i],
                                                                   fname=fake_sets_name[j])
        run_command = 'python ../../test_pytorch_gpu.py {opath} {mpath}'.format(opath=out_path, mpath=out_path)
        print(run_command, file=bat_file)


