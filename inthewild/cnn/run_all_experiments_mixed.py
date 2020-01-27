
import os

dataset_root = 'E:\\FACE_DATASETS\\'
data_root = 'E:\\gan_fingerprint_removal\\data\\'

real_sets_dev_path = ['VGG_FACE_2\\byid_alignedlib_0.3_train\\', 'CASIA-WebFace\\byid_alignedlib_0.3\\']
real_sets_dev_name = ['VF2', 'CASIA']

fake_sets_dev_path = ['NVIDIA_FakeFace\\byimg_alignedlib_0.3\\', '100K_FAKE\\byimg_alignedlib_0.3\\']
fake_sets_dev_name = ['TPDNE', '100F']

real_sets_eval_path = ['VGG_FACE_2\\byid_alignedlib_0.3_train\\', 'CASIA-WebFace\\byid_alignedlib_0.3\\']
real_sets_eval_name = ['VF2', 'CASIA']

fake_sets_eval_path = ['NVIDIA_FakeFace\\byimg_alignedlib_0.3\\', '100K_FAKE\\byimg_alignedlib_0.3\\']
fake_sets_eval_name = ['TPDNE', '100F']

bat_file = open('run_all_experiments_mixed.bat','w')

for i, rsdp in enumerate(real_sets_dev_path):
    for j, fsdp in enumerate(fake_sets_dev_path):

        for k, rsep in enumerate(real_sets_dev_path):
            for l, fsep in enumerate(fake_sets_eval_path):

                if j==l:
                    continue

                """ Create Dataset """

                out_path = '{droot}real2fake_{rsdname}_{fsdname}_train_{rsename}_{fsename}_test'.\
                    format(droot=data_root, rsdname=real_sets_dev_name[i], fsdname=fake_sets_dev_name[j],
                           rsename=real_sets_eval_name[k], fsename=fake_sets_eval_name[l])

                run_command = 'python create_real_to_fake_dataset_mixed.py {droot}{rsdpath} {droot}{fsdpath} {droot}{rsepath} {droot}{fsepath} {opath}'.\
                    format(droot=dataset_root, rsdpath=rsdp, fsdpath=fsdp, rsepath=rsep, fsepath=fsep, opath=out_path)

                print(run_command, file=bat_file)

                """ Test Model """

                model_path = '{droot}real2fake_{rname}_{fname}'.format(droot=data_root, rname=real_sets_dev_name[i],
                                                                   fname=fake_sets_dev_name[j])

                run_command = 'python ../../test_pytorch_gpu.py {opath} {mpath}'.format(opath=out_path, mpath=model_path)
                print(run_command, file=bat_file)


