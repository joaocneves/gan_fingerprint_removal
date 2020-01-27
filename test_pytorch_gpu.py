import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import sys
import torchvision
from torchvision import transforms, datasets
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F


def test_model(model, dataloaders):

    preds_list = []
    labels_list = []
    running_corrects = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            # Get model outputs and calculate correct label
            outputs = model(inputs)

            #_, preds=torch.max(outputs, 1)
            #norm_preds = F.softmax(outputs).cpu().numpy()
            preds = outputs
            for pred,label in zip(preds.cpu().numpy(),labels.cpu().numpy()):
                preds_list.append(pred[0])
                labels_list.append(label)

        # statistics
        running_corrects += torch.sum((preds>0.5) == labels.data)

    acc = running_corrects.double() / len(dataloaders['test'].dataset)

    print('Acc: {:.4f}'.format(acc))

    return preds_list, labels_list


if __name__ == '__main__':
    ''' ----------------------------- PARAMS -----------------------------  '''

    n_class = 2
    model_dir = "F:\\gan_fingerprint_removal\\data\\real2fake_VF2_TPDNE"
    data_dir = "F:\\gan_fingerprint_removal\\data\\real2fake_VF2_TPDNE"
    batch_size = 32
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale = 224
    use_parallel = True
    use_gpu = True

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        model_dir = sys.argv[2]

    ''' ----------------------------- LOAD DATA -----------------------------  '''

    test_transforms = {
            'test': transforms.Compose([
            transforms.Resize(scale),
            transforms.ToTensor(),
            ]),}



    # test set
    test_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          test_transforms[x]) for x in ['test']}
    test_dataloader = {x: torch.utils.data.DataLoader(test_datasets[x], batch_size=1, num_workers=1) for x in ['test']}


    ''' ----------------------------- LOAD MODEL -----------------------------  '''


    model_after_train = torch.load(model_dir + '.pretrained.model')

    print("\n[Test model begun ....]")
    scores_list, labels_list = test_model(model_after_train, test_dataloader)

    fpr, tpr, _ = roc_curve(labels_list, scores_list)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    fpr = np.expand_dims(np.asarray(fpr),axis=1)
    tpr = np.expand_dims(np.asarray(tpr),axis=1)
    np.savetxt(data_dir + '.pretrained.roccurve', np.concatenate([fpr,tpr],axis=1))

    scores_list = np.expand_dims(np.asarray(scores_list), axis=1)
    labels_list = np.expand_dims(np.asarray(labels_list), axis=1)
    np.savetxt(data_dir + '.pretrained.scores', np.concatenate([scores_list, labels_list], axis=1))

    ''' 
    plt.figure()
    plt.plot(fpr, tpr,
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    '''

