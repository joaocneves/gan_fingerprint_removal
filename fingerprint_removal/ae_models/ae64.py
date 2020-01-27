import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import numpy as np

save_list =[]

# PARAMS
batch_size = 128
num_epochs = 200
learning_rate = 1e-3
latent_code_size = 64

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

my_dataset = datasets.ImageFolder(root='W:\\DEEP_FAKE_DETECTION\\real2real\\', transform=data_transform)

dataset_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True
    )


class ConvolutionalAutoencoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, latent_code_size, 3, padding=1)

        self.trans1 = torch.nn.ConvTranspose2d(latent_code_size, 128, 3, padding=1)
        self.trans2 = torch.nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.trans3 = torch.nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.trans4 = torch.nn.ConvTranspose2d(32, 3, 3, padding=1)
        self.mp = torch.nn.MaxPool2d(2, return_indices=True)
        self.up = torch.nn.MaxUnpool2d(2)
        self.relu = torch.nn.ReLU()

    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu(x) # [?, 32, 224, 224]
        s1 = x.size()
        x, ind1 = self.mp(x) # [?, 32, 112, 112]
        x = self.conv2(x)
        x = self.relu(x) # [?, 64, 112, 112]
        s2 = x.size()
        x, ind2 = self.mp(x) # [?, 64, 56, 56]
        x = self.conv3(x)
        x = self.relu(x) # [?, 128, 56, 56]
        s3 = x.size()
        x, ind3 = self.mp(x) # [?, 128, 28, 28]
        x = self.conv4(x)
        x = self.relu(x) # [?, 32, 28, 28]

        #x = x.view(int(x.size()[0]), -1)
        #x = self.lc1(x)
        return x, ind1, s1, ind2, s2, ind3, s3

    def decoder(self, x, ind1, s1, ind2, s2, ind3, s3):
        #x = self.lc2(x)
        #x = x.view(int(x.size()[0]), 16, 16, 16)

        x = self.trans1(x)
        x = self.relu(x) # [128, 128, 28, 28]
        x = self.up(x, ind3, output_size=s3) # [128, 128, 56, 56]
        x = self.trans2(x)
        x = self.relu(x) # [128, 128, 56, 56]
        x = self.up(x, ind2, output_size=s2) # [128, 128, 112, 112]
        x = self.trans3(x)
        x = self.relu(x) # [128, 128, 112, 112]
        x = self.up(x, ind1, output_size=s1) # [128, 128, 224, 224]
        x = self.trans4(x)
        x = self.relu(x) # [128, 128, 224, 224]
        return x

    def forward(self, x):
        x, ind1, s1, ind2, s2, ind3, s3 = self.encoder(x)
        #print('x', x.shape)
        #print('s1', s1)
        #print('ind1', ind1.shape)
        #print('s2', s2)
        #print('ind2', ind2.shape)
        #l=input('Next')
        output = self.decoder(x, ind1, s1, ind2, s2, ind3, s3)
        #print('output', output.shape)
        #l=input('Done')
        return output


########################################################################
# GPU
ae = ConvolutionalAutoencoder().cuda()

# Loss & Optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    i = 0
    for data in dataset_loader:
        i = i + 1
        image, _ = data
        image = Variable(image).cuda()
        optimizer.zero_grad()

        output = ae(image)
    
        loss = loss_func(output,image)
        loss.backward()
        optimizer.step()
        if epoch+1 in save_list:
            save_image(image, './imgsae/e{}_i{}.png'.format(epoch+1, i))
            save_image(output, './imgsae/e{}_o{}.png'.format(epoch+1, i))
    print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, num_epochs, loss.data))
    torch.save(ae.state_dict(),
               'ae' + str(latent_code_size) + "\\ae" + str(latent_code_size) + "_epoch" + str(epoch) + ".pytorch")

