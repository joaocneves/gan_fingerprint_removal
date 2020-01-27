
import os
import cv2
import sys
import glob
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

class Autoencoder(torch.nn.Module):

    def __init__(self, latent_code_size):
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

def image_loader(image_name, input_shape):

    loader = transforms.Compose([transforms.Scale(input_shape), transforms.ToTensor()])

    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image

def recon_image(model, image_name, input_shape):

    image = image_loader(image_name, input_shape)
    image_recon = model(image.clone().detach())

    image_recon_cv = image_recon[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image_recon_cv = cv2.cvtColor(image_recon_cv, cv2.COLOR_RGB2BGR)

    return image_recon_cv

''' ----------------------------- PARAMS -----------------------------  '''

ae_epochs = 100
latent_code_size = 32
original_image_path = 'F:\\FACE_DATASETS\\NVIDIA_FakeFace\\byimg_alignedlib_0.3'
transformed_image_path = 'NVIDIA_FakeFace\\byimg_alignedlib_0.3_nofingerprint'


model_dir = 'fingerprint_removal\\ae_models\\ae{0}\\ae{0}_epoch{1}.pytorch'.format(latent_code_size, ae_epochs-1)

if len(sys.argv) > 1:
    original_image_path = sys.argv[1]
    transformed_image_path = sys.argv[2]
    model_dir = sys.argv[3]

# load ae
model = Autoencoder(latent_code_size)
model.load_state_dict(torch.load(model_dir))

# ----------------------- LOAD DATA ---------------------------- #

image_list = glob.glob(original_image_path + '\\*.jpg')


# ----------------------- CREATE FOLDERS ---------------------------- #

try:
    os.makedirs(transformed_image_path)
except OSError:
    print("Creation of the directory %s failed" % transformed_image_path)
else:
    print("Successfully created the directory %s " % transformed_image_path)

# ----------------------- TRANSFORM IMAGES ---------------------------- #


for image_path in image_list:

    output_image_path = image_path.replace(original_image_path, transformed_image_path)


    im = recon_image(model, image_path, 224)

    cv2.imwrite(output_image_path, im)





