
import sys, os
sys.path.insert(0, '../../../../vision/')
sys.path.append('../../../../../pytorch-segmentation-detection/')

# Use second GPU -pytorch-segmentation-detection- change if you want to use a first one
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torch.autograd import Variable
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated

import numpy as np

import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_path = sys.argv[2]

valid_transform = transforms.Compose(
                [
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

img_not_preprocessed = Image.open(img_path).convert('RGB')

img = valid_transform(img_not_preprocessed)

img = img.unsqueeze(0)

img = Variable(img.to(device))

model_path = sys.argv[1]

fcn = resnet_dilated.Resnet34_8s(num_classes=21, dilation=False)
fcn.load_state_dict(torch.load(model_path, map_location = torch.device(device)))
fcn.to(device)
fcn.eval()

res = fcn(img)

_, tmp = res.squeeze(0).max(0)

segmentation = tmp.data.cpu().numpy().squeeze()

plt.figure("image")
plt.imshow(img_not_preprocessed)

plt.figure("segmentation")
plt.imshow(segmentation)

plt.imsave("segmentation.png", segmentation)

def benchmark_fcn():

    img = valid_transform(img_not_preprocessed)

    img = img.unsqueeze(0)

    img = Variable(img.to(device))

    res = fcn(img)

img_not_preprocessed.size

tic = time.perf_counter()
benchmark_fcn()
toc = time.perf_counter()
print("inference time: ", toc-tic, "s")

plt.show()
