import torch.nn as nn
import torchvision.models as models


class ResNetDilated_s(nn.Module):
    def __init__(self, resnet_base, resnet_name="resnet", dilation=False, num_classes=1000):
        """ initialised a dilated ResNet
            resnet_base: function that returns a nn.ResNet (resnet18, resnet34, resnet50, resnet101)
        """
        super(ResNetDilated_s, self).__init__()

        # load fully convolutional ResNet with optional dilation
        resnet = resnet_base(fully_conv=True, replace_stride_with_dilation=[dilation]*3)

        if not isinstance(resnet, models.ResNet):
            raise AttributeError("base resnet (type: " + str(type(resnet)) + ") is not of " + str(models.ResNet))

        # Randomly initialize the 1x1 Conv scoring layer
        resnet.fc = nn.Conv2d(resnet.inplanes, num_classes, 1)
        self._normal_initialization(resnet.fc)

        # dynamic variable name for loading weights
        self.name = resnet_name
        setattr(self, resnet_name, resnet)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        x = getattr(self, self.name)(x)
        # x = self.resnet(x)
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        return x
