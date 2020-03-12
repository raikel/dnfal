import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

AGE_CLS_COUNT = 60


class GenderAgeModel(torch.nn.Module):

    def __init__(self):
        super(GenderAgeModel, self).__init__()

        self.res_net = models.resnet18(pretrained=True)

        self.age_cls_unit = AGE_CLS_COUNT
        self.fc1 = nn.Linear(512, 512)
        self.age_cls_pred = nn.Linear(512, self.age_cls_unit)
        self.fc2 = nn.Linear(512, 512)
        self.gen_cls_pred = nn.Linear(512, 2)

    def get_resnet_convs_out(self, x):
        """Get outputs from convolutional layers of ResNet.

        Parameters
        ----------
        x : tensor-like
            Input image tensor.

        Returns
        -------
        x : Middle output from layer2, and final ouput from layer4.
        """

        x = self.res_net.conv1(x)  # out = [N, 64, 112, 112]
        x = self.res_net.bn1(x)
        x = self.res_net.relu(x)
        x = self.res_net.maxpool(x)  # out = [N, 64, 56, 56]

        x = self.res_net.layer1(x)  # out = [N, 64, 56, 56]
        x = self.res_net.layer2(x)  # out = [N, 128, 28, 28]
        x = self.res_net.layer3(x)  # out = [N, 256, 14, 14]
        x = self.res_net.layer4(x)  # out = [N, 512, 7, 7]

        return x  # out = [N, 512, 1 ,1]

    def get_age_gender(self, last_conv_out):

        last_conv_out = self.res_net.avgpool(last_conv_out)
        last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)

        age_pred = F.relu(self.fc1(last_conv_out))
        age_pred = F.softmax(self.age_cls_pred(age_pred), 1)

        gen_pred = F.relu(self.fc2(last_conv_out))
        gen_pred = self.gen_cls_pred(gen_pred)

        return gen_pred, age_pred

    def forward(self, x):

        last1 = self.get_resnet_convs_out(x)
        gen_pred, age_pred = self.get_age_gender(last1)

        return gen_pred, age_pred
