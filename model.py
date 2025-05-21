import os.path

import torch

import torch.nn as nn

from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights


class Model(nn.Module):
    def __init__(self, n_classes=4):
        super(Model, self).__init__()
        self.backbone = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)  # eff2 output: 1280
        # self.backbone = resnet18(pretrained=True)
        # efficient net output: 1280
        for param in self.backbone.features[:5].parameters():
            param.requires_grad = False
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(in_features=1280, out_features=n_classes, bias=True),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


# for debugging
if __name__ == '__main__':
    model = Model(n_classes=4).to('cuda')
    # print(model)
    x = torch.randn(2, 3, 1024, 1024).cuda()  # [batch_size, 3, 224, 224]
    output = model(x)
    print(output.shape)

