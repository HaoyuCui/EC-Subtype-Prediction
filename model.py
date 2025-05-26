import os.path

import torch

import torch.nn as nn

from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

from torchvision.models import resnet50, vit_b_32


def get_params_num(model):
    """
    Get the number of parameters in the model.
    :param model: torch.nn.Module
    :return: int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Model(nn.Module):
    def __init__(self, model_name, n_classes=4):
        super(Model, self).__init__()
        if model_name == 'vit':
            self.backbone = vit_b_32(weights='DEFAULT')
            self.backbone.heads.head = nn.Linear(in_features=768, out_features=n_classes, bias=True)
        elif model_name == 'resnet50':
            self.backbone = resnet50(weights='DEFAULT')
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.25),
                nn.ReLU(),
                nn.Linear(in_features=2048, out_features=n_classes, bias=True),
            )
        elif model_name == 'efficientnet':
            self.backbone = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)  # eff2 output: 1280
            # self.backbone = resnet18(pretrained=True)
            # efficient net output: 1280
            print('Number of parameters in the model: {}'.format(get_params_num(self.backbone)))
            for param in self.backbone.features[:5].parameters():
                param.requires_grad = False
            print('Number of parameters in the model after freezing first 5 layers: {}'.format(get_params_num(self.backbone)))
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.25),
                nn.ReLU(),
                nn.Linear(in_features=1280, out_features=n_classes, bias=True),
            )
        else:
            raise NotImplementedError('model {} not implemented'.format(model_name))

    def forward(self, x):
        x = self.backbone(x)
        return x


# for debugging
if __name__ == '__main__':
    model = Model(model_name='efficientnet', n_classes=4).to('cuda')
    # print(model)
    x = torch.randn(2, 3, 1024, 1024).cuda()  # [batch_size, 3, 224, 224]
    output = model(x)
    print(output.shape)

