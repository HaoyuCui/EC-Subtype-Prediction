import torchvision
import torch

from mocov2 import MoCoV2
from ssl_dataset import MoCoDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    backbone = torchvision.models.resnet50(pretrained=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    feature_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()

    print(f'Feature size: {feature_size}')

    """
    We follow the hyperparameters in the im4MEC:
    https://github.com/AIRMEC/im4MEC?tab=readme-ov-file#self-supervised-training
    batch-size: 288
    moco-K: 73728
    moco-m: 0.999
    lr: 0.06
    epoch: 300
    """
    model = MoCoV2(backbone, feature_size, projection_dim=128, K=73728, m=0.999, temperature=0.07).cuda()

    # load fake CIFAR-like dataset
    dataset = MoCoDataset(data_dir=None)
    loader = DataLoader(dataset, batch_size=288, shuffle=True, num_workers=2, drop_last=True)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06)

    # set learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0)

    # switch to train mode
    model.train()

    loss_min, loss = 1000, 0

    # epoch training
    for epoch in range(300):
        print(f'Epoch {epoch}')
        for i, x in enumerate(loader):
            x = x.cuda()

            # zero the parameter gradients
            model.zero_grad()

            # compute loss
            loss = model(x)

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 50 == 0:
                print(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}, Current LR: {scheduler.get_last_lr()}')

        # save model if loss is lower than previous
        if loss.item() < loss_min:
            loss_min = loss.item()
            print(f'Saving model with loss {loss_min}')
            # save model
            torch.save(model.state_dict(), 'moco_v2_best.pth')
