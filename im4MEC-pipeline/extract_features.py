import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from comparison.mocov2 import MoCoV2
from tqdm import tqdm

feature_size = 2048

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def build_moco():
    backbone = torchvision.models.resnet50(weights=None)  # in this version of ResNet50, the output shape is 2048
    ft_size = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    return MoCoV2(backbone, ft_size, projection_dim=128, K=73728, m=0.999, temperature=0.07).cuda()



if __name__ == '__main__':
    target_dir = r'PATH/TO/PATCHES'
    output_dir = r'PATH/TO/pt_files'  # CLAM-like-output, supports CLAM

    os.makedirs(output_dir, exist_ok=True)

    moco = build_moco()
    moco.load_state_dict(torch.load(r'moco_v2_best.pth'))

    resnet = moco.backbone.cuda()

    resnet.eval()

    for dir in os.listdir(target_dir):
        if not os.path.isdir(os.path.join(target_dir, dir)):
            continue
        feats = torch.tensor([])
        tbar = tqdm(os.listdir(os.path.join(target_dir, dir)))
        tbar.desc = f'Processing {dir}'
        for file in tbar:
            if not file.endswith('.jpg') and not file.endswith('.png') :
                continue
            img = Image.open(os.path.join(target_dir, dir, file)).convert('RGB')
            img = transform(img).unsqueeze(0).cuda()
            with torch.no_grad():
                feature = resnet(img).detach().cpu()
            feats = torch.cat((feats, feature), dim=0)
        torch.save(feats, os.path.join(output_dir, f'{str(dir)}.pt'))





