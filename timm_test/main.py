import timm
import torch

torch.backends.cudnn.benchmark = True


model = timm.create_model('resnet18', pretrained=False)
