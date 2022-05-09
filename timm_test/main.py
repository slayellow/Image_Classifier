import timm
from timm.data.transforms_factory import create_transform
import torch

torch.backends.cudnn.benchmark = True


model = timm.create_model('resnet18', pretrained=False)
