from tarfile import is_tarfile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from timm.data import ImageDataset, create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset


class ImageNet:

    def __init__(self, train_path, valid_path, is_timm=False):
        self.train_path = train_path
        self.valid_path = valid_path
        self.is_timm = is_timm

        if is_timm:
            self.train_dataset = ImageDataset(self.train_path)
            
            self.valid_dataset = ImageDataset(self.valid_path)
        else:
            normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
            self.train_dataset = datasets.ImageFolder(
                self.train_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalization
                ])
            )
            self.valid_dataset = datasets.ImageFolder(
                self.valid_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalization
                ])
            )

    def get_train_size(self):
        return 0 if self.is_timm else len(self.train_dataset)

    def get_valid_size(self):
        return 0 if self.is_timm else len(self.valid_dataset)

    def get_train_loader(self, shuffle=True, batch_size=2, num_worker=0):
        if self.is_timm:
            return create_loader(self.train_dataset, input_size=(3, 224, 224), batch_size=batch_size, is_training=True, num_workers=num_worker, pin_memory=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), worker_seeding=42)
        else:
            return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
                          pin_memory=True, sampler=None)

    def get_valid_loader(self, shuffle=False, batch_size=2, num_worker=0):
        if self.is_timm:
            return create_loader(self.valid_dataset, input_size=(3, 224, 224), batch_size=batch_size, is_training=False, num_workers=num_worker, pin_memory=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        else:
            return DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
                          pin_memory=True, sampler=None)

