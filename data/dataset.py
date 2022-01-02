from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class ImageNet:

    def __init__(self, train_path, valid_path):
        self.train_path = train_path
        self.valid_path = valid_path

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
        return len(self.train_dataset)

    def get_valid_size(self):
        return len(self.valid_dataset)

    def get_train_loader(self, shuffle=True, batch_size=2, num_worker=0):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
                          pin_memory=True, sampler=None)

    def get_valid_loader(self, shuffle=False, batch_size=2, num_worker=0):
        return DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
                          pin_memory=True, sampler=None)
