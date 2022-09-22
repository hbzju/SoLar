import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .randaugment import RandomAugment
import copy

class CIFAR10_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels, transform=None):
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.transform = transform

        if self.transform is None:
            self.weak_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.strong_transform = copy.deepcopy(self.weak_transform)
            self.strong_transform.transforms.insert(1, RandomAugment(3,5))

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        if self.transform is None:
            each_image_w = self.weak_transform(self.images[index])
            each_image_s = self.strong_transform(self.images[index])
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            
            return each_image_w, each_image_s, each_label, each_true_label, index
        else:
            each_label = self.given_label_matrix[index]
            each_image = self.transform(self.images[index])
            each_true_label = self.true_labels[index]
            return each_image, each_label, each_true_label

