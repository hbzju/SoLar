from itertools import count
import torch
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from .randaugment import RandomAugment
from .utils_data import generate_uniform_cv_candidate_labels, binarize_class

from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

# from torchvision.datasets import SUN397
from torchvision.datasets.vision import VisionDataset

def load_sun397(data_dir, input_size, partial_rate, batch_size):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    test_transform = transforms.Compose(
        [
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])


    train_dataset = MySUN397(data_dir, download=False, data_tuple=None, partial_rate=partial_rate,
                            w_transform=weak_transform, s_transform=strong_transform)
    test_dataset = MySUN397(data_dir, test_transform=test_transform, 
                            data_tuple=train_dataset.get_test_data(train=False))
    est_dataset = MySUN397(data_dir, test_transform=test_transform, 
                            data_tuple=train_dataset.get_test_data(train=True), est=True)

    partialY = train_dataset._train_labels
    print('Average candidate num: ', partialY.sum(1).mean())

    train_label_cnt = torch.unique(train_dataset._labels, sorted=True, return_counts=True)[-1]
    print('Label Distribution: ', train_label_cnt)

    init_label_dist = torch.ones(397)/397.0

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    est_loader = torch.utils.data.DataLoader(dataset=est_dataset, 
        batch_size=batch_size * 4, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True)
    
    return partial_matrix_train_loader, partialY, test_loader, est_loader, init_label_dist, train_label_cnt


class MySUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.

    Args:
        root (string): Root directory of the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"

    def __init__(
        self,
        root: str,
        w_transform: Optional[Callable] = None,
        s_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        download: bool = False,
        data_tuple: tuple = None,
        est: bool = False,
        partial_rate: int = 0
    ) -> None:
        super().__init__(root, transform=None, target_transform=target_transform)
        self._data_dir = Path(self.root) / "SUN397"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]
        self.num_class = len(self.classes)
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._est = est

        if data_tuple is None:
            self._image_files = list(self._data_dir.rglob("sun_*.jpg"))
            self._image_files = np.array(self._image_files)

            self._labels = [
                self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
            ]
            self._labels = np.array(self._labels)
            counts = np.array([(self._labels == i).sum() for i in range(self.num_class)])
            labelmap = {(- counts).argsort()[i]:i for i in range(self.num_class)}
            self._labels = [
                int(labelmap[label]) for label in self._labels
            ]
            # rearrange labels according to sorted counts
            self._labels = np.array(self._labels)

            # split train_test
            test_idx = self.__split_train_test__()
            self.image_files_test = self._image_files[test_idx]
            self.labels_test = self._labels[test_idx]
            self._image_files = self._image_files[~test_idx]
            self._labels = self._labels[~test_idx]

            # generate candidate labels
            self._labels = torch.from_numpy(self._labels)
            self._train_labels = generate_uniform_cv_candidate_labels(self._labels, partial_rate)

            self.train_tuple = (self._image_files, self._labels)
            self.test_tuple = (self.image_files_test, self.labels_test)

            self.w_transform = w_transform
            self.s_transform = s_transform
            self._train = True
        else:
            (self._image_files, self._labels) = data_tuple
            self._train_labels = binarize_class(self._labels).float() if est else self._labels
            self.test_transform = test_transform
            self._train = self._est
            # if est mode, set train=True

    def get_test_data(self, train=False):
        return self.train_tuple if train else self.test_tuple

    def __split_train_test__(self):
        labels = np.array(self._labels)
        test_idx = np.zeros(len(labels)) > 1
        
        # All false array
        classes = np.arange(self.num_class)
        img_num_per_cls = np.ones(self.num_class).astype(int) * 50
        # 50 examples per class
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(labels == the_class)[0]
            np.random.shuffle(idx)
            test_idx[idx[:the_img_num]] = True
        return test_idx

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file = self._image_files[idx]
        label = self._train_labels[idx]

        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        if self._train:
            true_label = self._labels[idx]
            if not self._est:
                image_w = self.w_transform(image)
                image_s = self.s_transform(image)
                return image_w, image_s, label, true_label, idx
            else:
                return self.test_transform(image), label, true_label
        else:
            image = self.test_transform(image)
            return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        # download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)

    def __len__(self) -> int:
        return len(self._image_files)
