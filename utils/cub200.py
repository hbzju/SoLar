import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from .randaugment import RandomAugment
from .utils_data import *
import copy

def load_cub200(data_dir, input_size, partial_rate, batch_size, imb_factor):
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

    train_dataset = CUB200(root=data_dir,
                                download=False,
                                train=True, 
                                w_transform=weak_transform, 
                                s_transform=strong_transform, 
                                partial_rate=partial_rate,
                                imb_factor=imb_factor)
    est_dataset = CUB200(root=data_dir,
                                download=False,
                                train=True, 
                                test_transform=test_transform, 
                                partial_rate=partial_rate,
                                imb_factor=imb_factor,
                                est=True,
                                given_data=(train_dataset._train_data, 
                                            train_dataset._train_labels,
                                            train_dataset.true_labels)) 
    test_dataset = CUB200(root='~/cub200/',
                               download=False, 
                               train=False, 
                               test_transform=test_transform, 
                               partial_rate=partial_rate,
                               imb_factor=imb_factor)
    partialY = train_dataset._train_labels
    print('Average candidate num: ', partialY.sum(1).mean())

    train_label_cnt = torch.unique(train_dataset.true_labels, sorted=True, return_counts=True)[-1]
    print('Label Distribution: ', 100 * train_label_cnt/train_label_cnt.sum())

    init_label_dist = torch.ones(200)/200.0

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=1,
        pin_memory=True,
        drop_last=True)
    est_loader = torch.utils.data.DataLoader(dataset=est_dataset, 
        batch_size=batch_size * 4, 
        shuffle=False, 
        num_workers=1,
        pin_memory=True)
    
    return partial_matrix_train_loader, partialY, test_loader, est_loader, init_label_dist, train_label_cnt


class CUB200(torch.utils.data.Dataset):
    """CUB200 dataset.
    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """
    def __init__(self, root, train=True, w_transform=None,s_transform=None, test_transform=None,target_transform=None,
                 download=False,partial_type='binomial', partial_rate=0.1, imb_factor=0, est=False, given_data=None):
        """Load the dataset.
        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train
        self._est = est
        self.w_transform = w_transform
        self.s_transform = s_transform
        self.test_transform = test_transform
        self._target_transform = target_transform
        self.partial_rate=partial_rate
        self.partial_type=partial_type
        if self._checkIntegrity():
            print('Files already downloaded and verified.')
        elif download:
            url = ('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'
                   'CUB_200_2011.tgz')
            self._download(url)
            self._extract()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        # Now load the picked data.
        if given_data is not None:
            (self._train_data, self._train_labels, self.true_labels) = given_data
            # using given data
        elif self._train:
            self._train_data, self._train_labels = pickle.load(open(
                os.path.join(self._root, 'processed/train.pkl'), 'rb'))
            assert (len(self._train_data) == 5994
                    and len(self._train_labels) == 5994)
            self._train_labels = np.array(self._train_labels)

            # generate long-tail data
            self._train_data, self._train_labels = gen_imbalanced_data(self._train_data, self._train_labels, num_class=200, imb_type='exp', imb_factor=imb_factor, is_cub=True)

            # generate partial labeled data
            self.true_labels = copy.deepcopy(self._train_labels)
            if self.partial_rate != 0.0:
                self._train_labels = generate_uniform_cv_candidate_labels(self._train_labels, partial_rate)
                print('-- Average candidate num: ', self._train_labels.sum(1).mean(), self.partial_rate)
            else:
                self._train_labels = binarize_class(self._train_labels).float()
        else:
            self._test_data, self._test_labels = pickle.load(open(
                os.path.join(self._root, 'processed/test.pkl'), 'rb'))
            assert (len(self._test_data) == 5794
                    and len(self._test_labels) == 5794)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.
        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        # Doing this so that it is consistent with all other datasets.
        image = Image.fromarray(image)
 
        if self._target_transform is not None:
            target = self._target_transform(target)

        if self._train:
            each_label = target
            each_true_label = self.true_labels[index]
            if not self._est:
                each_image_w = self.w_transform(image)
                each_image_s = self.s_transform(image)
                return each_image_w, each_image_s, each_label, each_true_label, index
            else:
                return self.test_transform(image), each_label, each_true_label
        else:
            image = self.test_transform(image)
            return image, target

    def __len__(self):
        """Length of the dataset.
        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check whether we have already processed the data.
        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (
            os.path.isfile(os.path.join(self._root, 'processed/train.pkl'))
            and os.path.isfile(os.path.join(self._root, 'processed/test.pkl')))

    def _download(self, url):
        """Download and uncompress the tar.gz file from a given URL.
        Args:
            url, str: URL to be downloaded.
        """
        import six.moves
        import tarfile

        raw_path = os.path.join(self._root, 'raw')
        processed_path = os.path.join(self._root, 'processed')
        if not os.path.isdir(raw_path):
            os.mkdir(raw_path, mode=0o775)
        if not os.path.isdir(processed_path):
            os.mkdir(processed_path, mode=0x775)

        # Downloads file.
        fpath = os.path.join(self._root, 'raw/CUB_200_2011.tgz')
        try:
            print('Downloading ' + url + ' to ' + fpath)
            six.moves.urllib.request.urlretrieve(url, fpath)
        except six.moves.urllib.error.URLError:
            if url[:5] == 'https:':
                self._url = self._url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.')
                print('Downloading ' + url + ' to ' + fpath)
                six.moves.urllib.request.urlretrieve(url, fpath)

        # Extract file.
        cwd = os.getcwd()
        tar = tarfile.open(fpath, 'r:gz')
        os.chdir(os.path.join(self._root, 'raw'))
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def _extract(self):
        """Prepare the data for train/test split and save onto disk."""
        image_path = os.path.join(self._root, 'CUB_200_2011/images/')
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(os.path.join(
            self._root, 'CUB_200_2011/images.txt'), dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(os.path.join(
            self._root, 'CUB_200_2011/train_test_split.txt'), dtype=int)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for id_ in range(id2name.shape[0]):
            image = Image.open(os.path.join(image_path, id2name[id_, 1]))
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            # Convert gray scale image to RGB image.
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[id_, 1] == 1:
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_data.append(image_np)
                test_labels.append(label)

        pickle.dump((train_data, train_labels),
                    open(os.path.join(self._root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self._root, 'processed/test.pkl'), 'wb'))
