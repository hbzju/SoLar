import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
from .cifar10 import CIFAR10_Augmentention
from .cifar100 import CIFAR100_Augmentention
from .utils_algo import *
import os
import pickle

def load_cifar(args):
    batch_size=args.batch_size

    if args.dataset == 'cifar10':
        test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        test_dataset = dsets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4)
    # set test dataloader

    if args.dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        DatasetClass = CIFAR10_Augmentention
    elif args.dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        DatasetClass = CIFAR100_Augmentention
    else:
        raise NotImplementedError("Wrong dataset arguments.")
    
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    print('==> Loading local data copy in the long-tailed setup')
    data_file = "{ds}_{pr}_imb_{it}{imf}_sd{sd}.npy".format(
                                            ds=args.dataset,
                                            pr=args.partial_rate,
                                            it=args.imb_type,
                                            imf=args.imb_ratio,
                                            sd=args.seed)
    if args.hierarchical:
        data_file = 'hier_' + data_file
    save_path = os.path.join(args.data_dir, data_file)
    if not os.path.exists(save_path):
        data_dict = generate_data(args, data_file)
    else:
        data_dict = np.load(save_path, allow_pickle=True).item()
    train_data, train_labels = data_dict['train_data'], data_dict['train_labels']
    train_labels = torch.from_numpy(train_labels)
    partialY = torch.from_numpy(data_dict['partial_labels'])
    init_label_dist = torch.ones(args.num_class)/args.num_class
        
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), train_labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')
    # check partial labels
    print('Average candidate num: ', partialY.sum(1).mean())

    train_label_cnt = torch.unique(train_labels, sorted=True, return_counts=True)[-1]
    # train_label_cnt is also used for intialize Acc-shot object

    partial_matrix_dataset = DatasetClass(train_data, partialY.float(), train_labels.float())
    # generate partial label dataset
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    # estimation loader for distribution estimation
    est_dataset = DatasetClass(train_data, partialY.float(), train_labels, test_transform)
    est_loader = torch.utils.data.DataLoader(dataset=est_dataset, 
        batch_size=batch_size * 4, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True)
        
    return partial_matrix_train_loader, partialY, test_loader, est_loader, init_label_dist, train_label_cnt

def get_train_data(dataset):
    if dataset == 'cifar10':
        temp_train = dsets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    elif dataset == 'cifar100':
        temp_train = dsets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    data, labels = np.array(temp_train.data), np.array(temp_train.targets)
    return data, labels

def generate_imb_pll_data(args):
    data, labels = get_train_data(args.dataset)
    train_data, train_labels = gen_imbalanced_data(data, labels, args.num_class, args.imb_type, args.imb_factor)
    
    if args.hierarchical:
        partialY = generate_hierarchical_cv_candidate_labels(args.dataset, train_labels, args.partial_rate)
    elif args.partial_rate < 0:
        # non-uniform case for CIFAR-10
        partialY = generate_label_dependent_cv_candidate_labels(train_labels)
    else:
        partialY = generate_uniform_cv_candidate_labels(train_labels, args.partial_rate)
    return train_data, train_labels, partialY

def gen_imbalanced_data(data, targets, num_class, imb_type='exp', imb_factor=0.01, is_cub=False):

    img_max = len(data) / num_class
    img_num_per_cls = get_img_num_per_cls(num_class, imb_type, imb_factor, img_max)

    new_data = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)
    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        if is_cub:
            new_data += [data[t] for t in selec_idx]
        else:
            new_data.append(data[selec_idx, ...])          
        new_targets.extend([the_class, ] * the_img_num)
    if not is_cub:
        new_data = np.vstack(new_data)

    new_targets = torch.Tensor(new_targets).long()
    return new_data, new_targets

def get_img_num_per_cls(cls_num, imb_type, imb_factor, img_max):
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        raise NotImplementedError("You have chosen an unsupported imb type.")
    return img_num_per_cls


def get_transition_matrix(K):
    transition_matrix = np.zeros((K, K))
    for i in range(K):
        transition_matrix[i, i] = 1
        # set diagonal labels as 1
        transition_matrix[i, (i+1)%K] = 0.5
        transition_matrix[i, (i+2)%K] = 0.4
        transition_matrix[i, (i+3)%K] = 0.3
        transition_matrix[i, (i+4)%K] = 0.2
        transition_matrix[i, (i+5)%K] = 0.1
    return transition_matrix

def generate_label_dependent_cv_candidate_labels(train_labels):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = get_transition_matrix(K)
    print('==> Transition Matrix:')
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def generate_data(args, data_file):
    # First, the long-tailed learning is known to be unstable, 
    # we recommend running SoLar with a pre-processed data copy, 
    # which can be used for other baseline models as well. 
    print('''This is the first time you run this setup.
        Generating local data copies ...''')
    train_data, train_labels, partial_labels = generate_imb_pll_data(args)
    data_dict = {
            'train_data': train_data, 
            'train_labels': train_labels.numpy(), 
            'partial_labels': partial_labels.numpy()
        }
    if args.hierarchical:
        data_file = 'hier_' + data_file
    save_path = os.path.join(args.data_dir, data_file)
    with open(save_path, 'wb') as f:
        np.save(f, data_dict)
    print('local data saved at ', save_path)
    return data_dict

def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    print('==> Transition Matrix:')
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def generate_hierarchical_cv_candidate_labels(dataname, train_labels, partial_rate=0.1):
    assert dataname == 'cifar100'

    meta = unpickle('data/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]:i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]
            
        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    print('==> Transition Matrix:')
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0
    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def binarize_class(y):  
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto') 
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)     
    label = torch.from_numpy(label)
    return label