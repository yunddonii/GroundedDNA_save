from typing import Callable, Optional, T_co, Any
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

    

class ImgRtvDataset(Dataset):
    def __init__(self, root,
                 img_transform=None, target_transform=None, mode ="train", setting_name = "setting1", return_index = False, return_paired_aug_img=False):
        self.loader = self.default_loader
        self.root = os.path.expanduser(root)
        
        self.default_transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])    
        self.transform = img_transform
        self.target_transform = target_transform

        if mode == "train":
            self.base_folder = 'train.txt'
        elif mode == "test" or mode == "query":
            self.base_folder = 'test.txt'
        elif mode == "database":
            self.base_folder = 'database.txt'
        else:
            raise ValueError

        self.img_paths = []
        self.img_labels = []

        filename = os.path.join(self.root, setting_name, self.base_folder)

        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                pos_tmp = lines.split()[0]    
                pos_tmp = os.path.join(self.root, pos_tmp)
                label_tmp = lines.split()[1:]
                self.img_paths.append(pos_tmp)
                self.img_labels.append( list(map(int,label_tmp)))
        
        self.img_paths = np.array(self.img_paths)

        num_classes = len(self.img_labels[0])

        # self.img_labels = torch.tensor(self.img_labels)
        self.img_labels = np.array(self.img_labels, dtype=np.int64) #np.float
        self.img_labels.reshape((-1, num_classes))
        
        self.return_index = return_index
        if return_index:
            self.index_table = torch.arange(0, len(self))
        self.return_paired_aug_img = return_paired_aug_img
        if return_paired_aug_img:
            assert img_transform != None
        
    
    def default_loader(self, path):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return accimage_loader(path)
        else:
            return pil_loader(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img_name, target = self.img_paths[index], self.img_labels[index]
        img = self.loader(img_name)

        img_i = self.default_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        out = {}
        out['label'] = target
        out['img'] = img_i
        
        if self.return_index:
            out['idx'] = self.index_table[index]
            
        if self.return_paired_aug_img:
            assert self.transform  is not None
            out['img_tr1'] = self.transform(img)
            out['img_tr2'] = self.transform(img)
            
        return out

    def __len__(self):
        return len(self.img_paths)
        
def get_idx_for_uniform_sampling(dataset, n_class, n_samples_per_class,offset=0): # convert_train_test=False
    # train, test data를 모두 합침
    L = np.array(dataset.targets)
    first = True

    for label in range(n_class):
        # 레이블이 "label"에 해당하는 데이터의 인덱스를 찾음
        index = np.where(L == label)[0]

        # 인덱스를 랜덤으로 섞음
        N = index.shape[0]
        np.random.seed(0)
        perm = np.random.permutation(N)
        index = index[perm]

        # 레이블이 "label"에 해당하는 데이터 N개를 무작위로 선택 
        index = index[offset:offset+n_samples_per_class]

        # array에 추가
        if first:
            index_array = index
        else:
            index_array = np.concatenate((index_array, index))

        first = False

    return index_array


class ImgRtvCIFAR10(CIFAR10):
    def __init__(self,  root,
                 img_transform=None, target_transform=None, mode ="train", setting_name = "setting1", return_index=False,return_paired_aug_img=False) -> None:
       
        self.default_transform = transforms.Compose([transforms.Resize((224,224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

        if setting_name == "setting1":
            # Setting 1 - (MBE 논문 setting 2)
            # Train: 클래스당 500개를 Trainset에서 샘플링 (5k)
            # Test:  클래스당 100개를 Testset에서 샘플링 (1k)
            # Database: Testset 1k개 제외한 전체 (5.8k)

            if mode == "train":
                train = True; transform=img_transform
            elif mode == "test":
                train= False; transform=img_transform
            elif mode == "database":
                train= False; transform=img_transform
            else:
                raise KeyError
            
            super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=True)
            
            if mode == 'train':
                idx = get_idx_for_uniform_sampling(self, 10, n_samples_per_class=500)
            elif mode == "test": 
                idx = get_idx_for_uniform_sampling(self, 10, n_samples_per_class=100)
            elif mode == "database":
                trainset = CIFAR10(root, train=True, transform=img_transform, target_transform=target_transform)
                testset = CIFAR10(root, train=False, transform=img_transform, target_transform=target_transform)
                self.data = np.concatenate((trainset.data, testset.data))
                self.targets = np.concatenate(( np.array(trainset.targets), np.array(testset.targets)))
                
                # Test에서 뽑힌 N개 제외한 인덱스를 구함
                idx = get_idx_for_uniform_sampling(testset, 10, n_samples_per_class= 900, offset=100)
                idx = np.concatenate( (np.arange(0, len(trainset)), idx+len(trainset)))
                
                
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]
            
            

        elif setting_name == "setting2":         
            # Setting 2 -  (MBE 논문 setting 1)
            # Test: 전체 Testset (10k)/ Training: 전체 Trainset (50k)/ Database: 전체 Trainset(50k)
            if mode == "train":
                train=True; transform=img_transform
            elif mode == "test":
                train=False; transform=img_transform
            elif mode == "database":
                train=True; transform=img_transform
            else:
                raise KeyError
            super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=True)
        else:
            raise KeyError
        
        self.return_index = return_index
        
        if return_index:
            self.index_table = torch.arange(0, len(self))
        
        self.return_paired_aug_img = return_paired_aug_img
        if return_paired_aug_img:
            assert img_transform != None
                
    
    def __getitem__(self, index: Any) -> T_co:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        img_i = self.default_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        out = {}
        out['label'] = np.eye(10,dtype=np.int64)[target]
        out['img'] = img_i
        
        if self.return_index:
            out['idx'] = self.index_table[index]
            
        if self.return_paired_aug_img:
            assert self.transform  is not None
            out['img_tr1'] = self.transform(img)
            out['img_tr2'] = self.transform(img)
        
        return out

    




def load_dataset(dataset_dir, dataset_name, setting,train_transform=None, test_transform=None, load_train=True, load_test=True, load_database=True, return_index=False, return_paired_aug_img=False):
    root = os.path.join(dataset_dir, dataset_name)
    if load_train:
        train_dataset = DATASET[dataset_name](root, train_transform, None, "train", setting, return_index=return_index, return_paired_aug_img=return_paired_aug_img)
    else:
        train_dataset = None
    
    if load_test:
        test_dataset = DATASET[dataset_name](root, test_transform, None, "test", setting, return_index=return_index,return_paired_aug_img=return_paired_aug_img)
    else:
        test_dataset = None
    
    if load_database:
        database_dataset = DATASET[dataset_name](root, test_transform, None, "database", setting,return_index=return_index,return_paired_aug_img=return_paired_aug_img)
    else:
        database_dataset = None

    return train_dataset, test_dataset, database_dataset

        

DATASET = { 
           'DEFAULT': None,
           'CIFAR10': ImgRtvCIFAR10, 
            'Flickr25k': ImgRtvDataset, 
            'ImageNet100': ImgRtvDataset,
            'MSCOCO': ImgRtvDataset, 
            'NUSWIDE': ImgRtvDataset, 
            }

NUM_CLASS = { 
             'DEFAULT': {'setting1' : None},
             'CIFAR10': {'setting1' : 10,
                         'setting2'  : 10}, 
            'Flickr25k': {'setting1': 24}, 
            'ImageNet100':{'setting1': 100}, 
            'MSCOCO': {'setting1': 80}, 
            'NUSWIDE': {'setting1'  : 21,
                        'setting2'  : 10}, 
                
            }

MULTI_LABEL = { 
            'DEFAULT'   : None,
            'CIFAR10'   : False, 
            'Flickr25k' : True, 
            'ImageNet100': False,
            'MSCOCO'    : True, 
            'NUSWIDE'   : True, 
            }


if __name__ == "__main__":
    # dataset = ImgRtvDataset("./ImageHashing/dataset/MSCOCO", img_transform=None,mode="test", setting_name="setting1")
    # dataset = ImgRtvCIFAR10("./ImageHashing/dataset/CIFAR10", img_transform=None,mode="database", setting_name="setting1")
    
    for dataset in ['Flickr25k', 'ImageNet100', 'MSCOCO', 'NUSWIDE']:
        print(dataset)
        trainset, testset, databaseset = load_dataset("dataset", dataset, setting="setting1", train_transform=None, test_transform=None)
        
        trainloader = DataLoader(trainset, batch_size=64, pin_memory=True)
        testloader = DataLoader(testset, batch_size=64, pin_memory=True)
        databaseloader = DataLoader(databaseset, batch_size=64, pin_memory=True)


        for epoch in range(1):
            print(f"Train:{len(trainloader.dataset)}")
            for i in trainloader:
                print(i[-1].shape)
                break
            print(f"Test:{len(testloader.dataset)}")
            for i in testloader:
                break
            print(f"Database:{len(databaseloader.dataset)}")
            for i in databaseloader:
                break

            print(epoch)

