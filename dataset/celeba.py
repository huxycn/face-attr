import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class CelebA(Dataset):
    def __init__(self, data_root, transforms=None, attr=None, split=None, balance=1):
        self.data_root = data_root
        self.transforms = transforms
        self.attr = attr
        self.split = split
        self.img_dir = os.path.join(data_root, 'img_align_celeba')
        self.attr_file = os.path.join(data_root, 'list_attr_celeba.txt')
        self.attr_df = pd.read_csv(self.attr_file, delim_whitespace=True, header=1)[['img_name', attr]]

        pos_df = self.attr_df[self.attr_df[attr] == 1]
        neg_df = self.attr_df[self.attr_df[attr] == -1]
        nb_pos = len(pos_df)
        nb_neg = len(neg_df)
        if nb_pos > nb_neg:
            pos_df = pos_df[:int((1-balance) * nb_pos + balance * nb_neg)]
        else:
            neg_df = neg_df[:int((1-balance) * nb_neg + balance * nb_pos)]
        self.attr_df = pd.concat([pos_df, neg_df])
        self.attr_df = self.attr_df.sample(frac=1.0)  # shuffle

        # 划分数据集 train : val : test = 8 : 1 : 1
        offset1 = int(len(self.attr_df) * 0.8)
        offset2 = int(len(self.attr_df) * 0.9)
        self.train_df = self.attr_df[:offset1]
        self.val_df = self.attr_df[offset1: offset2]
        self.test_df = self.attr_df[offset2:]

        # 设置索引从0开始
        self.train_df.index = list(range(len(self.train_df)))
        self.val_df.index = list(range(len(self.val_df)))
        self.test_df.index = list(range(len(self.test_df)))

        if transforms is None:
            # mean和std值好像是固定的哦，有待研究
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if split == 'val' or split == 'test':
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])

            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        if self.split == 'train':
            item = self.train_df.iloc[index]
        elif self.split == 'val':
            item = self.val_df.iloc[index]
        elif self.split == 'test':
            item = self.test_df.iloc[index]
        else:
            raise ValueError('split: train | val | test')
        # item.values[0] -> img_name
        # item.values[1] -> attr_value
        data = Image.open(os.path.join(self.img_dir, item.values[0]))
        data = self.transforms(data)
        label = 1 if item.values[1] == 1 else 0
        return data, label

    def __len__(self):
        if self.split == 'train':
            return len(self.train_df)
        elif self.split == 'val':
            return len(self.val_df)
        elif self.split == 'test':
            return len(self.test_df)
        else:
            raise ValueError('split: train | val | test')
