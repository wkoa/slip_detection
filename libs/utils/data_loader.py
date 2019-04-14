import os
import re
import random
import cv2
import torch
from torch.utils import data

class Tactile_Vision_dataset(data.Dataset):
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.label_files = []
        self.train_data = []
        for root, dirs, files in os.walk(data_path, topdown=True):
            for file in files:
                if file.endswith('.dat'):
                    self.label_files.append(os.path.join(root, file))

        pat = re.compile(r'object([0-9]+)_result')
        for label_file in self.label_files:
            idx = pat.search(label_file).group(1)
            fp = open(label_file, 'r')
            lines = fp.readlines()
            self.train_data.extend([line.replace('\n','') + ' ' + idx for line in lines])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        train_data = self.train_data[index]
        output_tacitle_imgs = []
        output_rgb_imgs = []

        train_data = train_data.split(' ')
        object_id = train_data[-1]
        id_1 = train_data[-2]
        id_2 = train_data[-3]
        status = int(train_data[2][0]) # Label
        label = torch.tensor([status]).long()
        label = torch.squeeze(label)

        path = os.path.join(self.data_path, 'object' + object_id, id_2 + '_mm')
        rgb_img_paths = []
        for root, dirs, files in os.walk(path, topdown=True):
            for file in files:
                if ("external_" in file) and file.endswith('.jpg'):
                    rgb_img_paths.append(os.path.join(root, file))

        random_num = []
        while(len(random_num)<8):
            x = random.randint(0, len(rgb_img_paths) - 1)
            if x not in random_num:
                random_num.append(x)
        random_num.sort()

        for i in random_num:
            rgb_img_path = rgb_img_paths[i]
            cor_tacitle_img_path = rgb_img_path.replace('external', 'gelsight')
            rgb_img = cv2.imread(rgb_img_path)
            size = rgb_img.shape
            tacitle_img = cv2.imread(cor_tacitle_img_path)
            rgb_img_tensor = torch.from_numpy(rgb_img.reshape(size[2], size[0], size[1])).float()
            tacitle_img_tensor = torch.from_numpy(tacitle_img.reshape(size[2], size[0], size[1])).float()
            output_rgb_imgs.append(rgb_img_tensor)
            output_tacitle_imgs.append(tacitle_img_tensor)

        return output_rgb_imgs, output_tacitle_imgs, label

