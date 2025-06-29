import torch
import functools
import numpy as np
import pandas as pd
import clip
import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset4(Dataset):
    def __init__(self, txt_file, img_dir, preprocess, test, joint_text, is_aigc2013=False, get_loader=get_default_img_loader):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            img_dir (string): Directory of the images.
            preprocess (callable, optional): transform to be applied on a sample.
        """
        self.img_paths = []
        self.is_aigc2013 = is_aigc2013
        self.mos1 = []
        self.mos2 = []
        self.mos3 = []
        self.con_text_prompts = []
        self.aes_text_prompts = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:  # 读取label文件
                self.img_paths.append(os.path.join(img_dir, line.split('\t')[0]))
                self.mos1.append(float(line.split('\t')[1]))
                if not is_aigc2013:
                    self.mos2.append(float(line.split('\t')[2]))
                    self.mos3.append(float(line.split('\t')[3]))
                    self.con_text_prompts.append([line.split('\t')[4]])
                else:
                    self.mos2.append(0.0)
                    self.mos3.append(float(line.split('\t')[2]))
                    self.con_text_prompts.append([line.split('\t')[3]])
                self.aes_text_prompts.append(joint_text)
        # print('%d txt data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test

        self.img_paths = np.array(self.img_paths)
        self.con_text_prompts = np.array(self.con_text_prompts)
        self.aes_text_prompts = np.array(self.aes_text_prompts)
        self.mos1 = np.array(self.mos1)
        self.mos2 = np.array(self.mos2)
        self.mos3 = np.array(self.mos3)

    def __getitem__(self, index):
        image_name = self.img_paths[index]
        I = self.loader(image_name)
        I = self.preprocess(I)
        con_text_prompts = self.con_text_prompts[index]
        aes_text_prompts = self.aes_text_prompts[index]
        con_tokens = torch.cat([clip.tokenize(prompt) for prompt in con_text_prompts])
        aes_tokens = torch.cat([clip.tokenize(prompt) for prompt in aes_text_prompts])
        mos_q = self.mos1[index]
        mos_a = self.mos2[index]
        mos_c = self.mos3[index]
        sample = {'I': I, 'mos_q': mos_q, 'mos_a': mos_a, 'mos_c': mos_c, 'con_tokens': con_tokens, 'aes_tokens': aes_tokens, 'img_name': image_name}
        return sample

    def __len__(self):
        return len(self.img_paths)

