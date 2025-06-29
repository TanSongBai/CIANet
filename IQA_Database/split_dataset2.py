import os
import random
import csv
import scipy.io
import pandas as pd

class split_dataset:
    def __init__(self, path, idx):
        self.path = path
        self.index = idx

    def AGIQA(self):
        csv_path = os.path.join(self.path, 'data.csv')
        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = list(reader)[1:]
            names = [i[0] for i in data]
            texts = [i[1] for i in data]
            mos_quality = [i[5] for i in data]
            mos_align = [i[7] for i in data]
        prompts = sorted(set(texts))
        select_prompts = [prompts[i] for i in self.index]
        positions = [pos for pos, value in enumerate(texts) if value in select_prompts]
        data = []
        for i in positions:
            data.append(os.path.join('all', names[i]) + '\t' + mos_quality[i] + '\t' + mos_align[i] + '\t' + texts[i])
        return data

    def AIGCIQA2023(self):
        mat_path = os.path.join(self.path, 'DATA', 'MOS')
        mos_quality = scipy.io.loadmat(os.path.join(mat_path, 'mosz1.mat'))['MOSz'].flatten()
        mos_authenticity = scipy.io.loadmat(os.path.join(mat_path, 'mosz2.mat'))['MOSz'].flatten()
        mos_align = scipy.io.loadmat(os.path.join(mat_path, 'mosz3.mat'))['MOSz'].flatten()
        df = pd.read_excel(os.path.join(self.path, 'AIGCIQA2023_Prompts.xlsx'), header=None)
        prompts = df.iloc[:, 2].tolist()
        texts = [element for element in prompts for _ in range(4)] * 6
        names = []
        for i in range(len(mos_quality)):
            names.append(str(i)+'.png')
        data = []
        for i in self.index:
            data.append(os.path.join('Image', 'allimg', names[i]) + '\t' + str(mos_quality[i]) + '\t' +
                        str(mos_authenticity[i]) + '\t' + str(mos_align[i]) + '\t' + str(texts[i]))
        return data

    def PKUI2IQA(self):
        df = pd.read_excel(os.path.join(self.path, 'I2IQA_annotation.xlsx'))
        texts = df.iloc[:, 1].tolist()
        img_prompts = df.iloc[:, 0].tolist()
        names = df.iloc[:, 2].tolist()
        mos_quality = df.iloc[:, 3].tolist()
        mos_authenticity = df.iloc[:, 4].tolist()
        mos_align = df.iloc[:, 5].tolist()
        data = []
        for i in self.index:
            data.append(os.path.join('Generated_image/All', names[i]) + '\t' + str(mos_quality[i]) + '\t' + str(
                mos_authenticity[i]) + '\t' + str(mos_align[i]) + '\t' + str(texts[i]))
        return data

if __name__ == '__main__':

    dataset = 'AIGCIQA2023'
    index = {'AGIQA-3K': range(0, 300),
             'AIGCIQA2023': range(0, 2400),
             'PKU-I2IQA' : range(0, 1600)}

    save = os.path.join(dataset)
    path_dir = {'AGIQA-3K': '/public/tansongbai/dataset/AGIQA-3K',
             'AIGCIQA2023': '/public/tansongbai/dataset/AIGCIQA2023',
             'PKU-I2IQA' : '/public/tansongbai/dataset/I2IQA'}
    for split in range(1, 11):
        random.seed(split)
        save_txt = os.path.join(save, f'{split}')
        if not os.path.exists(save_txt):
            os.makedirs(save_txt)
        datapath = path_dir[dataset]
        index_ = list(index[dataset])
        random.shuffle(index_)
        if dataset == 'AGIQA-3K':
            train_index = index_[:int(0.8*len(index_))]
            test_index = index_[int(0.8*len(index_)):]
        elif dataset == 'AIGCIQA2023':
            index_ = sorted(index_)
            result = [index_[i:i+4] for i in range(0, len(index_), 4)]
            random_list = [random.randint(0, 3) for _ in range((int(len(index_)/4)))]
            test_index = [k[random_list[idx]] for idx, k in enumerate(result)]
            train_index = list(set(index_) - set(test_index))
        elif dataset == 'PKU-I2IQA':
            index_ = sorted(index_)
            result = [index_[i:i+4] for i in range(0, len(index_), 4)]
            random_list = [random.randint(0, 3) for _ in range((int(len(index_)/4)))]
            test_index = [k[random_list[idx]] for idx, k in enumerate(result)]
            train_index = list(set(index_) - set(test_index))

        if dataset == 'AGIQA-3K':
            train_data = split_dataset(datapath, train_index).AGIQA()
            test_data = split_dataset(datapath, test_index).AGIQA()

        if dataset == 'AIGCIQA2023':
            train_data = split_dataset(datapath, train_index).AIGCIQA2023()
            test_data = split_dataset(datapath, test_index).AIGCIQA2023()

        if dataset == 'PKU-I2IQA':
            train_data = split_dataset(datapath, train_index).PKUI2IQA()
            test_data = split_dataset(datapath, test_index).PKUI2IQA()

        with open(os.path.join(save_txt, 'train.txt'), 'a') as f:
            for data in train_data:
                f.write(data + '\n')
            f.close()
        with open(os.path.join(save_txt, 'test.txt'), 'a') as f:
            for data in test_data:
                f.write(data + '\n')
            f.close()