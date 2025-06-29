import torch
import torch.nn as nn
import numpy as np
from fit_fun import convert_obj_score
from matplotlib import pyplot as plt
from model import model_modify
import random
import scipy.stats
from utils import set_dataset4, _preprocess2, _preprocess3, compute_metric
import torch.nn.functional as F
import os

qualitys = ['bad', 'poor', 'fair', 'good', 'excellent']

##############################general setup####################################
AGIQA3K_set = r'/public/tansongbai/dataset/AGIQA-3K'
AIGCIQA2023_set = r'/public/tansongbai/dataset/AIGCIQA2023'
PKUI2IQA_set = r'/public/tansongbai/dataset/I2IQA'

seed = 2222

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################### hyperparameter #####################
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
dataset = "AGIQA3K" #choose AGIQA3K | AIGCIQA2023 | PKUI2IQA
initial_lr1 = 1e-4
opt = 1
initial_lr2 = 5e-6
weight_decay = 0.001
num_epoch = 100
bs = 32
early_stop = 0
clip_net = 'ViT-B/32'
in_size = 512
istrain = True

#################### hyperparameter #####################

joint_texts = [f"The authenticity of the image is {d}" for d in qualitys]

##############################general setup####################################

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()
loss_fn = torch.nn.MSELoss().to(device)


def do_batch(x, con_text, aes_text):
    batch_size = x.shape[0]
    input_token_c = con_text.view(-1, 77)
    input_token_a = aes_text.view(-1, 77)
    logits_per_qua, logits_per_con, logits_per_aes = model.forward(x, input_token_c, input_token_a)
    logits_per_aes = F.softmax(logits_per_aes, dim=1)

    return logits_per_qua, logits_per_con, logits_per_aes


def eval(loader):
    model.eval()
    y_q = []
    y_pred_q = []
    y_a = []
    dis_aes = []
    y_pred_a = []
    y_c = []
    y_pred_c = []
    all_names = []
    for step, sample_batched in enumerate(loader):
        x, mos_q, mos_a, mos_c, con_tokens, aes_tokens = sample_batched['I'], sample_batched['mos_q'], \
                                                         sample_batched['mos_a'], sample_batched['mos_c'], \
                                                         sample_batched['con_tokens'], sample_batched['aes_tokens']
        names = sample_batched['img_name']
        all_names.extend(names)
        x = x.to(torch.float32).to(device)
        mos_q = mos_q.to(torch.float32).to(device)
        mos_a = mos_a.to(torch.float32).to(device)
        mos_c = mos_c.to(torch.float32).to(device)
        con_tokens = con_tokens.to(device)
        aes_tokens = aes_tokens.to(device)
        with torch.no_grad():
            logits_per_qua, logits_per_con, logits_per_aes = do_batch(x, con_tokens, aes_tokens)

            logits_per_aes = logits_per_aes.view(-1, 5)

            # quality logits:
            weight_aes = 1 * logits_per_aes[:, 0] + 2 * logits_per_aes[:, 1] + 3 * logits_per_aes[:, 2] + \
                         4 * logits_per_aes[:, 3] + 5 * logits_per_aes[:, 4]
            weight_qua = logits_per_qua[:, 0]
            weight_con = logits_per_con[:, 0]

            y_pred_q.extend(weight_qua.cpu().numpy())
            y_pred_a.extend(weight_aes.cpu().numpy())
            y_pred_c.extend(weight_con.cpu().numpy())
            dis_aes.extend(logits_per_aes.cpu().numpy())
            y_q.extend(mos_q.cpu().numpy())
            y_a.extend(mos_a.cpu().numpy())
            y_c.extend(mos_c.cpu().numpy())

    _, PLCC1, SRCC1, KRCC1 = compute_metric(np.array(y_q), np.array(y_pred_q), istrain)
    # y_pred_q = list(y_pred_q)
    # y_q = list(y_q)
    # plt.figure(figsize=(6, 6.9))
    # plt.scatter(y_pred_q, y_q, s=10, c='g')
    # x_fit = np.linspace(min(y_pred_q), max(y_pred_q), 300)
    # y_fit = list(convert_obj_score(np.array(y_pred_q), np.array(y_q), x_fit))
    # plt.plot(x_fit, y_fit, color='red', label='Fit Curve (Linear)')
    # plt.xlabel('CIA-Net', fontsize='20', loc='center')
    # plt.ylabel('MOS', fontsize='20', loc='center')
    # plt.xticks(fontsize='23')
    # plt.yticks(fontsize='23')
    # plt.xlim((0, 5))
    # plt.ylim((0, 5))
    # if not os.path.exists('Scatter_plot'):
    #     os.makedirs('Scatter_plot')
    # plt.savefig(os.path.join('Scatter_plot', 'CIA-Net' + '.png'))
    # plt.show()

    data = ''
    for i, name in enumerate(all_names):
        name_i = os.path.split(name)[-1]
        dis = dis_aes[i]
        dis_i = ''
        for s in dis:
            dis_i = dis_i + str(s) + '\t'
        data += name_i + '\t' + str(y_a[i]) + '\t' + str(y_pred_a[i]) + '\t' + dis_i + '\n'
    with open(dataset+'dis'+'.txt', 'w') as f:
        f.write(data)
    f.close()

    if mtl != 0:
        _, PLCC2, SRCC2, KRCC2 = compute_metric(np.array(y_a), np.array(y_pred_a), istrain)
    else:
        _, PLCC2, SRCC2, KRCC2 = 0.0, 0.0, 0.0, 0.0
    _, PLCC3, SRCC3, KRCC3 = compute_metric(np.array(y_c), np.array(y_pred_c), istrain)

    out = [SRCC1, PLCC1, KRCC1,
           SRCC2, PLCC2, KRCC2,
           SRCC3, PLCC3, KRCC3]
    return out

num_workers = 8

mtl_map = {'AGIQA3K': 0, 'AIGCIQA2023': 1, 'PKUI2IQA': 2}
mtl = mtl_map[dataset]

print('test on ', dataset)

for session in range(0, 10):
    model = model_modify.CIANet(device=device, clip_net=clip_net, in_size=in_size)
    model = model.to(device)

    pth = os.path.join('./checkpoints', dataset, 'MTD_IQAv6', str(session+1))
    if opt == 1:
        pth = os.path.join(pth, 'quality_best_ckpt.pt')
    elif opt == 2:
        pth = os.path.join(pth, 'authenticity_best_ckpt.pt')
    elif opt == 3:
        pth = os.path.join(pth, 'correspondence_best_ckpt.pt')
    model.load_state_dict(torch.load(pth, map_location=device)['model_state_dict'], strict=True)

    pretrain = True
    best_result = {'avg': 0.0, 'quality': 0.0, 'authenticity': 0.0, 'correspondence': 0.0}
    best_epoch = {'avg': 0, 'quality': 0, 'authenticity': 0, 'correspondence': 0}

    AGIQA3K_train_txt = os.path.join('./IQA_Database/AGIQA-3K', str(session+1), 'train.txt')
    AGIQA3K_test_txt = os.path.join('./IQA_Database/AGIQA-3K', str(session + 1), 'test.txt')

    AIGCIQA2023_train_txt = os.path.join('./IQA_Database/AIGCIQA2023', str(session + 1), 'train.txt')
    AIGCIQA2023_test_txt = os.path.join('./IQA_Database/AIGCIQA2023', str(session + 1), 'test.txt')

    PKUI2IQA_train_txt = os.path.join('./IQA_Database/PKU-I2IQA', str(session + 1), 'train.txt')
    PKUI2IQA_test_txt = os.path.join('./IQA_Database/PKU-I2IQA', str(session + 1), 'test.txt')

    AGIQA3K_train_loader = set_dataset4(AGIQA3K_train_txt, bs, AGIQA3K_set, num_workers, preprocess3, joint_texts, 0, False)
    AGIQA3K_test_loader = set_dataset4(AGIQA3K_test_txt, bs, AGIQA3K_set, num_workers, preprocess2, joint_texts, 0, True)

    AIGCIQA2023_train_loader = set_dataset4(AIGCIQA2023_train_txt, bs, AIGCIQA2023_set, num_workers, preprocess3, joint_texts, 1, False)
    AIGCIQA2023_test_loader = set_dataset4(AIGCIQA2023_test_txt, bs, AIGCIQA2023_set, num_workers, preprocess2, joint_texts, 1, True)

    PKUI2IQA_train_loader = set_dataset4(PKUI2IQA_train_txt, bs, PKUI2IQA_set, num_workers, preprocess3, joint_texts, 2, False)
    PKUI2IQA_test_loader = set_dataset4(PKUI2IQA_test_txt, bs, PKUI2IQA_set, num_workers, preprocess2, joint_texts, 2, True)

    train_loders_dir = {'AGIQA3K': AGIQA3K_train_loader, 'AIGCIQA2023': AIGCIQA2023_train_loader, 'PKUI2IQA': PKUI2IQA_train_loader}
    test_loaders_dir = {'AGIQA3K': AGIQA3K_test_loader, 'AIGCIQA2023': AIGCIQA2023_test_loader, 'PKUI2IQA': PKUI2IQA_test_loader}
    train_loaders, test_loaders = train_loders_dir[dataset], test_loaders_dir[dataset]

    out = eval(test_loaders)

    if opt == 1:
        print(out[0:3])
    elif opt == 2:
        print(out[3:6])
    elif opt == 3:
        print(out[6:9])
