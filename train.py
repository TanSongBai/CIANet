import torch
import numpy as np
from model import model_modify
import random
from torch.utils.tensorboard import SummaryWriter as sum_writer
from Loss_func import loss_m3
from utils import set_dataset4, _preprocess2, _preprocess3, convert_models_to_fp32, compute_metric
import torch.nn.functional as F
import os
from tqdm import tqdm
import pickle

### the best outcome
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
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
datasets = ["AGIQA3K"] #choose AGIQA3K | AIGCIQA2023 | PKUI2IQA
initial_lr1 = 1e-4
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

def freeze_model(f_model, opt):
    f_model.logit_scale.requires_grad = False
    if opt == 0: #do nothing
        return
    elif opt == 1: # freeze text encoder
        for p in f_model.token_embedding.parameters():
            p.requires_grad = False
        for p in f_model.transformer.parameters():
            p.requires_grad = False
        f_model.positional_embedding.requires_grad = False
        f_model.text_projection.requires_grad = False
        for p in f_model.ln_final.parameters():
            p.requires_grad = False
    elif opt == 2: # freeze visual encoder
        for p in f_model.visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in f_model.parameters():
            p.requires_grad = False
    elif opt == 4:
        for p in f_model.parameters():
            p.requires_grad = True

def do_batch(x, con_text, aes_text):
    batch_size = x.shape[0]
    input_token_c = con_text.view(-1, 77)
    input_token_a = aes_text.view(-1, 77)
    logits_per_qua, logits_per_con, logits_per_aes = model.forward(x, input_token_c, input_token_a)
    logits_per_aes = F.softmax(logits_per_aes, dim=1)

    return logits_per_qua, logits_per_con, logits_per_aes

def train(model, best_result, best_epoch):
    model.eval()

    global early_stop
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    # if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
    #     scheduler.step()
    #     print(optimizer.state_dict()['param_groups'][0]['lr'])

    for idx, sample_batched in enumerate(tqdm(train_loaders)):

        x, mos_q, mos_a, mos_c, con_tokens, aes_tokens = sample_batched['I'], sample_batched['mos_q'], \
                                                         sample_batched['mos_a'], sample_batched['mos_c'], \
                                                         sample_batched['con_tokens'], sample_batched['aes_tokens']

        img_name = sample_batched['img_name']
        x = x.to(torch.float32).to(device)
        mos_q = mos_q.to(torch.float32).to(device)
        mos_a = mos_a.to(torch.float32).to(device)
        mos_c = mos_c.to(torch.float32).to(device)
        con_tokens = con_tokens.to(device)
        aes_tokens = aes_tokens.to(device)

        optimizer.zero_grad()
        logits_per_qua, logits_per_con, logits_per_aes = do_batch(x, con_tokens, aes_tokens)
        logits_per_aes = logits_per_aes.view(-1, 5)

        #quality logits:
        weight_aes = 1 * logits_per_aes[:, 0] + 2 * logits_per_aes[:, 1] + 3 * logits_per_aes[:, 2] + \
                     4 * logits_per_aes[:, 3] + 5 * logits_per_aes[:, 4]
        weight_qua = logits_per_qua[:, 0]
        weight_con = logits_per_con[:, 0]
        loss_q = loss_fn(weight_qua, mos_q.detach())
        loss_c = loss_m3(weight_con, mos_c.detach())

        if pretrain:
            total_loss = loss_q
        else:
            if mtl == 0:
                total_loss = loss_q + loss_c
            elif mtl == 1:
                loss_a = loss_m3(weight_aes, mos_a.detach())
                total_loss = loss_q + loss_a + loss_c
            elif mtl == 2:
                loss_a = loss_m3(weight_aes, mos_a.detach())
                total_loss = loss_q + loss_a + loss_c

        if torch.any(torch.isnan(total_loss)):
            print('nan in', idx)

        total_loss.backward()
        # statistics
        if not pretrain:
            global global_step
            logger.add_scalar(tag='total_loss', scalar_value=total_loss.item(), global_step=global_step)
            logger.add_scalar(tag='loss_q', scalar_value=loss_q.item(), global_step=global_step)
            if mtl != 0:
                logger.add_scalar(tag='loss_a', scalar_value=loss_a.item(), global_step=global_step)
            logger.add_scalar(tag='loss_c', scalar_value=loss_c.item(), global_step=global_step)
            global_step += 1

        convert_models_to_fp32(model)
        optimizer.step()

    out = eval(loader=test_loaders)
    srcc_q = out[0]
    srcc_a = out[3]
    srcc_c = out[6]
    srcc_avg = (srcc_q + srcc_a + srcc_c) / 3
    print("srccc_avg: {:.3f}\tsrcc_q: {:.3f}\tsrcc_a: {:.3f}\tsrcc_c: {:.3f}\tloss: {:.3f}".format(srcc_avg, srcc_q, srcc_a, srcc_c, total_loss))

    if not os.path.exists(os.path.join('checkpoints', dataset, 'MTD_IQAv6', str(session+1))):
        os.makedirs(os.path.join('checkpoints', dataset, 'MTD_IQAv6', str(session+1)))
    if srcc_avg > best_result['avg']:
        early_stop = 0
        best_epoch['avg'] = epoch
        best_result['avg'] = srcc_avg

    if srcc_q > best_result['quality']:
        ckpt_name = os.path.join('checkpoints', dataset, 'MTD_IQAv6', str(session+1), 'quality_best_ckpt.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'all_results': out
        }, ckpt_name)  # just change to your preferred folder/filename
        best_epoch['quality'] = epoch
        best_result['quality'] = srcc_q

    if srcc_a > best_result['authenticity']:
        ckpt_name = os.path.join('checkpoints', dataset, 'MTD_IQAv6', str(session+1), 'authenticity_best_ckpt.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'all_results': out
        }, ckpt_name)  # just change to your preferred folder/filename
        best_epoch['authenticity'] = epoch
        best_result['authenticity'] = srcc_a

    if srcc_c > best_result['correspondence']:
        ckpt_name = os.path.join('checkpoints', dataset, 'MTD_IQAv6', str(session+1), 'correspondence_best_ckpt.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'all_results': out
        }, ckpt_name)  # just change to your preferred folder/filename
        best_epoch['correspondence'] = epoch
        best_result['correspondence'] = srcc_c

    early_stop += 1

    return best_result, best_epoch, out

def eval(loader):
    model.eval()
    y_q = []
    y_pred_q = []
    y_a = []
    y_pred_a = []
    y_c = []
    y_pred_c = []
    for step, sample_batched in enumerate(loader):
        x, mos_q, mos_a, mos_c, con_tokens, aes_tokens = sample_batched['I'], sample_batched['mos_q'], \
                                                         sample_batched['mos_a'], sample_batched['mos_c'], \
                                                         sample_batched['con_tokens'], sample_batched['aes_tokens']
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
            y_q.extend(mos_q.cpu().numpy())
            y_a.extend(mos_a.cpu().numpy())
            y_c.extend(mos_c.cpu().numpy())

    _, PLCC1, SRCC1, KRCC1 = compute_metric(np.array(y_q), np.array(y_pred_q), istrain)
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
for dataset in datasets:
    mtl_map = {'AGIQA3K': 0, 'AIGCIQA2023': 1, 'PKUI2IQA': 2}
    mtl = mtl_map[dataset]
    change_epoch = {'AGIQA3K': 60, 'AIGCIQA2023': 60, 'PKUI2IQA': 60}

    print('train on ', dataset)

    for session in range(0, 3):
        model = model_modify.CIANet(device=device, clip_net=clip_net, in_size=in_size)
        model = model.to(device)

        runs_path = os.path.join('./log', dataset, 'MTD_IQAv6', str(session+1))
        logger = sum_writer(runs_path)
        train_loss = []
        early_stop = 0
        start_epoch = 0
        global_step = 0
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

        optimizer1 = torch.optim.AdamW(model.parameters(), lr=initial_lr1, weight_decay=weight_decay)
        optimizer2 = torch.optim.AdamW(model.parameters(), lr=initial_lr2, weight_decay=weight_decay)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=5)

        result_pkl = {}

        # pretrain
        for epoch in range(0, 10):
            freeze_model(model.base, opt=3)
            optimizer = optimizer1
            best_result, best_epoch, all_result = train(model, best_result, best_epoch)
            print(epoch, best_result)
        pre_pth = torch.load(os.path.join('checkpoints', dataset, 'MTD_IQAv6', str(session+1), 'quality_best_ckpt.pt'))
        model.load_state_dict(pre_pth['model_state_dict'], strict=True)
        pretrain = False
        freeze_model(model.base, opt=4)
        for epoch in range(0, num_epoch):
            if epoch >= change_epoch[dataset]:
                optimizer2 = torch.optim.AdamW(model.parameters(), lr=initial_lr2/10, weight_decay=weight_decay)
            optimizer = optimizer2

            print(f'begin session {session+1}, epoch {epoch}')
            best_result, best_epoch, all_result = train(model, best_result, best_epoch)

            result_pkl[str(epoch)] = all_result

            if epoch % 5 == 0:
                print('...............current average best...............')
                print('best average epoch:{}'.format(best_epoch['avg']))
                print('best average result:{}'.format(best_result['avg']))

                print('...............current quality best...............')
                print('best quality epoch:{}'.format(best_epoch['quality']))
                print('best quality result:{}'.format(best_result['quality']))

                if mtl != 0:
                    print('...............current authenticity best...............')
                    print('best scene epoch:{}'.format(best_epoch['authenticity']))
                    print('best scene result:{}'.format(best_result['authenticity']))

                print('...............current correspondence best...............')
                print('best correspondence epoch:{}'.format(best_epoch['correspondence']))
                print('best correspondence result:{}'.format(best_result['correspondence']))

            if early_stop > 20:
                print(f'early stopping at epoch {epoch}!')
                break

        pkl_name = os.path.join('checkpoints', dataset, 'MTD_IQAv6', str(session+1), 'all_results.pkl')
        with open(pkl_name, 'wb') as f:
            pickle.dump(result_pkl, f)
