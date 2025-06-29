import os

datasets = ['AIGCIQA2023']
txt_dirs = {'AGIQA3K': '/public/tansongbai/code/AGIQA/IQA_Database/AGIQA-3K',
            'AIGCIQA2023': '/public/tansongbai/code/AGIQA/IQA_Database/AIGCIQA2023',
            'PKUI2IQA': '/public/tansongbai/code/AGIQA/IQA_Database/PKU-I2IQA'}

for dataset in datasets:
    for split in range(1, 11):
        txt_path = os.path.join(txt_dirs[dataset], str(split), 'test.txt')
        sv_path = os.path.join(txt_dirs[dataset], str(split), 'idv_test')
        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            f.close()
        names = [x.split('\t')[0] for x in lines]
        if dataset == 'AGIQA3K' or dataset == 'PKUI2IQA':
            ai_models = set([x.split('/')[-1].split('_')[0] for x in names])
            sv_dirs = {}
            for i in ai_models:
                sv_dirs[i] = []
            for idx, i in enumerate([x.split('/')[-1].split('_')[0] for x in names]):
                sv_dirs[i].append(lines[idx])
            for key in sv_dirs:
                data = ''
                datas = sv_dirs[key]
                for i in datas:
                    data += i

                sv_txt_path = os.path.join(sv_path, key+'.txt')
                with open(sv_txt_path, 'w') as f:
                    f.write(data)
                    f.close()

        elif dataset == 'AIGCIQA2023':
            margin = {'Controlnet': range(0, 400),
                      'DALLE': range(400, 800),
                      'Glide': range(800, 1200),
                      'Lafite': range(1200, 1600),
                      'stable-diffusion': range(1600, 2000),
                      'Unidiffuser': range(2000, 2400)}
            for ai_model in margin:
                data = ''
                for line in lines:
                    sec = int(os.path.split(line.split('\t')[0])[-1][:-4])
                    if sec in margin[ai_model]:
                        data += line

                sv_txt_path = os.path.join(sv_path, ai_model + '.txt')
                with open(sv_txt_path, 'w') as f:
                    f.write(data)
                    f.close()



