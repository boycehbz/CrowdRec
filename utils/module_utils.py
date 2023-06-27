'''
 @FileName    : module_utils.py
 @EditTime    : 2022-09-27 14:38:28
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import numpy as np
import random
import torch
import pickle
import os
import json

def save_json(out_path, data):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    with open(out_path, 'w') as f:
        json.dump(data, f)

def save_camparam(path, intris, extris):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, 'w')
    for ind, (intri, extri) in enumerate(zip(intris, extris)):
        f.write(str(ind)+'\n')
        for i in intri:
            f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
        f.write('0 0 \n')
        for i in extri[:3]:
            f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')
        f.write('\n')
    f.close()

def load_pkl(path):
    """"
    load pkl file
    """
    with open(path, 'rb') as f:
        param = pickle.load(f, encoding='iso-8859-1')
    return param

def seed_worker(worker_seed=7):
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    # Set a constant random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g
