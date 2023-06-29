'''
 @FileName    : main.py
 @EditTime    : 2022-01-18 14:10:30
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import torch
import os
from torch.utils.data import DataLoader
from utils.cmd_parser import parse_config
from utils.module_utils import seed_worker, set_seed
from utils.modules import init, LossLoader, ModelLoader, DatasetLoader
from utils.logger import savefig

###########Load config file in debug mode#########
import sys
sys.argv = ['','--config=cfg_files/config.yaml'] #MoCap End2End HMAE_conv Lifting

def main(**args):
    seed = 7
    g = set_seed(seed)

    # Global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'), type='cuda')
    mode = args.get('mode')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # Load loss function
    loss = LossLoader(device=device, **args)

    # Load model
    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)

    # create data loader
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    if mode == 'train':
        train_dataset = dataset.load_trainset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batchsize, shuffle=False,
            num_workers=workers, pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if args.get('use_sch'):
            model.load_scheduler(1)

    # Load handle function with the task name
    task = args.get('task')
    exec('from utils.process import %s_train' %task)

    for epoch in range(num_epoch):
        # training mode
        if mode == 'train':
            training_loss = eval('%s_train' %task)(model, loss, train_loader, epoch, num_epoch, device=device)

            # save trained model
            if (epoch) % 1 == 0:
                model.save_model(epoch, task)

        lr = model.optimizer.state_dict()['param_groups'][0]['lr']
        logger.append([int(epoch + 1), lr, training_loss, 0])

    logger.close()
    logger.plot(['Epoch', 'Test Loss'])
    savefig(os.path.join(out_dir, 'log.jpg'))

if __name__ == "__main__":
    args = parse_config()
    main(**args)





