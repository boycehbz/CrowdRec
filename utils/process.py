'''
 @FileName    : process.py
 @EditTime    : 2022-09-27 16:18:51
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import torch
import numpy as np
import cv2
from tqdm import tqdm
import time
import os
from utils.imutils import cam_crop2full


def to_device(data, device):
    assert data['norm_img'].shape[0] == 1

    imnames = {'instance':[d[0] for d in data['instance']]}
    origin_imgs = {'origin_img':data['origin_img']}
    data = {k:v[0].to(device).float() for k, v in data.items() if k not in ['instance', 'origin_img']}
    data = {**imnames, **data, **origin_imgs}

    return data

def pseudo_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    # model.model.train(mode=True)
    model.model.eval()
    origin_name = None
    train_loss = 0.
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # if i < 244:
        #     continue
        batchsize = data["norm_img"].shape[0]
        data = to_device(data, device)

        cur_name = data['origin_img'][0]
        if origin_name != cur_name:
            origin_name = cur_name
            model.reload_optimizer(0.00001)
            total_count = 260
        else:
            model.reload_scheduler()
            total_count = 100

        # model.reload_optimizer(0.00001)
        # total_count = 260

        model.model.init_trans = None

        losses = []
        
        count = 0
        not_fitted = True

        while count < total_count or not_fitted:
            if model.scheduler is not None:
                model.scheduler.step()

            pred = model.model(data)

            if total_count == 0:
                break

            if 'pose' not in data.keys():
                data['pose'] = pred['pred_pose'].detach()
                data['betas'] = pred['pred_shape'].detach()
                data['verts'] = pred['pred_verts'].detach()
                data['joints'] = pred['pred_joints'].detach()

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)

            # backward
            model.optimizer.zero_grad()
            loss.backward()

            # optimize
            model.optimizer.step()
            if model.scheduler is not None:
                model.scheduler.batch_step()
            
            loss_batch = loss.detach() #/ batchsize
            lr = model.optimizer.state_dict()['param_groups'][0]['lr']
            print('lr: %.8f count: %d/%d epoch: %d/%d, batch: %d/%d, loss: %.6f' %(lr, count, total_count, epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
            train_loss += loss_batch

            losses.append(loss_batch.cpu().numpy())

            if count > total_count:
                if loss_batch.cpu().numpy() < np.sort(np.array(losses))[:5].max():
                    not_fitted = False

            count += 1

        results = {}
        results.update(instance=data['instance'])
        results.update(imgs=data['origin_img'])
        results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
        results.update(focal_length=data["focal_length"].detach().cpu().numpy().astype(np.float32))
        results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
        results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
        results.update(pred_trans=pred["pred_cam_t"].detach().cpu().numpy().astype(np.float32))
        results.update(keypoints=data["origin_keypoints"].detach().cpu().numpy().astype(np.float32))

        # results.update(pred_cam_t=pred_cam_full.detach().cpu().numpy().astype(np.float32))
        model.save_results(results, i, batchsize)

    return train_loss/len_data
