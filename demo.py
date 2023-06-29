'''
 @FileName    : demo.py
 @EditTime    : 2023-06-27 16:20:37
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import torch
from utils.cmd_parser import parse_config
from utils.modules import init, LossLoader, ModelLoader, CrowdRec
from yolox.yolox import Predictor
from alphapose.alphapose_core import AlphaPose_Predictor

###########Load config file in debug mode#########
import sys
sys.argv = ['','--config=cfg_files/demo.yaml']

def main(**args):
    # Global setting
    dtype = torch.float32
    device = torch.device(index=args.get('gpu_index'), type='cuda')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # human detection
    yolox_model_dir = R'pretrain_model/bytetrack_x_mot17.pth.tar'
    yolox_thres = 0.3
    yolox_predictor = Predictor(yolox_model_dir, yolox_thres)

    # 2D pose estimation
    alpha_config = R'alphapose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
    alpha_checkpoint = R'pretrain_model/halpe26_fast_res50_256x192.pth'
    alpha_thres = 0.1
    alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)

    # Load loss function
    loss = LossLoader(device=device, **args)

    # Load model
    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)

    crowdrec = CrowdRec(loss, model, yolox_predictor, alpha_predictor, device, **args)

    crowdrec.predict()


if __name__ == "__main__":
    args = parse_config()
    main(**args)





