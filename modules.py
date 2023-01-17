'''
 @FileName    : modules.py
 @EditTime    : 2022-09-27 14:45:21
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import os
import torch
import time
import yaml
from utils.imutils import vis_img
from utils.logger import Logger
from loss_func import *
import torch.optim as optim
from utils.cyclic_scheduler import CyclicLRWithRestarts
from datasets.dataset import MyData
from utils.smpl_torch_batch import SMPLModel
from utils.renderer_pyrd import Renderer
import cv2
import pickle

def init(note='occlusion', dtype=torch.float32, **kwargs):
    # Create the folder for the current experiment
    mon, day, hour, min, sec = time.localtime(time.time())[1:6]
    out_dir = os.path.join('output', note)
    out_dir = os.path.join(out_dir, 'gigacrowd') #os.path.join(out_dir, '%02d.%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create the log for the current experiment
    logger = Logger(os.path.join(out_dir, 'log.txt'), title="template")
    logger.set_names([note])
    logger.set_names(['%02d/%02d-%02dh%02dm%02ds' %(mon, day, hour, min, sec)])
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Test Loss'])

    # Store the arguments for the current experiment
    conf_fn = os.path.join(out_dir, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwargs, conf_file)

    # load smpl model 
    model_smpl = SMPLModel(
                        device=torch.device('cpu'),
                        model_path='./data/SMPL_NEUTRAL.pkl', 
                        data_type=dtype,
                    )

    return out_dir, logger, model_smpl


class LossLoader():
    def __init__(self, train_loss='L1', test_loss='L1', device=torch.device('cpu'), **kwargs):
        self.train_loss_type = train_loss.split(' ')
        self.test_loss_type = test_loss.split(' ')
        self.device = device

        # Parse the loss functions
        self.train_loss = {}
        for loss in self.train_loss_type:
            if loss == 'L1':
                self.train_loss.update(L1=L1(self.device))
            if loss == 'L2':
                self.train_loss.update(L2=L2(self.device))
            if loss == 'SMPL_Loss':
                self.train_loss.update(SMPL_Loss=SMPL_Loss(self.device))
            if loss == 'Keyp_Loss':
                self.train_loss.update(Keyp_Loss=Keyp_Loss(self.device))
            if loss == 'Mesh_Loss':
                self.train_loss.update(Mesh_Loss=Mesh_Loss(self.device))
            if loss == 'Joint_Loss':
                self.train_loss.update(Joint_Loss=Joint_Loss(self.device))
            if loss == 'Plane_Loss':
                self.train_loss.update(Plane_Loss=Plane_Loss(self.device))
            if loss == 'Height_Loss':
                self.train_loss.update(Height_Loss=Height_Loss(self.device))
            # You can define your loss function in loss_func.py, e.g., Smooth6D, 
            # and load the loss by adding the following lines

            # if loss == 'Smooth6D':
            #     self.train_loss.update(Smooth6D=Smooth6D(self.device))

        self.test_loss = {}
        for loss in self.test_loss_type:
            if loss == 'L1':
                self.test_loss.update(L1=L1(self.device))
            if loss == 'MPJPE':
                self.test_loss.update(MPJPE=MPJPE(self.device))

    def calcul_trainloss(self, pred, gt):
        loss_dict = {}
        for ltype in self.train_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.train_loss['L1'](pred, gt))
            elif ltype == 'L2':
                loss_dict.update(L2=self.train_loss['L2'](pred, gt))
            elif ltype == 'SMPL_Loss':
                SMPL_loss = self.train_loss['SMPL_Loss'](pred['pred_rotmat'], gt['pose'], pred['pred_shape'], gt['betas'])
                loss_dict = {**loss_dict, **SMPL_loss}
            elif ltype == 'Keyp_Loss':
                Keyp_loss = self.train_loss['Keyp_Loss'](pred['pred_keypoints_2d'], gt['keypoints'])
                loss_dict = {**loss_dict, **Keyp_loss}
            elif ltype == 'Mesh_Loss':
                Mesh_loss = self.train_loss['Mesh_Loss'](pred['pred_verts'], gt['verts'], pred['pred_joints'], gt['joints'])
                loss_dict = {**loss_dict, **Mesh_loss}
            elif ltype == 'Joint_Loss':
                Joint_Loss = self.train_loss['Joint_Loss'](pred['pred_joints'], gt['gt_joints'])
                loss_dict = {**loss_dict, **Joint_Loss}
            elif ltype == 'Plane_Loss':
                Plane_Loss = self.train_loss['Plane_Loss'](pred['pred_joints'], gt['plane_point'], gt['plane_norm'])
                loss_dict = {**loss_dict, **Plane_Loss}
            elif ltype == 'Height_Loss':
                Height_Loss = self.train_loss['Height_Loss'](pred['pred_joints'])
                loss_dict = {**loss_dict, **Height_Loss}
            # Calculate your loss here

            # elif ltype == 'Smooth6D':
            #     loss_dict.update(Smooth6D=self.train_loss['Smooth6D'](pred_pose))
            else:
                pass
        loss = 0
        for k in loss_dict:
            loss_temp = loss_dict[k] * 60.
            loss += loss_temp
            loss_dict[k] = round(float(loss_temp.detach().cpu().numpy()), 6)
        return loss, loss_dict

    def calcul_refineloss(self, pred, gt):
        loss_dict = {}
        for ltype in self.train_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.train_loss['L1'](pred, gt))
            elif ltype == 'L2':
                loss_dict.update(L2=self.train_loss['L2'](pred, gt))
            elif ltype == 'SMPL_Loss':
                SMPL_loss = self.train_loss['SMPL_Loss'](pred['pred_rotmat'], gt['pose'], pred['pred_shape'], gt['betas'])
                loss_dict = {**loss_dict, **SMPL_loss}
            elif ltype == 'Keyp_Loss':
                Keyp_loss = self.train_loss['Keyp_Loss'](pred['pred_keypoints_2d'], gt['keypoints'])
                loss_dict = {**loss_dict, **Keyp_loss}
            elif ltype == 'Mesh_Loss':
                Mesh_loss = self.train_loss['Mesh_Loss'](pred['pred_verts'], gt['verts'], pred['pred_joints'], gt['joints'])
                loss_dict = {**loss_dict, **Mesh_loss}
            elif ltype == 'Joint_Loss':
                Joint_Loss = self.train_loss['Joint_Loss'](pred['pred_joints'], gt['gt_joints'])
                loss_dict = {**loss_dict, **Joint_Loss}
            # Calculate your loss here

            # elif ltype == 'Smooth6D':
            #     loss_dict.update(Smooth6D=self.train_loss['Smooth6D'](pred_pose))
            else:
                pass
        loss = 0
        for k in loss_dict:
            loss_temp = loss_dict[k] * 60.
            loss += loss_temp
            loss_dict[k] = round(float(loss_temp.detach().cpu().numpy()), 6)
        return loss, loss_dict

    def calcul_testloss(self, pred, gt):
        loss_dict = {}
        for ltype in self.test_loss:
            if ltype == 'L1':
                loss_dict.update(L1=self.test_loss['L1'](pred, gt))
            elif ltype == 'MPJPE':
                loss_dict.update(MPJPE=self.test_loss['MPJPE'](pred['pred_joints'], gt['joints']))
            else:
                print('The specified loss: %s does not exist' %ltype)
                pass
        loss = 0
        for k in loss_dict:
            loss += loss_dict[k]
            loss_dict[k] = round(float(loss_dict[k].detach().cpu().numpy()), 6)
        return loss, loss_dict


class ModelLoader():
    def __init__(self, dtype=torch.float32, output='', device=torch.device('cpu'), model=None, lr=0.001, pretrain=False, pretrain_dir='', batchsize=32, task=None, save_mesh=False, save_img=False, **kwargs):

        self.output = output
        self.device = device
        self.batchsize = batchsize
        self.save_mesh = save_mesh
        self.save_img = save_img

        # load smpl model 
        self.model_smpl_gpu = SMPLModel(
                            device=torch.device('cuda'),
                            model_path='./data/SMPL_NEUTRAL.pkl', 
                            data_type=dtype,
                        )
        # # Setup renderer for visualization
        # self.renderer = Renderer(focal_length=5000., img_res=224, faces=self.model_smpl_gpu.faces)

        # Load model according to model name
        self.model_type = model
        exec('from model.' + self.model_type + ' import ' + self.model_type)
        self.model = eval(self.model_type)(self.model_smpl_gpu)
        print('load model: %s' %self.model_type)

        # Calculate model size
        model_params = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad == True:
                model_params += parameter.numel()
        print('INFO: Model parameter count:', model_params)

        if torch.cuda.is_available():
            self.model.to(self.device)
            print("device: cuda")
        else:
            print("device: cpu")

        self.optimizer = optim.AdamW(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = None

        # Load pretrain parameters
        if pretrain:
            model_dict = self.model.state_dict()
            params = torch.load(pretrain_dir)
            premodel_dict = params['model']
            premodel_dict = {k.replace('module.', ''): v for k ,v in premodel_dict.items() if k.replace('module.', '') in model_dict}
            self.init_weights = premodel_dict
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)
            print("Load pretrain parameters from %s" %pretrain_dir)
            #self.optimizer.load_state_dict(params['optimizer'])
            print("Load optimizer parameters")
            
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr

        if task == 'relation':
            model_dict = self.model.state_dict()
            params = torch.load('pretrain_model/SPIN.pkl')
            premodel_dict = params['model']
            premodel_dict = {'hmr.' + k: v for k ,v in premodel_dict.items() if 'hmr.' + k in model_dict}
            model_dict.update(premodel_dict)
            self.model.load_state_dict(model_dict)

    def reload_optimizer(self, lr):
        model_dict = self.model.state_dict()
        model_dict.update(self.init_weights)
        self.model.load_state_dict(model_dict)
        self.optimizer = optim.AdamW(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = CyclicLRWithRestarts(optimizer=self.optimizer, batch_size=self.batchsize, epoch_size=1, restart_period=20, t_mult=2, policy="cosine", verbose=True)

    def reload_scheduler(self):
        self.scheduler = CyclicLRWithRestarts(optimizer=self.optimizer, batch_size=self.batchsize, epoch_size=1, restart_period=140, t_mult=2, policy="cosine", verbose=True)

    def save_pkl(self, path, result):
        """"
        save pkl file
        """
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        with open(path, 'wb') as result_file:
            pickle.dump(result, result_file, protocol=2)


    def load_scheduler(self, epoch_size):
        self.scheduler = CyclicLRWithRestarts(optimizer=self.optimizer, batch_size=self.batchsize, epoch_size=epoch_size, restart_period=10, t_mult=2, policy="cosine", verbose=True)

    def save_model(self, epoch, task):
        # save trained model
        output = os.path.join(self.output, 'trained model')
        if not os.path.exists(output):
            os.makedirs(output)

        model_name = os.path.join(output, '%s_epoch%03d.pkl' %(task, epoch))
        torch.save({'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}, model_name)
        print('save model to %s' % model_name)

    def save_results(self, results, iter, batchsize):
        output = os.path.join(self.output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        if self.save_img:
            img_origin = results['imgs'][0]
            img_origin = cv2.imread(img_origin)
            img = img_origin.copy()
            renderer = Renderer(focal_length=results['focal_length'][0], img_w=img.shape[1], img_h=img.shape[0],
                                faces=self.model_smpl_gpu.faces,
                                same_mesh_color=True)
            front_view = renderer.render_front_view(results['pred_verts'],
                                                    bg_img_rgb=img.copy())

        for index, (pred_verts, focal, imname, pose, shape, trans, keyps) in enumerate(zip(results['pred_verts'], results['focal_length'], results['instance'], results['pred_pose'], results['pred_shape'], results['pred_trans'], results['keypoints'])):

            # for kp in keyps:
            #     if kp[2] > 0.3:
            #         front_view = cv2.circle(front_view, tuple(kp[:2].astype(np.int)), 5, (0,0,255), -1)

            param_name = os.path.join(output, 'params/%s.pkl' %(imname))
            if not os.path.exists(param_name):
                self.save_pkl(param_name, {'pose':pose, 'betas':shape, 'trans':trans})

            if self.save_mesh:
                mesh_name = os.path.join(output, 'meshes/%s.obj' %(imname))
                self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

        if self.save_img:
            # vis_img('img', front_view)
            front_view = np.concatenate((front_view, img), axis=1)
            render_name = "%s.jpg" % (imname.replace('/', '_').replace('\\', '_'))
            cv2.imwrite(os.path.join(output, render_name), front_view)

            renderer.delete()
            # vis_img('pred_smpl', pred_smpl)
            # vis_img('gt_smpl', gt_smpl)


class DatasetLoader():
    def __init__(self, trainset=None, testset=None, data_folder='./data', dtype=torch.float32, smpl=None, task=None, **kwargs):
        self.data_folder = data_folder
        self.trainset = trainset.split(' ')
        self.testset = testset.split(' ')
        self.dtype = dtype
        self.smpl = smpl
        self.task = task

    def load_trainset(self):
        train_dataset = []
        for i in range(len(self.trainset)):
            train_dataset.append(MyData(True, self.dtype, self.data_folder, self.trainset[i], self.smpl))
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        return train_dataset

    def load_testset(self):
        test_dataset = []
        for i in range(len(self.testset)):
            test_dataset.append(MyData(False, self.dtype, self.data_folder, self.testset[i], self.smpl))
        test_dataset = torch.utils.data.ConcatDataset(test_dataset)
        return test_dataset


