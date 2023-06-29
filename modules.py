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
from utils.module_utils import save_camparam
import constants
from process import to_device
import math

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

        intri = np.eye(3)
        intri[0][0] = results['focal_length'][0]
        intri[1][1] = results['focal_length'][0]
        intri[0][2] = img.shape[1] / 2
        intri[1][2] = img.shape[0] / 2
        cam_name = "%s.txt" % (imname.replace('/', '_').replace('\\', '_'))
        save_camparam(os.path.join(output, 'camparams', cam_name), [intri], [np.eye(4)])


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


class CrowdRec(object):
    def __init__(
        self,
        loss,
        backbone_model,
        detector,
        pose_estimator,
        device=torch.device('cuda'),
        data_folder='',
        **kwargs
    ):
        self.loss_func = loss
        self.model = backbone_model
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.data_folder = data_folder
        self.intris = None
        self.device = device

        self.images = os.listdir(self.data_folder)

        self.len = len(self.images)

    def bbox_from_detector(self, bbox, rescale=1.1):
        """
        Get center and scale of bounding box from bounding box.
        The expected format is [min_x, min_y, max_x, max_y].
        """
        # center
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        center = torch.tensor([center_x, center_y])

        # scale
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_size = max(bbox_w * 256 / float(192), bbox_h)
        scale = bbox_size / 200.0
        # adjust bounding box tightness
        scale *= rescale
        return center, scale

    def get_transform(self, center, scale, res, rot=0):
        """Generate transformation matrix."""
        # res: (height, width), (rows, cols)
        crop_aspect_ratio = res[0] / float(res[1])
        h = 200 * scale
        w = h / crop_aspect_ratio
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / w
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / w + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t


    def transform(self, pt, center, scale, res, invert=0, rot=0):
        """Transform pixel location to different reference."""
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
        new_pt = np.dot(t, new_pt)
        return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1



    def crop(self, img, center, scale, res):
        """
        Crop image according to the supplied bounding box.
        res: [rows, cols]
        """
        # Upper left point
        ul = np.array(self.transform([1, 1], center, scale, res, invert=1)) - 1
        # Bottom right point
        br = np.array(self.transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape, dtype=np.float32)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        try:
            new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
        except Exception as e:
            print(e)

        new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

        return new_img, ul, br


    def process_image(self, orig_img_rgb, bbox,
                    crop_height=256,
                    crop_width=192):
        """
        Read image, do preprocessing and possibly crop it according to the bounding box.
        If there are bounding box annotations, use them to crop the image.
        If no bounding box is specified but openpose detections are available, use them to get the bounding box.
        """
        try:
            center, scale = self.bbox_from_detector(bbox)
        except Exception as e:
            print("Error occurs in person detection", e)
            # Assume that the person is centered in the image
            height = orig_img_rgb.shape[0]
            width = orig_img_rgb.shape[1]
            center = np.array([width // 2, height // 2])
            scale = max(height, width * crop_height / float(crop_width)) / 200.

        img, ul, br = self.crop(orig_img_rgb, center, scale, (crop_height, crop_width))
        crop_img = img.copy()

        img = img / 255.
        mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
        std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
        norm_img = (img - mean) / std
        norm_img = np.transpose(norm_img, (2, 0, 1))

        return norm_img, center, scale, ul, br, crop_img

    def create_data(self, img, imgname, bboxes, pose2ds, instances, pose3ds=None):
        load_data = {}

        # Load image
        img = img[:,:,::-1].copy().astype(np.float32)

        img_h, img_w, _ = img.shape
        load_data["origin_img"] = imgname

        num_people = len(bboxes)

        gt_jointss = torch.zeros((num_people, 26, 4)).float()
        imgnames = ['empty'] * num_people
        norm_imgs = torch.zeros((num_people, 3, 256, 192)).float()
        centers = torch.zeros((num_people, 2)).float()
        scales = torch.zeros((num_people)).float()
        crop_uls = torch.zeros((num_people, 2)).float()
        crop_brs = torch.zeros((num_people, 2)).float()
        img_hs = torch.zeros((num_people)).float()
        img_ws = torch.zeros((num_people)).float()
        focal_lengthes = torch.zeros((num_people)).float()
        pose2d_gt = torch.zeros((num_people, 26, 3)).float()
        pose2d_origin = torch.zeros((num_people, 26, 3)).float()

        for i in range(num_people):

            if self.intris is not None:
                focal_length = self.intris[i][0][0]
            else:
                focal_length = (img_h ** 2 + img_w ** 2) ** 0.5

            bbox = bboxes[i]

            norm_img, center, scale, crop_ul, crop_br, _ = self.process_image(img.copy(), bbox)

            # Get 2D keypoints and apply augmentation transforms
            h = 200 * scale
            s = float(256) / h
            keypoints = torch.from_numpy(pose2ds[i].copy()).float() 
            keypoints[:,:2] = keypoints[:,:2] - center
            keypoints[:,:2] = (keypoints[:,:2] * s) / 256
            origin_keypoints = torch.from_numpy(pose2ds[i].copy()).float() 

            if pose3ds is None:
                gt_joints = torch.zeros((26, 4)).float()
            else:
                gt_joints = pose3ds[i].copy()
                mask = np.sum(gt_joints, axis=1)
                mask = 1 - (mask == 0).astype(np.float)
                mask = mask.reshape(-1, 1)
                gt_joints = np.concatenate((gt_joints, mask), axis=1)
                gt_joints = torch.from_numpy(gt_joints).float() 

            gt_jointss[i] = gt_joints
            imgnames[i] = os.path.join(imgname[:-4], '%04d') %instances[i]
            norm_imgs[i] = torch.from_numpy(norm_img)
            centers[i] = center
            scales[i] = scale
            crop_uls[i] = torch.from_numpy(crop_ul)
            crop_brs[i] = torch.from_numpy(crop_br)
            img_hs[i] = img_h
            img_ws[i] = img_w
            focal_lengthes[i] = focal_length
            pose2d_gt[i] = keypoints
            pose2d_origin[i] = origin_keypoints

        load_data["gt_joints"] = gt_jointss[None,:]
        load_data['instance'] = [[n] for n in imgnames]
        load_data["norm_img"] = norm_imgs[None,:]
        load_data["center"] = centers[None,:]
        load_data["scale"] = scales[None,:]
        load_data["crop_ul"] = crop_uls[None,:]
        load_data["crop_br"] = crop_brs[None,:]
        load_data["img_h"] = img_hs[None,:]
        load_data["img_w"] = img_ws[None,:]
        load_data["focal_length"] = focal_lengthes[None,:]
        load_data["keypoints"] = pose2d_gt[None,:]
        load_data["origin_keypoints"] = pose2d_origin[None,:]
        load_data["origin_img"] = [imgname]

        return load_data

    def optimization(self, data, idx):

        data = to_device(data, self.device)

        cur_name = data['origin_img'][0]
        if self.origin_name != cur_name:
            self.origin_name = cur_name
            self.model.reload_optimizer(0.00001)
            total_count = 100
        else:
            self.model.reload_scheduler()
            total_count = 50

        self.model.model.init_trans = None

        losses = []
        
        count = 0
        not_fitted = True

        while count < total_count or not_fitted:
            if self.model.scheduler is not None:
                self.model.scheduler.step()

            pred = self.model.model(data)

            if total_count == 0:
                break

            if 'pose' not in data.keys():
                data['pose'] = pred['pred_pose'].detach()
                data['betas'] = pred['pred_shape'].detach()
                data['verts'] = pred['pred_verts'].detach()
                data['joints'] = pred['pred_joints'].detach()

            # calculate loss
            loss, cur_loss_dict = self.loss_func.calcul_trainloss(pred, data)

            # backward
            self.model.optimizer.zero_grad()
            loss.backward()

            # optimize
            self.model.optimizer.step()
            if self.model.scheduler is not None:
                self.model.scheduler.batch_step()
            
            loss_batch = loss.detach() #/ batchsize
            lr = self.model.optimizer.state_dict()['param_groups'][0]['lr']
            print('lr: %.8f count: %d/%d batch: %d/%d, loss: %.6f' %(lr, count, total_count, idx, self.len, loss_batch), cur_loss_dict)

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
        self.model.save_results(results, idx, 1)

    def predict(self,):
        batch_size = 50
        self.origin_name = None
        self.model.model.eval()
        for idx in range(self.len):
            img_path = os.path.join(self.data_folder, self.images[idx])
            img = cv2.imread(img_path)

            results, result_img = self.detector.predict(img, viz=False)

            pose = self.pose_estimator.predict(img, results['bbox'])

            instances = np.arange(len(pose))

            batch = math.ceil(len(instances) / batch_size)

            for b in range(batch):
                start = b * batch_size
                end = b * batch_size + batch_size
                if len(instances) - start < 8 and start != 0:
                    start = start - 10

                batch_bbox = results['bbox'][start:end]
                batch_pose = pose[start:end]
                batch_instances = instances[start:end]

                load_data = self.create_data(img, img_path, batch_bbox, batch_pose, batch_instances)

                self.optimization(load_data, idx)