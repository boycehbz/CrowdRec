'''
 @FileName    : loss_func.py
 @EditTime    : 2022-01-13 19:16:39
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : define loss functions here
'''

import torch.nn as nn
import torch
import numpy as np
from utils.geometry import batch_rodrigues

class L1(nn.Module):
    def __init__(self, device):
        super(L1, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss(size_average=False)

    def forward(self, x, y):
        b = x.shape[0]
        diff = self.L1Loss(x, y)
        diff = diff / b
        return diff

class SMPL_Loss(nn.Module):
    def __init__(self, device):
        super(SMPL_Loss, self).__init__()
        self.device = device
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.beta_loss_weight = 0.001
        self.pose_loss_weight = 5.0

    def forward(self, pred_rotmat, gt_pose, pred_betas, gt_betas):
        loss_dict = {}
        pred_rotmat_valid = pred_rotmat
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)

        pred_betas_valid = pred_betas
        gt_betas_valid = torch.zeros_like(gt_betas, dtype=gt_betas.dtype, device=gt_betas.device)
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)

        loss_dict['pose_Loss'] = loss_regr_pose * self.pose_loss_weight
        loss_dict['shape_Loss'] = loss_regr_betas * self.beta_loss_weight
        return loss_dict


class Keyp_Loss(nn.Module):
    def __init__(self, device):
        super(Keyp_Loss, self).__init__()
        self.device = device
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.keyp_weight = 5.0

    def forward(self, pred_keypoints_2d, gt_keypoints_2d):
        loss_dict = {}
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()

        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()

        loss_dict['keyp_Loss'] = loss * self.keyp_weight
        return loss_dict

class Height_Loss(nn.Module):
    def __init__(self, device):
        super(Height_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.height_weight = 0.001

    def forward(self, pred_joints):
        loss_dict = {}
        b = pred_joints.shape[0]

        if b <= 1:
            dis_std = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
        else:
            bottom = (pred_joints[:,15] + pred_joints[:,16]) / 2
            top = pred_joints[:,17]

            l = (top - bottom) / torch.norm(top - bottom, dim=1)[:,None]
            norm = torch.mean(l, dim=0)

            root = pred_joints[:,19]

            proj = torch.matmul(root, norm)

            dis_std = proj.std()

        loss_dict['plane_loss'] = dis_std * self.height_weight
        
        return loss_dict

class Plane_Loss(nn.Module):
    def __init__(self, device):
        super(Plane_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.joint_weight = 5.0
        self.verts_weight = 5.0

    def forward(self, pred_joints, point, norm):
        loss_dict = {}
        b = pred_joints.shape[0]
        assert b == 1

        ankle = pred_joints[0,[15,16]]
        point = point[0,None,:].repeat(ankle.shape[0], 1)
        norm = norm[0]
        ankle = ankle - point
        vet = torch.matmul(ankle, norm.T)
        dis = torch.abs(vet) / torch.norm(norm)
        dis = dis.sum() / b

        loss_dict['plane_loss'] = dis
        
        return loss_dict

def point2ground(ankles, point, norm):
    '''ankles: N*3'''
    # point = torch.tensor(point, dtype=ankles.dtype, device=ankles.device)
    # norm = torch.tensor(norm, dtype=ankles.dtype, device=ankles.device)
    point_ = point.repeat(ankles.shape[0], 1)
    tmp_vet = ankles - point_
    vet = torch.matmul(tmp_vet, norm.T)
    dis = torch.abs(vet) / torch.norm(norm)
    return dis

class Joint_Loss(nn.Module):
    def __init__(self, device):
        super(Joint_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.joint_weight = 5.0
        self.verts_weight = 5.0

    def forward(self, pred_joints, gt_joints):
        loss_dict = {}
        
        conf = gt_joints[:, :, -1].unsqueeze(-1).clone()

        gt_pelvis = gt_joints[:,19,:3]
        gt_joints[:,:,:-1] = gt_joints[:,:,:-1] - gt_pelvis[:,None,:]

        pred_pelvis = pred_joints[:,19,:]
        pred_joints = pred_joints - pred_pelvis[:,None,:]

        gt_joints = gt_joints
        pred_joints = pred_joints

        if len(gt_joints) > 0:
            joint_loss = (conf * self.criterion_joint(pred_joints, gt_joints[:, :, :-1])).mean()
        else:
            joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['joint_loss'] = joint_loss * self.joint_weight
        return loss_dict

class Mesh_Loss(nn.Module):
    def __init__(self, device):
        super(Mesh_Loss, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_joint = nn.MSELoss().to(self.device)
        self.joint_weight = 5.0
        self.verts_weight = 5.0

    def forward(self, pred_vertices, gt_vertices, pred_joints, gt_joints):
        loss_dict = {}
        pred_vertices_with_shape = pred_vertices
        gt_vertices_with_shape = gt_vertices
        
        vert_loss = self.criterion_vert(pred_vertices_with_shape, gt_vertices_with_shape)
        joint_loss = self.criterion_joint(pred_joints, gt_joints)


        loss_dict['vert_loss'] = vert_loss * self.verts_weight
        loss_dict['joint_loss'] = joint_loss * self.joint_weight
        return loss_dict


class L2(nn.Module):
    def __init__(self, device):
        super(L2, self).__init__()
        self.device = device
        self.L2Loss = nn.MSELoss(size_average=False)

    def forward(self, x, y):
        b = x.shape[0]
        diff = self.L2Loss(x, y)
        diff = diff / b
        return diff

class Smooth6D(nn.Module):
    def __init__(self, device):
        super(Smooth6D, self).__init__()
        self.device = device
        self.L1Loss = nn.L1Loss(size_average=False)

    def forward(self, x, y):
        b, f = x.shape[:2]
        diff = self.L1Loss(x, y)
        diff = diff / b / f
        return diff

class MPJPE(nn.Module):
    def __init__(self, device):
        super(MPJPE, self).__init__()
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward_instance(self, pred_pose, gt_pose, pred_shape, gt_shape):
        loss_dict = {}
        b, _, f, j = pred_pose.shape
        dtype = pred_pose.dtype
        pred_pose = pred_pose.permute((0,2,3,1)).contiguous().view(-1,72)
        gt_pose = gt_pose.permute((0,2,3,1)).contiguous().view(-1,72)

        pred_shape = pred_shape.expand(b, f, 10).contiguous().view(-1,10)

        gt_shape = gt_shape[:,None,:]
        gt_shape = gt_shape.expand(b, f, 10).contiguous().view(-1,10)

        # shape = torch.zeros((pred_pose.size(0), 10), device=self.device, dtype=dtype)
        trans = torch.zeros((pred_pose.size(0), 3), device=self.device, dtype=dtype)

        pred_mesh, _ = self.model(pred_shape, pred_pose, trans)
        with torch.no_grad():
            gt_mesh, _ = self.model(gt_shape, gt_pose, trans)

        pred_joints = torch.matmul(self.regressor, pred_mesh)
        gt_joints = torch.matmul(self.regressor, gt_mesh)

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]))
        diff = torch.mean(diff, dim=[1])
        diff = diff * 1000
        
        return diff.detach().cpu().numpy().reshape(b,f)

    def forward(self, pred_joints, gt_joints):
        loss_dict = {}

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]))
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        
        return diff

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]

        return joints - pelvis[:,None,:].repeat(1, 14, 1)
