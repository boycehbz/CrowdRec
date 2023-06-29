'''
 @FileName    : base.py
 @EditTime    : 2022-10-04 15:54:18
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import torch.utils.data as data
import torch
import os
import numpy as np
from torchvision.transforms import Normalize
import utils.constants as constants
import pickle
import cv2
from utils.geometry import estimate_translation_np
from utils.imutils import crop, flip_img, flip_pose, flip_kp, surface_projection, transform, rot_aa

class base(data.Dataset):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None):
        self.is_train = train
        self.dataset_dir = os.path.join(data_folder, name)
        self.dtype = dtype
        self.np_type = np.float32
        self.smpl = smpl

        # Augmentation parameters
        self.noise_factor = 0.4
        self.rot_factor = 30
        self.scale_factor = 0.25
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        self.PoseInd = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21]
        self.flip_index = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,17,18,20,19]


    def load_pkl(self, path):
        """"
        load pkl file
        """
        with open(path, 'rb') as f:
            param = pickle.load(f, encoding='iso-8859-1')
        return param


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc


    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def calc_aabb(self, ptSets):
        lt = np.array([ptSets[0][0], ptSets[0][1]])
        rb = lt.copy()
        for pt in ptSets:
            lt[0] = min(lt[0], pt[0])
            lt[1] = min(lt[1], pt[1])
            rb[0] = max(rb[0], pt[0])
            rb[1] = max(rb[1], pt[1])
        return lt, rb

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [256, 192], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/256 - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def vis_img(self, name, im):
        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow(name,0)
        cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        #cv2.moveWindow(name,0,0)
        if im.max() > 1:
            im = im/255.
        cv2.imshow(name,im)
        if name != 'mask':
            cv2.waitKey()

    def estimate_trans(self, joints, keypoints):
        keypoints = keypoints.clone().detach().numpy()
        joints = joints.detach().numpy()
        keypoints[:,:-1] = (keypoints[:,:-1] + 1.) * constants.IMG_RES * 0.5

        gt_cam_t = estimate_translation_np(joints, keypoints[:,:2], keypoints[:,2])
        return gt_cam_t

