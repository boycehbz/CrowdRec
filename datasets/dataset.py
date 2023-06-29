'''
 @FileName    : dataset.py
 @EditTime    : 2022-09-27 16:03:55
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
import torch
import numpy as np
import cv2
from utils.imutils import surface_projection, vis_img
from datasets.base import base
import utils.constants as constants


class MyData(base):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None):
        super(MyData, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)
        self.is_train = False
    
        dataset_annot = os.path.join(self.dataset_dir, 'annot/test.pkl')

        params = self.load_pkl(dataset_annot)

        self.pose2ds, self.imnames, self.instance, self.bboxs, self.pose3ds, self.intris = [], [], [], [], [], []
        for seqid, seq in enumerate(params):
            # if seqid != 1:
            #     continue
            if len(seq) < 1:
                continue
            for i, frame in enumerate(seq):
                h, w = frame['h_w']
                if w > 33000 or h > 25000:
                    continue
                pose2d, ins, bbox, pose3d, intri = [], [], [], [], []
                img_path = frame['img_path'].replace('\\', '/')
                self.imnames.append(img_path)
                del frame['img_path']
                del frame['h_w']
                for j, key in enumerate(frame.keys()):
                    if j % 50 == 0 and j != 0:
                        self.instance.append(ins)
                        self.pose2ds.append(pose2d)
                        self.bboxs.append(bbox)
                        self.pose3ds.append(pose3d)
                        self.intris.append(intri)
                        pose2d, ins, bbox, pose3d, intri = [], [], [], [], []
                        if len(frame) - j >= 8:
                            self.imnames.append(img_path)
                    ins.append(os.path.join(img_path.rstrip('.jpg'), key))
                    pred_2d = np.array(frame[key]['halpe_joints_2d_pred'], dtype=self.np_type) if frame[key]['halpe_joints_2d_pred'] is not None else np.zeros((26,3), dtype=self.np_type)
                    pose2d.append(pred_2d)
                    pose3d.append(np.array(frame[key]['halpe_joints_3d'], dtype=self.np_type) if frame[key]['halpe_joints_3d'] is not None else None)
                    bbox.append(np.array(frame[key]['bbox'], dtype=self.np_type).reshape(-1,))
                    intri.append(np.array(frame[key]['intri'], dtype=self.np_type) if frame[key]['intri'] is not None else None)
                if len(ins) < 8 and j >= 50:
                    self.instance[-1] = self.instance[-1] + ins
                    self.pose2ds[-1] = self.pose2ds[-1] + pose2d
                    self.bboxs[-1] = self.bboxs[-1] + bbox
                    self.pose3ds[-1] = self.pose3ds[-1] + pose3d
                    self.intris[-1] = self.intris[-1] + intri
                else:
                    self.instance.append(ins)
                    self.pose2ds.append(pose2d)
                    self.bboxs.append(bbox)
                    self.pose3ds.append(pose3d)
                    self.intris.append(intri)
        del frame
        del params

        self.len = len(self.imnames)


    def vis_input(self, image, keypoints):
        # Show image
        mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
        std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
        image = image.transpose((1,2,0))
        image = image[:,:,::-1]
        image = image * std + mean
        image = image
        vis_img('img', image)

        # Show keypoints
        keypoints = (keypoints[:,:-1].detach().numpy() + 1.) * 256 * 0.5
        keypoints = keypoints.astype(np.int)
        image = (image*255).astype(np.uint8)
        for k in keypoints:
            image = cv2.circle(image, tuple(k), 3, (0,0,255), -1)
        self.vis_img('keyp', image)



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


    # Data preprocess
    def create_data(self, index=0):
        
        load_data = {}
        
        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        
        # Load image
        imgname = os.path.join(self.dataset_dir, self.imnames[index])
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        img_h, img_w, _ = img.shape
        load_data["origin_img"] = imgname

        num_people = len(self.bboxs[index])

        gt_jointss = torch.zeros((num_people, 26, 4)).float()
        imgnames = ['empty'] * num_people
        norm_imgs = torch.zeros((num_people, 3, 256, 192)).float()
        centers = torch.zeros((num_people, 2)).float()
        scales = torch.zeros((num_people)).float()
        crop_uls = np.zeros((num_people, 2), dtype=np.float32)
        crop_brs = np.zeros((num_people, 2), dtype=np.float32)
        img_hs = np.zeros((num_people), dtype=np.float32)
        img_ws = np.zeros((num_people), dtype=np.float32)
        focal_lengthes = np.zeros((num_people), dtype=np.float32)
        pose2d_gt = torch.zeros((num_people, 26, 3)).float()
        pose2d_origin = torch.zeros((num_people, 26, 3)).float()

        for i in range(num_people):

            if self.intris[index][i] is not None:
                focal_length = self.intris[index][i][0][0]
            else:
                focal_length = (img_h ** 2 + img_w ** 2) ** 0.5

            bbox = self.bboxs[index][i]

            norm_img, center, scale, crop_ul, crop_br, _ = self.process_image(img.copy(), bbox)

            # Get 2D keypoints and apply augmentation transforms
            h = 200 * scale
            s = float(256) / h
            keypoints = torch.from_numpy(self.pose2ds[index][i].copy()).float() 
            keypoints[:,:2] = keypoints[:,:2] - center
            keypoints[:,:2] = (keypoints[:,:2] * s) / 256
            origin_keypoints = torch.from_numpy(self.pose2ds[index][i].copy()).float() 

            if self.pose3ds[index][i] is None:
                gt_joints = torch.zeros((26, 4)).float()
            else:
                gt_joints = self.pose3ds[index][i].copy()
                mask = np.sum(gt_joints, axis=1)
                mask = 1 - (mask == 0).astype(np.float)
                mask = mask.reshape(-1, 1)
                gt_joints = np.concatenate((gt_joints, mask), axis=1)
                gt_joints = torch.from_numpy(gt_joints).float() 

            gt_jointss[i] = gt_joints
            imgnames[i] = self.instance[index][i]
            norm_imgs[i] = torch.from_numpy(norm_img)
            centers[i] = center
            scales[i] = scale
            crop_uls[i] = crop_ul
            crop_brs[i] = crop_br
            img_hs[i] = img_h
            img_ws[i] = img_w
            focal_lengthes[i] = focal_length
            pose2d_gt[i] = keypoints
            pose2d_origin[i] = origin_keypoints

        load_data["gt_joints"] = gt_jointss
        load_data['instance'] = imgnames
        load_data["norm_img"] = norm_imgs
        load_data["center"] = centers
        load_data["scale"] = scales
        load_data["crop_ul"] = crop_uls
        load_data["crop_br"] = crop_brs
        load_data["img_h"] = img_hs
        load_data["img_w"] = img_ws
        load_data["focal_length"] = focal_lengthes
        load_data["keypoints"] = pose2d_gt
        load_data["origin_keypoints"] = pose2d_origin


        # self.vis_input(norm_img, keypoints)

        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len













