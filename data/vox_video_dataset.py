import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO

import torch

from data.vox_dataset import VoxDataset
from data.vox_dataset import format_for_lmdb

class VoxVideoDataset(VoxDataset):
    def __init__(self, opt, is_inference):
        super(VoxVideoDataset, self).__init__(opt, is_inference)
        self.video_index = -1
        self.cross_id = opt.cross_id
        # whether normalize the crop parameters when performing cross_id reenactments
        # set it as "True" always brings better performance
        self.norm_crop_param = True

    def __len__(self):
        return len(self.video_items)

    def load_next_video(self):
        data={}
        self.video_index += 1
        video_item = self.video_items[self.video_index]
        source_video_item = self.random_video(video_item) if self.cross_id else video_item 

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(source_video_item['video_name'], 0)
            img_bytes_1 = txn.get(key) 
            img1 = Image.open(BytesIO(img_bytes_1))
            data['source_image'] = self.transform(img1)

            semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((video_item['num_frame'],-1))
            if self.cross_id and self.norm_crop_param:
                semantics_source_key = format_for_lmdb(source_video_item['video_name'], 'coeff_3dmm')
                semantics_source_numpy = np.frombuffer(txn.get(semantics_source_key), dtype=np.float32)
                semantic_source_numpy = semantics_source_numpy.reshape((source_video_item['num_frame'],-1))[0:1]
                crop_norm_ratio = self.find_crop_norm_ratio(semantic_source_numpy, semantics_numpy)
            else:
                crop_norm_ratio = None            

            data['target_image'], data['target_semantics'] = [], []
            for frame_index in range(video_item['num_frame']):
                key = format_for_lmdb(video_item['video_name'], frame_index)
                img_bytes_1 = txn.get(key) 
                img1 = Image.open(BytesIO(img_bytes_1))
                data['target_image'].append(self.transform(img1))
                data['target_semantics'].append(
                    self.transform_semantic(semantics_numpy, frame_index, crop_norm_ratio)
                )
            data['video_name'] = self.obtain_name(video_item['video_name'], source_video_item['video_name'])
        return data  
    
    def random_video(self, target_video_item):
        target_person_id = target_video_item['person_id']
        assert len(self.person_ids) > 1 
        source_person_id = np.random.choice(self.person_ids)
        if source_person_id == target_person_id:
            source_person_id = np.random.choice(self.person_ids)
        source_video_index = np.random.choice(self.idx_by_person_id[source_person_id])
        source_video_item = self.video_items[source_video_index]
        return source_video_item

    def find_crop_norm_ratio(self, source_coeff, target_coeffs):
        alpha = 0.3
        exp_diff = np.mean(np.abs(target_coeffs[:,80:144] - source_coeff[:,80:144]), 1)
        angle_diff = np.mean(np.abs(target_coeffs[:,224:227] - source_coeff[:,224:227]), 1)
        index = np.argmin(alpha*exp_diff + (1-alpha)*angle_diff)
        crop_norm_ratio = source_coeff[:,-3] / target_coeffs[index:index+1, -3]
        return crop_norm_ratio
   
    def transform_semantic(self, semantic, frame_index, crop_norm_ratio):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index,...]
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:300] #crop param

        if self.cross_id and self.norm_crop_param:
            crop[:, -3] = crop[:, -3] * crop_norm_ratio

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)   

    def obtain_name(self, target_name, source_name):
        if not self.cross_id:
            return target_name
        else:
            source_name = os.path.splitext(os.path.basename(source_name))[0]
            return source_name+'_to_'+target_name