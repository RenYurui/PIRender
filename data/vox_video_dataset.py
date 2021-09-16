import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO

from data.vox_dataset import VoxDataset
from data.vox_dataset import format_for_lmdb

class VoxVideoDataset(VoxDataset):
    def __init__(self, opt, is_inference):
        super(VoxVideoDataset, self).__init__(opt, is_inference)
        self.video_index = -1

    def __len__(self):
        return len(self.person_ids)

    def load_next_video(self):
        data={}
        self.video_index += 1
        video_item = self.video_items[self.video_index]

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 0)
            img_bytes_1 = txn.get(key) 
            img1 = Image.open(BytesIO(img_bytes_1))
            data['source_image'] = self.transform(img1)

            semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((video_item['num_frame'],-1))

            data['target_image'], data['target_semantics'] = [], []
            for frame_index in range(video_item['num_frame']):
                key = format_for_lmdb(video_item['video_name'], frame_index)
                img_bytes_1 = txn.get(key) 
                img1 = Image.open(BytesIO(img_bytes_1))
                data['target_image'].append(self.transform(img1))
                data['target_semantics'].append(
                    self.transform_semantic(semantics_numpy, frame_index)
                )
            data['video_name'] = video_item['video_name']
        return data  
   
