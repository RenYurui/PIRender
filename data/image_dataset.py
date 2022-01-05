import os
import glob
import time
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F



class ImageDataset():
    def __init__(self, opt, input_name):
        self.opt = opt
        self.IMAGEEXT = ['png', 'jpg']
        self.input_image_list, self.coeff_list = self.obtain_inputs(input_name)
        self.index = -1
        # load image dataset opt
        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius

    def next_image(self):
        self.index += 1
        image_name = self.input_image_list[self.index]
        coeff_name = self.coeff_list[self.index]
        img = Image.open(image_name)
        input_image = self.trans_image(img)

        coeff_3dmm = np.loadtxt(coeff_name).astype(np.float32)
        coeff_3dmm = self.transform_semantic(coeff_3dmm)
        
        return {
            'source_image': input_image[None],
            'target_semantics': coeff_3dmm[None],
            'name': os.path.splitext(os.path.basename(image_name))[0]
        }

    def obtain_inputs(self, root):
        filenames = list()

        IMAGE_EXTENSIONS_LOWERCASE = {'jpg', 'png', 'jpeg', 'webp'}
        IMAGE_EXTENSIONS = IMAGE_EXTENSIONS_LOWERCASE.union({f.upper() for f in IMAGE_EXTENSIONS_LOWERCASE})
        extensions = IMAGE_EXTENSIONS

        for ext in extensions:
            filenames += glob.glob(f'{root}/*.{ext}', recursive=True)
        filenames = sorted(filenames)
        coeffnames = sorted(glob.glob(f'{root}/*_3dmm_coeff.txt'))     

        return filenames, coeffnames

    def transform_semantic(self, semantic):
        semantic = semantic[None].repeat(self.semantic_radius*2+1, 0)
        ex_coeff = semantic[:,80:144] #expression
        angles = semantic[:,224:227] #euler angles for pose
        translation = semantic[:,254:257] #translation
        crop = semantic[:,259:262] #crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)   

    def trans_image(self, image):
        image = F.resize(
            image, size=self.resolution, interpolation=Image.BICUBIC)
        image = F.to_tensor(image)
        image = F.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        return image
        
    def __len__(self):
        return len(self.input_image_list)

        
