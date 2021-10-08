import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import savemat

import torch 

from models import create_model
from options.inference_options import InferenceOptions
from util.preprocess import align_img
from util.load_mats import load_lm3d
from util.util import tensor2im, save_image


def get_data_path(root, keypoint_root):
    filenames = list()
    keypoint_filenames = list()

    IMAGE_EXTENSIONS_LOWERCASE = {'jpg', 'png', 'jpeg', 'webp'}
    IMAGE_EXTENSIONS = IMAGE_EXTENSIONS_LOWERCASE.union({f.upper() for f in IMAGE_EXTENSIONS_LOWERCASE})
    extensions = IMAGE_EXTENSIONS

    for ext in extensions:
        filenames += glob.glob(f'{root}/*.{ext}', recursive=True)
    filenames = sorted(filenames)
    for filename in filenames:
        name = os.path.splitext(os.path.basename(filename))[0]
        keypoint_filenames.append(
            os.path.join(keypoint_root, name + '.txt')
        )
    return filenames, keypoint_filenames


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, txt_filenames, bfm_folder):
        self.filenames = filenames
        self.txt_filenames = txt_filenames
        self.lm3d_std = load_lm3d(bfm_folder) 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        filename = self.filenames[i]
        txt_filename = self.txt_filenames[i]
        imgs, _, trans_params = self.read_data(filename, txt_filename)
        return {
            'imgs':imgs,
            'trans_param':trans_params,
            'filename': filename
        }

    def image_transform(self, images, lm):
        W,H = images.size
        if np.mean(lm) == -1:
            lm = (self.lm3d_std[:, :2]+1)/2.
            lm = np.concatenate(
                [lm[:, :1]*W, lm[:, 1:2]*H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, img, lm, _ = align_img(images, lm, self.lm3d_std)        
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        lm = torch.tensor(lm)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, lm, trans_params        

    def read_data(self, filename, txt_filename):
        images = Image.open(filename).convert('RGB')
        lm = np.loadtxt(txt_filename).astype(np.float32)
        lm = lm.reshape([-1, 2]) 
        imgs, lms, trans_params = self.image_transform(images, lm)
        return imgs, lms, trans_params


def main(opt, model):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    filenames, keypoint_filenames = get_data_path(opt.input_dir, opt.keypoint_dir)
        
    dataset = ImagePathDataset(filenames, keypoint_filenames, opt.bfm_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    ) 
    pred_coeffs, pred_trans_params = [], []
    print('nums of images:', dataset.__len__())
    for iteration, data in tqdm(enumerate(dataloader)):
        data_input = {                
                'imgs': data['imgs'],
                }
        
        model.set_input(data_input)  
        model.test()
        pred_coeff = {key:model.pred_coeffs_dict[key].cpu().numpy() for key in model.pred_coeffs_dict}
        pred_coeff = np.concatenate([
            pred_coeff['id'], 
            pred_coeff['exp'], 
            pred_coeff['tex'], 
            pred_coeff['angle'],
            pred_coeff['gamma'],
            pred_coeff['trans']], 1)
        pred_coeffs.append(pred_coeff) 
        trans_param = data['trans_param'].cpu().numpy()
        pred_trans_params.append(trans_param)
        if opt.save_split_files:
            for index, filename in enumerate(data['filename']):
                basename = os.path.splitext(os.path.basename(filename))[0]
                output_path = os.path.join(opt.output_dir, basename+'.mat')
                savemat(
                    output_path, 
                    {'coeff':pred_coeff[index], 
                    'transform_params':trans_param[index]}
                )
        # visuals = model.get_current_visuals()  # get image results
        # for name in visuals:
        #     images = visuals[name]
        #     for i in range(images.shape[0]):
        #         image_numpy = tensor2im(images[i])
        #         save_image(image_numpy, os.path.basename(data['filename'][i])+'.png')                

    pred_coeffs = np.concatenate(pred_coeffs, 0)
    pred_trans_params = np.concatenate(pred_trans_params, 0)
    savemat(os.path.join(opt.output_dir, 'ffhq.mat'), {'coeff':pred_coeffs, 'transform_params':pred_trans_params})


if __name__ == '__main__':
    opt = InferenceOptions().parse()  # get test options
    model = create_model(opt)
    model.setup(opt)
    model.device = 'cuda:0'
    model.parallelize()
    model.eval()
    lm3d_std = load_lm3d(opt.bfm_folder) 
    main(opt, model)


