import os
import math
import argparse
import numpy as np
from scipy.io import savemat,loadmat

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from config import Config
from util.logging import init_logging, make_logging_dir
from util.distributed import init_dist
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from util.distributed import master_only_print as print
from data.image_dataset import ImageDataset
from inference import write2video


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--input_name', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    return args

def get_control(input_name):
    control_dict = {}
    control_dict['rotation_center'] = torch.tensor([0,0,0,0,0,0.45])
    control_dict['rotation_left_x'] = torch.tensor([0,0,math.pi/10,0,0,0.45])
    control_dict['rotation_right_x'] = torch.tensor([0,0,-math.pi/10,0,0,0.45])

    control_dict['rotation_left_y'] = torch.tensor([math.pi/10,0,0,0,0,0.45])
    control_dict['rotation_right_y'] = torch.tensor([-math.pi/10,0,0,0,0,0.45])        

    control_dict['rotation_left_z'] = torch.tensor([0,math.pi/8,0,0,0,0.45])
    control_dict['rotation_right_z'] = torch.tensor([0,-math.pi/8,0,0,0,0.45])   

    expession = loadmat('{}/expression.mat'.format(input_name))

    for item in ['expression_center', 'expression_mouth', 'expression_eyebrow', 'expression_eyes']:
        control_dict[item] = torch.tensor(expession[item])[0]

    sort_rot_control = [
                        'rotation_left_x',  'rotation_center', 
                        'rotation_right_x', 'rotation_center',
                        'rotation_left_y',  'rotation_center',
                        'rotation_right_y', 'rotation_center',
                        'rotation_left_z',  'rotation_center',
                        'rotation_right_z', 'rotation_center'
                        ]
    
    sort_exp_control = [
                        'expression_center', 'expression_mouth',
                        'expression_center', 'expression_eyebrow',
                        'expression_center', 'expression_eyes',
                        ]
    return control_dict, sort_rot_control, sort_exp_control

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)

    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = torch.cuda.current_device()

    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)

    # create a model
    net_G, net_G_ema, opt_G, sch_G \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_G_ema, \
                          opt_G, sch_G, None)

    current_epoch, current_iteration = trainer.load_checkpoint(
        opt, args.which_iter)                          
    net_G = trainer.net_G_ema.eval()

    output_dir = os.path.join(
        args.output_dir, 
        'epoch_{:05}_iteration_{:09}'.format(current_epoch, current_iteration)
        )

    os.makedirs(output_dir, exist_ok=True)
    image_dataset = ImageDataset(opt.data, args.input_name)

    control_dict, sort_rot_control, sort_exp_control = get_control(args.input_name)
    for _ in range(image_dataset.__len__()):
        with torch.no_grad():
            data = image_dataset.next_image()
            num = 10
            output_images = []     
            # rotation control
            current = control_dict['rotation_center']
            for control in sort_rot_control: 
                for i in range(num):
                    rotation = (control_dict[control]-current)*i/(num-1)+current
                    data['target_semantics'][:, 64:70, :] = rotation[None, :, None]
                    output_dict = net_G(data['source_image'].cuda(), data['target_semantics'].cuda())
                    output_images.append(
                        output_dict['fake_image'].cpu().clamp_(-1, 1)
                        )    
                current = rotation

            # expression control
            current = data['target_semantics'][0, :64, 0]
            for control in sort_exp_control: 
                for i in range(num):
                    expression = (control_dict[control]-current)*i/(num-1)+current
                    data['target_semantics'][:, :64, :] = expression[None, :, None]
                    output_dict = net_G(data['source_image'].cuda(), data['target_semantics'].cuda())
                    output_images.append(
                        output_dict['fake_image'].cpu().clamp_(-1, 1)
                        )    
                current = expression
            output_images = torch.cat(output_images, 0)   
            print('write results to file {}/{}'.format(output_dir, data['name']))
            write2video('{}/{}'.format(output_dir, data['name']), output_images)

