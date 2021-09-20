import os
import cv2
import lmdb
import argparse
import multiprocessing
import numpy as np

from glob import glob
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
from torchvision.transforms import functional as trans_fn

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

class Resizer:
    def __init__(self, size, kp_root, coeff_3dmm_root, img_format):
        self.size = size
        self.kp_root = kp_root
        self.coeff_3dmm_root = coeff_3dmm_root
        self.img_format = img_format

    def get_resized_bytes(self, img, img_format='jpeg'):
        img = trans_fn.resize(img, (self.size, self.size), interpolation=Image.BICUBIC)
        buf = BytesIO()
        img.save(buf, format=img_format)
        img_bytes = buf.getvalue()
        return img_bytes

    def prepare(self, filename):
        frames = {'img':[], 'kp':None, 'coeff_3dmm':None}
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame)
                img_bytes = self.get_resized_bytes(img_pil, self.img_format)
                frames['img'].append(img_bytes)
            else:
                break
        cap.release()
        video_name = os.path.splitext(os.path.basename(filename))[0]
        keypoint_byte = get_others(self.kp_root, video_name, 'keypoint')
        coeff_3dmm_byte = get_others(self.coeff_3dmm_root, video_name, 'coeff_3dmm')
        frames['kp'] = keypoint_byte
        frames['coeff_3dmm'] = coeff_3dmm_byte
        return frames

    def __call__(self, index_filename):
        index, filename = index_filename
        result = self.prepare(filename)
        return index, result, filename

def get_others(root, video_name, data_type):
    if root is None:
        return
    else:
        assert data_type in ('keypoint', 'coeff_3dmm')
    if os.path.isfile(os.path.join(root, 'train', video_name+'.mat')):
        file_path = os.path.join(root, 'train', video_name+'.mat')
    else:
        file_path = os.path.join(root, 'test', video_name+'.mat')
    
    if data_type == 'keypoint':
        return_byte = convert_kp(file_path)
    else:
        return_byte = convert_3dmm(file_path)
    return return_byte

def convert_kp(file_path):
    file_mat = loadmat(file_path)
    kp_byte = file_mat['landmark'].tobytes()
    return kp_byte

def convert_3dmm(file_path):
    file_mat = loadmat(file_path)
    coeff_3dmm = file_mat['coeff']
    crop_param = file_mat['transform_params']
    _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
    crop_param = np.concatenate([ratio, t0, t1], 1)
    coeff_3dmm_cat = np.concatenate([coeff_3dmm, crop_param], 1) 
    coeff_3dmm_byte = coeff_3dmm_cat.tobytes()
    return coeff_3dmm_byte


def prepare_data(path, keypoint_path, coeff_3dmm_path, out, n_worker, sizes, chunksize, img_format):
    filenames = list()
    VIDEO_EXTENSIONS_LOWERCASE = {'mp4'}
    VIDEO_EXTENSIONS = VIDEO_EXTENSIONS_LOWERCASE.union({f.upper() for f in VIDEO_EXTENSIONS_LOWERCASE})
    extensions = VIDEO_EXTENSIONS
    for ext in extensions:
        filenames += glob(f'{path}/**/*.{ext}', recursive=True)
    train_video, test_video = [], []
    for item in filenames:
        if "/train/" in item:
            train_video.append(item)
        else:
            test_video.append(item)
    print(len(train_video), len(test_video))
    with open(os.path.join(out, 'train_list.txt'),'w') as f:
        for item in train_video:
            item = os.path.splitext(os.path.basename(item))[0]
            f.write(item + '\n')

    with open(os.path.join(out, 'test_list.txt'),'w') as f:
        for item in test_video:
            item = os.path.splitext(os.path.basename(item))[0]
            f.write(item + '\n')      


    filenames = sorted(filenames)
    total = len(filenames)
    os.makedirs(out, exist_ok=True)
    for size in sizes:
        lmdb_path = os.path.join(out, str(size))
        with lmdb.open(lmdb_path, map_size=1024 ** 4, readahead=False) as env:
            with env.begin(write=True) as txn:
                txn.put(format_for_lmdb('length'), format_for_lmdb(total))
                resizer = Resizer(size, keypoint_path, coeff_3dmm_path, img_format)
                with multiprocessing.Pool(n_worker) as pool:
                    for idx, result, filename in tqdm(
                            pool.imap_unordered(resizer, enumerate(filenames), chunksize=chunksize),
                            total=total):
                        filename = os.path.basename(filename)
                        video_name = os.path.splitext(filename)[0]
                        txn.put(format_for_lmdb(video_name, 'length'), format_for_lmdb(len(result['img'])))

                        for frame_idx, frame in enumerate(result['img']):
                            txn.put(format_for_lmdb(video_name, frame_idx), frame)

                        if result['kp']:
                            txn.put(format_for_lmdb(video_name, 'keypoint'), result['kp'])
                        if result['coeff_3dmm']:
                            txn.put(format_for_lmdb(video_name, 'coeff_3dmm'), result['coeff_3dmm'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='a path to input directiory')
    parser.add_argument('--keypoint_path', type=str, help='a path to output directory', default=None)
    parser.add_argument('--coeff_3dmm_path', type=str, help='a path to output directory', default=None)
    parser.add_argument('--out', type=str, help='a path to output directory')
    parser.add_argument('--sizes', type=int, nargs='+', default=(256,))
    parser.add_argument('--n_worker', type=int, help='number of worker processes', default=8)
    parser.add_argument('--chunksize', type=int, help='approximate chunksize for each worker', default=10)
    parser.add_argument('--img_format', type=str, default='jpeg')
    args = parser.parse_args()
    prepare_data(**vars(args))