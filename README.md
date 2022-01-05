<p align='center'>
  <b>
    <a href="https://renyurui.github.io/PIRender_web/"> Website</a>
    | 
    <a href="https://arxiv.org/abs/2109.08379">ArXiv</a>
    | 
    <a href="#Get-Start">Get Start</a>
    | 
    <a href="https://youtu.be/gDhcRcPI1JU">Video</a>
  </b>
</p> 


# PIRenderer

The source code of the ICCV2021 paper "[PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering](https://arxiv.org/abs/2109.08379)" (ICCV2021)

The proposed **PIRenderer** can synthesis portrait images by intuitively controlling the face motions with fully disentangled 3DMM parameters. This model can be applied to tasks such as:

* **Intuitive Portrait Image Editing**

  <p align='center'>  
    <img src='https://renyurui.github.io/PIRender_web/intuitive_fast.gif' width='700'/>
  </p>
  <p align='center'>  
    <b>Intuitive Portrait Image Control</b> 
  </p>
  <p align='center'>  
    <img src='https://renyurui.github.io/PIRender_web/intuitive_editing_fast.gif' width='700'/>
  </p>
  <p align='center'>  
    <b>Pose & Expression Alignment</b> 
  </p>
  
  
* **Motion Imitation**
  <p align='center'> 
    <img src='https://user-images.githubusercontent.com/30292465/133969233-d7ce0c02-ce6a-4cef-bc5e-d8f55b709f81.gif' width='700'/>
  </p>
  <p align='center'>  
    <b>Same & Corss-identity Reenactment</b> 
  </p>
  
* **Audio-Driven Facial Reenactment**

  <p align='center'>  
    <img src='https://renyurui.github.io/PIRender_web/audio.gif' width='700'/>
  </p>
  <p align='center'>  
    <b>Audio-Driven Reenactment</b> 
  </p>

## News

* 2021.9.20 Code for PyTorch is available!



## Colab Demo

Coming soon


## Get Start

### 1). Installation

#### Requirements

* Python 3
* PyTorch 1.7.1
* CUDA 10.2

#### Conda Installation

```bash
# 1. Create a conda virtual environment.
conda create -n PIRenderer python=3.6
conda activate PIRenderer
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2

# 2. Install other dependencies
pip install -r requirements.txt
```

### 2). Dataset

We train our model using the [VoxCeleb](https://arxiv.org/abs/1706.08612). You can download the demo dataset for inference or prepare the dataset for training and testing.

#### Download the demo dataset

The demo dataset contains all 514 test videos. You can download the dataset with the following code:

```bash
./scripts/download_demo_dataset.sh
```

Or you can choose to download the resources with these links: 

​	[Google Driven](https://drive.google.com/drive/folders/16Yn2r46b4cV6ZozOH6a8SdFz_iG7BQk1?usp=sharing) & [BaiDu Driven](https://pan.baidu.com/s/1e615bBHvM4Wz-2snk-86Xw) with extraction passwords ”p9ab“

Then unzip and save the files to `./dataset`

#### Prepare the dataset

1. The dataset is preprocessed follow the method used in [First-Order](https://github.com/AliaksandrSiarohin/video-preprocessing). You can follow the instructions in their repo to download and crop videos for training and testing.

2. After obtaining the VoxCeleb videos, we extract 3DMM parameters using [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction). 

   The folder are with format as:

   ```
   ${DATASET_ROOT_FOLDER}
   └───path_to_videos
       └───train
           └───xxx.mp4
           └───xxx.mp4
           ...
       └───test
           └───xxx.mp4
           └───xxx.mp4
           ...
   └───path_to_3dmm_coeff
       └───train
           └───xxx.mat
           └───xxx.mat
           ...
       └───test
           └───xxx.mat
           └───xxx.mat
           ...
   ```
   
   **News**: We provide Scripts for extracting 3dmm coeffs from videos. Please check the [DatasetHelper](./DatasetHelper.md) for more details.
   
3. We save the video and 3DMM parameters in a lmdb file. Please run the following code to do this 

   ```bash
   python scripts/prepare_vox_lmdb.py \
   --path path_to_videos \
   --coeff_3dmm_path path_to_3dmm_coeff \
   --out path_to_output_dir
   ```

### 3). Training and Inference

#### Inference

The trained weights can be downloaded by running the following code:

```bash
./scripts/download_weights.sh
```

Or you can choose to download the resources with these links: 

[Google Driven](https://drive.google.com/file/d/1-0xOf6g58OmtKtEWJlU3VlnfRqPN9Uq7/view?usp=sharing) & [Baidu Driven](https://pan.baidu.com/s/18B3xfKMXnm4tOqlFSB8ntg) with extraction passwards "4sy1".

Then unzip and save the files to `./result/face`.

**Reenactment**

Run the demo for face reenactment:

```bash
# same identity
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 inference.py \
--config ./config/face_demo.yaml \
--name face \
--no_resume \
--output_dir ./vox_result/face_reenactment

# cross identity
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 inference.py \
--config ./config/face_demo.yaml \
--name face \
--no_resume \
--output_dir ./vox_result/face_reenactment_cross \
--cross_id
```

The output results are saved at `./vox_result/face_reenactment` and `./vox_result/face_reenactment_cross`

**Intuitive Control**

Our model can generate results by providing intuitive controlling coefficients. 
We provide the following code for this task. Please note that you need to build the environment of [DeepFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/73d491102af6731bded9ae6b3cc7466c3b2e9e48) first.

```bash
# 1. Copy the provided scrips to the folder `Deep3DFaceRecon_pytorch`.
cp scripts/face_recon_videos.py ./Deep3DFaceRecon_pytorch
cp scripts/extract_kp_videos.py ./Deep3DFaceRecon_pytorch
cp scripts/coeff_detector.py ./Deep3DFaceRecon_pytorch
cp scripts/inference_options.py ./Deep3DFaceRecon_pytorch/options

cd Deep3DFaceRecon_pytorch

# 2. Extracte the 3dmm coefficients of the demo images.
python coeff_detector.py \
--input_dir ../demo_images \
--keypoint_dir ../demo_images \
--output_dir ../demo_images \
--name=model_name \
--epoch=20 \
--model facerecon   

# 3. control the source image with our model
cd ..
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 intuitive_control.py \
--config ./config/face_demo.yaml \
--name face \
--no_resume \
--output_dir ./vox_result/face_intuitive \
--input_name ./demo_images
```


#### Train

Our model can be trained with the following code

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 train.py \
--config ./config/face.yaml \
--name face
```


## Citation

If you find this code is helpful, please cite our paper

```tex
@misc{ren2021pirenderer,
      title={PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering}, 
      author={Yurui Ren and Ge Li and Yuanqi Chen and Thomas H. Li and Shan Liu},
      year={2021},
      eprint={2109.08379},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement 

We build our project base on [imaginaire](https://github.com/NVlabs/imaginaire). Some dataset preprocessing methods are derived from [video-preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing).

