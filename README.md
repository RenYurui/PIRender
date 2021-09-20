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

The proposed **PIRenderer** can synthesis portrait images by intuitively controlling the face motions with fully disentangled 3DMM parameters. This model can be applied :

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
    <b>Pose&Expression Alignment</b> 
  </p>
  
  
* **Motion Imitation**

  <p align='center'>  
    <img src='https://renyurui.github.io/PIRender_web/reenactment_fast.gif' width='700'/>
  </p>
  <p align='center'>  
    <b>Same&Corss-identity Reenactment</b> 
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

``` shell
# 1. Create a conda virtual environment.
conda create -n PIRenderer python=3.6
conda activate PIRenderer

# 2. Install dependency
pip install -r requirements.txt
```

### 2). Dataset

We train our model using the [VoxCeleb](https://arxiv.org/abs/1706.08612). You can download the demo dataset for inference or prepare the dataset for training and testing.

#### Download the demo dataset

You can download the demo dataset with the following code:

``` bash
./download_dataset.sh
```

#### Prepare the dataset

1. The dataset is preprocessed follow the method used in [First-Order](https://github.com/AliaksandrSiarohin/video-preprocessing). You can follow the instructions in their repo to download and crop videos for training and testing.

2. After obtaining the VoxCeleb videos, we extract 3DMM parameters using [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction). 

3. We save the video and 3DMM parameters in a lmdb file. Please run the following code to do this 

   ``` bash
   python util.write_data_to_lmdb.py
   ```


### 3). Training and Inference

#### Inference

The trained weights can be downloaded by running the following code:

``` bash
./download_weights.sh
```

Or you can choose to download the resources with these links: coming soon. Then save the files to `./result/face`

##### Reenactment 

Run the the demo for face reenactment:

``` bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 inference.py \
--config ./config/face.yaml \
--name face \
--no_resume \
--output_dir ./vox_result/face_reenactment
```

The output results are saved at `./vox_result/face_reenactment`

##### Intuitive Control

coming soon

#### Train

Our model can be trained with the following code

``` bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 train.py \
--config ./config/face.yaml \
--name face
```


## Citation

If you find this code is helpful, please cite our paper

``` tex
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

