### Extract 3DMM Coefficients for Videos

We provide scripts for extracting 3dmm coefficients for videos by using [DeepFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/73d491102af6731bded9ae6b3cc7466c3b2e9e48).

1. Follow the instructions of their repo to build the environment of DeepFaceRecon.

2. Copy the provided scrips to the folder `Deep3DFaceRecon_pytorch`.

   ```bash
   cp scripts/face_recon_videos.py ./Deep3DFaceRecon_pytorch
   cp scripts/extract_kp_videos.py ./Deep3DFaceRecon_pytorch
   cp scripts/coeff_detector.py ./Deep3DFaceRecon_pytorch
   cp scripts/inference_options.py ./Deep3DFaceRecon_pytorch/options

   cd Deep3DFaceRecon_pytorch
   ```

3. Extract facial landmarks from videos.

   ```bash
   python extract_kp_videos.py \
   --input_dir path_to_viodes \
   --output_dir path_to_keypoint \
   --device_ids 0,1,2,3 \
   --workers 12
   ```

4. Extract coefficients for videos

   ```bash
   python face_recon_videos.py \
   --input_dir path_to_videos \
   --keypoint_dir path_to_keypoint \
   --output_dir output_dir \
   --inference_batch_size 100 \
   --name=model_name \
   --epoch=20 \
   --model facerecon
   ```

   



