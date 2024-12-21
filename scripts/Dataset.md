## Dataset construction pipeline
The code structure is shown as follows:
```
scripts
├── XVFI-main
├── run.sh
├── ...
└── deblur_nerf_data
    ├── factory
    │   ├── raw_data
    │   ├── blur_data
    │   ├── spike_data
    │   └── sharp_data
    ├── ...
    └── wine
        ├── raw_data
        ├── blur_data
        ├── spike_data
        └── sharp_data
```
### Step 1: Blender

The synthetic dataset scenes are from [Deblur-NeRF](https://github.com/limacv/Deblur-NeRF/). Since we need to simulate high frame-rate spike stream, 455 images are exported for each scene during rendering. The rendered images are saved under the folder `deblur_nerf_data/{scene}/raw_data`, where `scene` corresponds to different scene names in Deblur-NeRF.

### Step 2: Frame Interpolation
We use the XVFI frame interpolation algorithm on the raw data folder `deblur_nerf_data/{scene}/raw_data` to insert 7 additional imgs between two adjacent imgs, increasing the frame rate of the image sequence, which is time-consuming and takes up a large amount of space. Run
```
cd XVFI-main/
python main.py --custom_path deblur_nerf_data/wine --gpu 0 --phase test_custom --exp_num 1 --dataset X4K1000FPS --module_scale_factor 4 --S_tst 5 --multiple 8 
cd ..
```

### Step 3: Blur Synthesis
We synthesize a blurred frame using 97 images in the dataset after frame interpolation (corresponding to 13 images before interpolation) on the raw data folder `deblur_nerf_data/{scene}/raw_data`. Run
```
python blur_syn.py --raw_folder deblur_nerf_data/wine/raw_data --overlap_len 7 --blur_num 13 --height 400 --width 600
```

> The blur generation in this project does not work, it is just for visualization of what the images should look like when captured by an RGB camera in a high-speed motion scene, as analyzed in [S-SDM](https://github.com/chenkang455/S-SDM).


### Step 4: Spike Simulation
We apply a spike generation physical model to simulate spikes, obtaining the spike stream corresponding to the virtual exposure time in Step 3: Blur Synthesis. Run
```
python spike_simulate.py --raw_folder deblur_nerf_data/wine/raw_data --overlap_len 7 --blur_num 13 --spike_add 40 --height 400 --width 600
```
Since the spike stream also contains the additional 40 spike frames out of the exposure period, the start and end segments of the spike stream cannot be utilized. (For example, 000006.png is the first frame of the blur_data folder while 000019.dat is the first spike data of the spike_data folder.)

❗ Here is an explanation of why there are 40 redundant frames before and after each ideal exposure window: the ideal exposure window set by Recon-Net in this paper is 41 frames. This means that to reconstruct the image of the first frame in the exposure window, we need the 20 frames before and 20 frames after along with the current frame. The additional 20 frames are provided because we found that the spike simulator has a large error during the initial process, so an extra 20 frames of redundant signals are used to simulate and generate the spike. This is reflected in the calculation method for the long TFP: `tfp_long = torch.mean(data['spike'][40:-40].float(), dim=0, keepdim=False)[..., None].repeat(1, 1, 3)`.


### Step 5: Sharp Extract
For obtaining the single sharp frame corresponding to each view:
```
python sharp_extract.py --raw_folder deblur_nerf_data/wine/raw_data --overlap_len 7 --blur_num 13
```

### Step 6: Pose Estimation
For pose estimation, run `COLMAP` implemented by the `nerfstudio` on the sharp image folder:
```
ns-process-data images \
    --data deblur_nerf_data/wine/sharp_data \
    --output-dir deblur_nerf_data/wine
```
