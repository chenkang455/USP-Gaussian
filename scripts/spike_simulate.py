import os
import cv2
import numpy as np
from tqdm import trange
from utils import *
from utils_spike import *
import argparse

#? convert the imgs under raw_folder to spike sequence on spike_folder
#? Structure as:
#?  base_folder
#?  ├── blur_folder
#?  ├── raw_folder
#?  └── spike_folder

# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_folder', type=str, default=r'GOPRO/test/raw_data')
    parser.add_argument('--img_type', type=str, default='.png')
    parser.add_argument('--overlap_len', type=int, default= 7,
                    help = 'Overlap length between two blurry images. Assume [0-12] -> blur_6, [13,25] -> blur_19, overlap_len is the length of interpolated frames between 12.png and 13.png,i.e.,12_0.png,...12_6.png')
    parser.add_argument('--height', type=int, default= 720)
    parser.add_argument('--width', type=int, default= 1280)
    parser.add_argument('--use_resize', action='store_true', help = 'Resize the image size to the half.')
    parser.add_argument('--blur_num', type=int, default= 13,help = 'Number of images before interpolation to synthesize one blurry frame.')
    parser.add_argument('--spike_add', type=int, default= 40,help = 'Additional spike out of the exposure period.')
    
    opt = parser.parse_args()
    width = opt.width
    height = opt.height
    resize = opt.use_resize
    raw_folder = opt.raw_folder
    base_folder = os.path.dirname(raw_folder)
    spike_folder = os.path.join(base_folder,'spike_data')
    os.makedirs(spike_folder,exist_ok = True)
    for dirpath, sub_dirs, sub_files in os.walk(raw_folder):
        if len(sub_files) == 0 or sub_files[0].endswith(opt.img_type) == False:
            continue
        print(dirpath)
        output_folder = dirpath.replace(raw_folder,spike_folder)
        os.makedirs(output_folder,exist_ok = True)
        sub_files = sorted(sub_files)
        bais = 0 # bais that overlap between two blurry imgs
        num_blur_raw = opt.blur_num # number of sharp imgs before interpolated per blurry one
        num_inter = 7 # number of interpolated imgs between two imgs
        num_blur = (num_blur_raw - 1) * (num_inter + 1) + 1 # number of interpolated imgs per blurry one
        spike_add = opt.spike_add # additional spike out of the exposure period
        num_omit = 1 # reduce the number of spike sequence to [num/num_omit]
        str_len = len(sub_files[0].split('.')[0])
        imgs = []
        start = 0
        for i in trange(len(sub_files)):
            if i + bais >= len(sub_files):
                break
            file_name = sub_files[i + bais]
            img = cv2.imread(os.path.join(dirpath,file_name))
            img = img.astype(np.float32) / 255
            if resize == True:
                img = cv2.resize(img,(width // 4,height // 4),interpolation=cv2.INTER_LINEAR)
            # GRAY=0.3*R+0.59*G+0.11*B 
            img = 0.11 * img[...,0] + 0.59 * img[...,1] + 0.3 * img[...,2]
            imgs.append(img)
            # simulate the spike sequence during the exposure
            if i % (num_blur) == num_blur - 1:
                end = i + bais
                # skip the first blurry image
                if start == 0:
                    imgs = []
                    bais += opt.overlap_len
                    start = i + bais + 1
                    continue
                if end + spike_add >= len(sub_files):
                    break
                # add the spike data out of the exposure period
                for jj in range(1,spike_add + 1):
                    img_start = cv2.imread(os.path.join(dirpath,sub_files[start - jj]))
                    if resize == True:
                        img_start = cv2.resize(img_start,(width // 4,height // 4),interpolation=cv2.INTER_LINEAR )
                    img_start = img_start.astype(np.float32) / 255
                    img_start = 0.11 * img_start[...,0] + 0.59 * img_start[...,1] + 0.3 * img_start[...,2]
                    img_end = cv2.imread(os.path.join(dirpath,sub_files[end + jj]))
                    img_end = img_end.astype(np.float32) / 255
                    if resize == True:
                        img_end = cv2.resize(img_end,(width // 4,height // 4),interpolation=cv2.INTER_LINEAR )
                    img_end = 0.11 * img_end[...,0] + 0.59 * img_end[...,1] + 0.3 * img_end[...,2]
                    imgs.append(img_end)
                    imgs.insert(0,img_start)
                #! reduce the number of spikes to / num_omit 
                imgs = imgs[::num_omit]
                # todo from zjy
                spike = SimulationSimple_Video(imgs)
                noise = Inherent_Noise_fast_torch(spike.shape[0],  H=spike.shape[1], W=spike.shape[2])
                spike = torch.bitwise_or(spike.permute((1,2,0)), noise)
                spike = spike.permute((2, 0, 1))
                # todo from csy
                # spike = v2s_interface(imgs,threshold=2)
                
                # save spike
                print(spike.shape)
                SpikeToRaw(os.path.join(output_folder, str((num_blur_raw - 1 + (opt.overlap_len + 1) // (num_inter + 1)) * ((i + 1) // num_blur - 1) + num_blur_raw // 2).zfill(str_len) + ".dat"), spike)
                print(f"Generating spikes from {sub_files[start - spike_add]} to {sub_files[end + spike_add]}")
                print(f"Blur area ranges from {sub_files[start]} to {sub_files[end]}")
                imgs = []
                bais += opt.overlap_len
                start = i + bais + 1