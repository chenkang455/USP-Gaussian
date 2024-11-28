import numpy as np
def load_vidar_dat(filename, frame_cnt=None, width=600, height=400, reverse_spike=True):
    '''
    output: <class 'numpy.ndarray'> (frame_cnt, height, width) {0，1} float32
    '''
    array = np.fromfile(filename, dtype=np.uint8)

    len_per_frame = height * width // 8
    framecnt = frame_cnt if frame_cnt != None else len(array) // len_per_frame

    spikes = []
    for i in range(framecnt):
        compr_frame = array[i * len_per_frame: (i + 1) * len_per_frame]
        blist = []
        for b in range(8):
            blist.append(np.right_shift(np.bitwise_and(
                compr_frame, np.left_shift(1, b)), b))

        frame_ = np.stack(blist).transpose()
        frame_ = frame_.reshape((height, width), order='C')
        if reverse_spike:
            frame_ = np.flipud(frame_)
        spikes.append(frame_)

    return np.array(spikes)


import logging
# log info
def setup_logging(log_file):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='w')  # 使用'w'模式打开文件
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


import cv2
import torch
import numpy as np
import imageio
import os
import torch.nn as nn
import random
# Save Network 
def save_network(network, save_path):
    if isinstance(network, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

def set_random_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


def middleTFI(spike, middle, window):
    B, C, H, W = spike.shape
    lindex, rindex = torch.zeros([B,1, H, W]), torch.zeros([B,1,H, W])
    l, r = middle+1, middle+1
    for r in range(middle+1, middle + window+1):
        l = l - 1
        if l>=0:
            newpos = spike[:,l:l+1, :, :]*(1 - torch.sign(lindex)) 
            distance = l*newpos
            lindex += distance
        if r<C:
            newpos = spike[:,r:r+1, :, :]*(1 - torch.sign(rindex))
            distance = r*newpos
            rindex += distance
    rindex[rindex==0] = window+middle
    lindex[lindex==0] = middle-window
    interval = rindex - lindex
    tfi = 1.0 / interval
    return tfi

def save_opt(opt,opt_path):
    with open(opt_path, 'w') as f:
        for key, value in vars(opt).items():
            f.write(f"{key}: {value}\n")

def save_gif(image_list, gif_path = 'test', duration = 2,RGB = True,nor = False):
    imgs = []
    with imageio.get_writer(os.path.join(gif_path + '.gif'), mode='I',duration = 1000 * duration / len(image_list),loop=0) as writer:
        for i in range(len(image_list)):
            img = normal_img(image_list[i],RGB,nor)
            writer.append_data(img)

def save_video(image_list,path = 'test',duration = 2,RGB = True,nor = False):
    os.makedirs('Video',exist_ok = True)
    imgs = []
    for i in range(len(image_list)):
        img = normal_img(image_list[i],RGB,nor)
        imgs.append(img)
    imageio.mimwrite(os.path.join('Video',path + '.mp4'), imgs, fps=30, quality=8)


def normal_img(img,RGB = True,nor = True):
    if nor:
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
    if (img.shape[0] == 3 or img.shape[0] == 1) and isinstance(img,torch.Tensor):
        img = img.permute(1,2,0)
    if isinstance(img,torch.Tensor):
        img = np.array(img.detach().cpu())
    if len(img.shape) == 2:
        img = img[...,None]
    if img.shape[-1] == 1:
        img = np.repeat(img,3,axis = -1)
    img = img.astype(np.uint8)
    if RGB == False:
        img = img[...,::-1]
    return img

def save_img(path = 'test.png',img = None,nor = True):
    if nor:
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
    if isinstance(img,torch.Tensor):
        img = np.array(img.detach().cpu())
    img = img.astype(np.uint8)
    cv2.imwrite(path,img)

def make_folder(path):
    os.makedirs(path,exist_ok = True)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count




def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
    
    
def generate_labels(file_name):
    num_part = file_name.split('/')[-1]
    non_num_part = file_name.replace(num_part, '')
    num = int(num_part)
    labels = [non_num_part + str(num + 2 * i).zfill(len(num_part)) + '.png' for i in range(-3, 4)]
    return labels
