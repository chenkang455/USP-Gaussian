import cv2
import torch
import numpy as np
import imageio
# from moviepy.editor import ImageSequenceClip
import os


def save_gif(image_list, gif_path = 'test', duration = 2,RGB = True,nor = False):
    imgs = []
    os.makedirs('Video',exist_ok = True)
    with imageio.get_writer(os.path.join('Video',gif_path + '.gif'), mode='I',duration = 1000 * duration / len(image_list),loop=0) as writer:
        for i in range(len(image_list)):
            img = normal_img(image_list[i],RGB,nor)
            writer.append_data(img)

def save_video(image_list,path = 'test',duration = 2,RGB = True,nor = False):
    os.makedirs('Video',exist_ok = True)
    img_size = (image_list[0].shape[1], image_list[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videowriter = cv2.VideoWriter(os.path.join('Video',path + '.avi'), fourcc, len(image_list) / duration, img_size)
    for i in range(len(image_list)):
        img = normal_img(image_list[i],RGB,nor)
        videowriter.write(img)


def normal_img(img,RGB = True,nor = True):
    if nor:
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
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
    cv2.imwrite(os.path.join(os.getcwd(),'imgs',path),img)

def make_folder(path):
    os.makedirs(path,exist_ok = True)

def video_to_spike(
    sourefolder=None,
    imgs = None, 
    savefolder_debug=None, 
    threshold=5.0,
    init_noise=True,
    format="png",
    ):
    """
        函数说明
        :param 参数名: 参数说明
        :return: 返回值名: 返回值说明
    """
    if sourefolder != None:
        filelist = sorted(os.listdir(sourefolder))
        datas = [fn for fn in filelist if fn.endswith(format)]
        
        T = len(datas)
        
        frame0 = cv2.imread(os.path.join(sourefolder, datas[0]))
        H, W, C = frame0.shape

        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

        spikematrix = np.zeros([T, H, W], np.uint8)

        if init_noise:
            integral = np.random.random(size=([H,W])) * threshold
        else:
            integral = np.random.zeros(size=([H,W]))
        
        Thr = np.ones_like(integral).astype(np.float32) * threshold

        for t in range(0, T):
            frame = cv2.imread(os.path.join(sourefolder, datas[t]))
            if C > 1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray / 255.0
            integral += gray
            fire = (integral - Thr) >= 0
            fire_pos = fire.nonzero()
            
            integral[fire_pos] -= threshold
            spikematrix[t][fire_pos] = 1

        if savefolder_debug:
            np.save(os.path.join(savefolder_debug, "spike_debug.npy"), spikematrix)
    elif imgs != None:
        frame0 = imgs[0]
        H, W, C = frame0.shape
        T = len(imgs)
        spikematrix = np.zeros([T, H, W], np.uint8)

        if init_noise:
            integral = np.random.random(size=([H,W])) * threshold
        else:
            integral = np.random.zeros(size=([H,W]))
        
        Thr = np.ones_like(integral).astype(np.float32) * threshold

        for t in range(0, T):
            frame = imgs[t]
            if C > 1:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray / 255.0
            integral += gray
            fire = (integral - Thr) >= 0
            fire_pos = fire.nonzero()
            
            integral[fire_pos] -= threshold
            spikematrix[t][fire_pos] = 1
    return spikematrix


def load_vidar_dat(filename, left_up=(0, 0), window=None, frame_cnt = None, height = 800,width = 800):
    if isinstance(filename, str):
        array = np.fromfile(filename, dtype=np.uint8)
    elif isinstance(filename, (list, tuple)):
        l = []
        for name in filename:
            a = np.fromfile(name, dtype=np.uint8)
            l.append(a)
        array = np.concatenate(l)
    else:
        raise NotImplementedError
    if window == None:
        window = (height - left_up[0], width - left_up[0])
    len_per_frame = height * width // 8
    framecnt = frame_cnt if frame_cnt != None else len(array) // len_per_frame
    spikes = []
    for i in range(framecnt):
        compr_frame = array[i * len_per_frame: (i + 1) * len_per_frame]
        blist = []
        for b in range(8):
            blist.append(np.right_shift(np.bitwise_and(compr_frame, np.left_shift(1, b)), b))
        frame_ = np.stack(blist).transpose()
        frame_ = np.flipud(frame_.reshape((height, width), order='C'))
        if window is not None:
            spk = frame_[left_up[0]:left_up[0] + window[0], left_up[1]:left_up[1] + window[1]]
        else:
            spk = frame_
        spk = torch.from_numpy(spk.copy().astype(np.float32))
        spikes.append(spk)
    return torch.stack(spikes,dim = 0) 