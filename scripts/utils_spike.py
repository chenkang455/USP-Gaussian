from tqdm import trange
import torch
import os
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms


def Inherent_Noise_fast_torch(T, mu=140.0, std=50.0, H=250, W=400):
    """
    Generate Gaussian distributed inherent noise.
    args:
        - T: Simulation time length
        - mu: Mean
        - std: Standard deviation
    return:
        - The array records the location of noise
    """
    shape = [H, W, T]
    size = H * W * T
    gaussian = torch.normal(mu, std, size=(size,))
    noise = torch.zeros(size, dtype=torch.int16)
    keys = torch.cumsum(gaussian, dim=0)
    keys = keys[keys<size-1].long()
    noise[keys] = 1

    return torch.reshape(noise, shape)


def check_folder(folderpath):
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)


def SimulationSimple(I, T=50):
    # Initialize Sensor Parameters 
    # ! bug Vth = 2 (1 in truth)
    Vth = 2.0
    Eta = 10**(-13)*1.09
    Lambda = 10**(-4)*1.83
    Cpd = 10.0**(-15)*15
    CLK = 10.0**(6)*10
    delta_t = 2 / CLK
    K = delta_t * Eta / (Lambda * Cpd)
    # print(K)
    # vol = 0
    H, W = I.shape
    vol = torch.zeros(size=(H,W))
    syn_rec = torch.zeros(size=(T,H,W), dtype=torch.int16)

    for n in trange(T):
        g = torch.rand(size=(H,W)) if n == 0 else I
        # g = I
        # print(noise_1.shape)

        # print(g_all[:, 100,100])
        # flag, vol = inner_clock(g * K, Vth, vol)
        # print(torch.max(flag), torch.max(vol))
        # syn_rec[n] = flag
        vol += g * K * 250
        vol_to_Thr = vol > Vth
        syn_rec[n][vol_to_Thr] = 1
        vol[vol_to_Thr] = vol[vol_to_Thr] % Vth
        # for t in range(250):
        #     # print(g.shape, noise_1[t].shape)
        #     vol += g * K
        #     vol_to_Thr = vol >= Vth
        #     syn_rec[n][vol_to_Thr] = 1
        #     vol[vol_to_Thr] = 0
    return syn_rec

def v2s_interface(imgs, savefolder=None, threshold=5.0):
    T = len(imgs)
    # frame0[..., 0:2] = 0
    # cv2.imshow('red', frame0)
    # cv2.waitKey(0)
    H, W = imgs[0].shape
    # exit(0)

    spikematrix = np.zeros([T, H, W], np.uint8)
    # integral = np.array(frame0gray).astype(np.float32)
    integral = np.random.random(size=([H,W])) * threshold
    Thr = np.ones_like(integral).astype(np.float32) * threshold

    for t in range(0, T):
        # print('spike frame %s' % datas[t])
        frame = imgs[t]
        # gray = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        gray = frame
        integral += gray
        fire = (integral - Thr) >= 0
        fire_pos = fire.nonzero()
        integral[fire_pos] -= threshold
        # integral[fire_pos] = 0.0
        spikematrix[t][fire_pos] = 1
    return spikematrix

def SimulationSimple_Video(I):
    # Initialize Sensor Parameters 
    Vth = 4.0
    Eta = 10**(-13)*1.09
    Lambda = 10**(-4)*1.83
    Cpd = 10.0**(-15)*15
    CLK = 10.0**(6)*10
    delta_t = 2 / CLK
    K = delta_t * Eta / (Lambda * Cpd)
    # print(K)
    # vol = 0
    T = len(I)
    H, W = I[0].shape
    vol = torch.zeros(size=(H,W))
    syn_rec = torch.zeros(size=(T+1,H,W), dtype=torch.int16)

    for n in range(T+1):
        g = torch.rand(size=(H,W)) if n == 0 else I[n-1]
        # g = I
        # print(noise_1.shape)

        # print(g_all[:, 100,100])
        # flag, vol = inner_clock(g * K, Vth, vol)
        # print(torch.max(flag), torch.max(vol))
        # syn_rec[n] = flag
        vol += g * K * 250
        vol_to_Thr = vol > Vth
        syn_rec[n][vol_to_Thr] = 1
        vol[vol_to_Thr] = vol[vol_to_Thr] % Vth
        # for t in range(250):
        #     # print(g.shape, noise_1[t].shape)
        #     vol += g * K
        #     vol_to_Thr = vol >= Vth
        #     syn_rec[n][vol_to_Thr] = 1
        #     vol[vol_to_Thr] = 0
    return syn_rec[1:]


def SpikeToRaw(save_path, SpikeSeq, filpud=True, delete_if_exists=True):
    """
        save spike sequence to .dat file
        save_path: full saving path (string)
        SpikeSeq: Numpy array (T x H x W)
        Rui Zhao
    """
    if delete_if_exists:
        if os.path.exists(save_path):
            os.remove(save_path)

    sfn, h, w = SpikeSeq.shape
    remainder = int((h * w) % 8)
    # assert (h * w) % 8 == 0
    base = np.power(2, np.linspace(0, 7, 8))
    fid = open(save_path, 'ab')
    for img_id in range(sfn):
        if filpud:
            # 模拟相机的倒像
            spike = np.flipud(SpikeSeq[img_id, :, :])
        else:
            spike = SpikeSeq[img_id, :, :]
        # numpy按自动按行排，数据也是按行存的
        # spike = spike.flatten()
        if remainder == 0:
            spike = spike.flatten()
        else:
            spike = np.concatenate([spike.flatten(), np.array([0]*(8-remainder))])
        spike = spike.reshape([int(h*w/8), 8])
        data = spike * base
        data = np.sum(data, axis=1).astype(np.uint8)
        fid.write(data.tobytes())
    fid.close()
    return

def load_vidar_dat(filename, frame_cnt=None, width=640, height=480, reverse_spike=True):
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

    return np.array(spikes).astype(np.float32)

if __name__ == "__main__":
    # main()
    imgs = np.ones((100,300,300))
    print(v2s_interface(imgs).shape)
    # spike = load_vidar_dat('I:\\Datasets\\REDS\\train\\train_spike\\000\\00000007.dat', width=260, height=640)
    # cv2.imwrite('test_spike_load.png', np.mean(spike, axis=0)[..., None] * 255)
            