import numpy as np
import csv
from base64 import b64decode
from os.path import join
import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time



def calculate_ftm_distance(ftm_measurement):
    delay = ((ftm_measurement[3] - ftm_measurement[0]) - (ftm_measurement[2] - ftm_measurement[1])) / 2e12
    dist = 2.998e8 * delay
    return dist

def print_ftm_counts(path):
    mac_counts={}
    with open(join(path, 'ftm.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            mac = row[1]
            if mac not in mac_counts.keys():
                mac_counts[mac] = 0
            else:
                mac_counts[mac] = mac_counts[mac] + 1

    print("FTMs received per MAC:")
    for m in mac_counts:
        print(f"{m}: {mac_counts[m]}")

def shift_interp(h, shift):
    h[64:] *= np.exp(1.0j*shift)
    h_up = h[2]
    h_down = h[-2]
    h[0] = h_up/2 + h_down/2
    h[1] = h_up*3/4 + h_down*1/4
    h[-1] = h_up*1/4 + h_down*3/4
    h /= h[0,None]
    return h

def l1_err(h):
    return np.sum(np.abs(np.fft.ifft(h)))

def optimal_interp_phase(h, iters=4, density=8):
    window = 2*np.pi
    center = 0
    #shifts_init = np.arange(0, 2*np.pi, 2*np.pi/density)
    for ii in range(iters):
        shifts = np.arange(-window/2 + center, window/2 + center, window/density)
        errs = np.zeros_like(shifts)
        for k in range(len(shifts)):
            errs[k] = l1_err(shift_interp(np.copy(h), shifts[k]))
            #plt.plot(shifts, errs, marker='.', linestyle='none')
        window /= density/2
        center = shifts[np.argmin(errs)]
        #print(center)
    return shift_interp(np.copy(h), center)

IDFT_MAT = np.exp(2.0j*np.pi*np.arange(128)[:,None]*np.arange(128)[None,:]/128)/np.sqrt(128)
EXCLUDED_SUBC = np.isin(np.arange(128),[0,1,59,60,61,62,63,64,65,66,67,68,69,127])
ZIDX = np.logical_not(EXCLUDED_SUBC)
def optimal_interp_edges(h):
    iters = 20
    eps = 1e-1
    h_ = np.copy(h)

    ii = 0
    fail = 0
    loss = np.inf
    # hs = []
    # Hs = []
    # losses = []
    while(ii < iters):
        sgn_ax_ = np.fft.ifft(h_,norm='ortho')
        sgn_ax_ = sgn_ax_ / (np.abs(sgn_ax_) + 1e-9)
        grad_ = np.fft.fft(sgn_ax_,norm='ortho')
        grad_[ZIDX] = 0.0
        loss_new = np.sum(np.abs(np.fft.ifft(h_ - eps*grad_)))
        if(loss_new <= loss):
            eps *= 1.1
            ii += 1
            h_ -= eps*grad_
            loss = loss_new
            fail = 0
            # losses.append(loss)
            # hs.append(np.fft.ifft(np.copy(h_)))
            # Hs.append(np.copy(h_))
        else:
            eps *= 0.5
            fail = fail + 1
            if fail > 10:
                break
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,3,1)
    # ax2 = fig.add_subplot(1,3,2)
    # ax3 = fig.add_subplot(1,3,3)
    # def animate(idx):
    #     ax1.clear()
    #     ax2.clear()
    #     ax3.clear()
    #     ax1.plot(np.abs(np.fft.fftshift(Hs[idx])))
    #     ax2.plot(np.abs(np.fft.fftshift(hs[idx])))
    #     ax3.plot(losses)
    #     ax3.axvline(idx, color='k')
    #     ax1.set_title("CSI")
    #     ax2.set_title("CIR")
    #     ax3.set_title("Loss")

    #     ax1.set_ylim([0,np.max(np.abs(h_))*1.2])
    # anim = ani.FuncAnimation(fig, animate, frames=range(len(Hs)))
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.ylim([0,np.max(np.abs(h_))*1.2])
    # plt.plot(np.abs(np.fft.fftshift(h)))
    # plt.title("Initial")
    # plt.subplot(2,2,3)
    # plt.plot(np.abs(np.fft.fftshift(np.fft.ifft(h))))

    # plt.subplot(2,2,2)
    # plt.ylim([0,np.max(np.abs(h_))*1.2])
    # plt.plot(np.abs(np.fft.fftshift(h_)))
    # plt.title("Final")
    # plt.subplot(2,2,4)
    # plt.plot(np.abs(np.fft.fftshift(np.fft.ifft(h_))))

    
    # plt.show()
    return h_

def normalize_csi(h, stitching='l1_edges'):
    h = h.conj()
    h_interp = np.zeros_like(h)
    if stitching == 'phase_only':
        for hi in range(h.shape[0]):
            h_interp[hi,:] = optimal_interp_phase(h[hi,:])
    elif stitching == 'l1_edges':
        for hi in tqdm.trange(h.shape[0]):
            h_ = h[hi,:]
            h_[64:] *= np.exp(1.0j*np.pi/2)
            h_ /= np.sum(np.abs(h_))
            h_interp[hi,:] = optimal_interp_edges(h_)
    elif stitching == 'constant':
        for hi in range(h.shape[0]):
            h_ = h[hi,:]
            h_[64:] *= np.exp(1.0j*np.pi/2)
            h_ /= np.sum(np.abs(h_))
            h_interp[hi,:] = h_
    elif stitching == 'phase_slope':

        for hi in range(h.shape[0]):
            h_ = h[hi,:]
            h_[64:] *= np.exp(1.0j*np.pi/2)
            h_ /= np.sum(np.abs(h_))
            ang = np.fft.fftshift(np.unwrap(np.angle(h_)))
            idx = np.arange(-64,64)
            line = np.linalg.lstsq(np.vstack([idx,np.ones(128)]).T, ang)[0]
            exp = np.fft.fftshift(np.exp(1.0j*(idx * line[0] + line[1])))
            mean_abs = np.mean(np.abs(h_[ZIDX]))
            h_[EXCLUDED_SUBC] = exp[EXCLUDED_SUBC]*mean_abs
            h_interp[hi,:] = h_
        pass
    else:
        assert False, "invalid mode"

    return h_interp
        
def generate_ftm_csi_pairs(path, target_mac):
    h_ms = []
    h_stamp = []
    h_seq = []
    h_H = []
    with open(join(path, 'csi.csv'),'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[2] == target_mac:
                h_ms.append(int(row[0]))
                h_stamp.append(int(row[1]))
                h_seq.append(int(row[3]))
                h = np.frombuffer(b64decode(row[4]), dtype=np.int8)
                h = h[::2] + 1.0j * h[1::2]
                h_H.append(h)
    t_ms = []
    t_dlog = []
    t_t0 = []
    t_t1 = []
    t_t2 = []
    t_t3 = []
    t_rssi = []
    
    with open(join(path, 'ftm.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[3] == target_mac:
                t_dlog.append(int(row[4]))
                t_ms.append(int(row[1]))
                t_t0.append(int(row[6]))
                t_t1.append(int(row[7]))
                t_t2.append(int(row[8]))
                t_t3.append(int(row[9]))
                t_rssi.append(int(row[5]))
    h_stamp = np.asarray(h_stamp)
    t_stamp = np.asarray(t_t1)//1000

    # associate each CSI with the corresponding timestamp
    f_ms = []
    f_rssi = []
    f_h = []
    f_t = []
    for ii in range(len(t_stamp)):
        delta = np.abs(t_stamp[ii] - h_stamp)
        dmin = np.min(delta)
        if dmin > 2:
            print(f"Couldn't find good match for FTM at {t_stamp[ii]}, skipping...")
            continue
        h_idx = np.argmin(delta)
        #f_ms.append(h_ms[h_idx])
        f_ms.append(t_ms[ii])
        f_h.append(h_H[h_idx])
        f_t.append([t_t0[ii],t_t1[ii],t_t2[ii],t_t3[ii]])
        f_rssi.append(t_rssi[ii])
    return f_ms, f_t, f_h, f_rssi
