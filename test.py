import os
import sys
import math
import torch
import random
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.fft import fft, fftfreq, fftshift
from datetime import datetime

import numpy as np
from scipy.integrate import dblquad
from scipy import integrate
import pickle
from scipy.special import erf
from scipy.special import i0e, i0
from tqdm import tqdm
import pickle
import colorsys

# debug
from PIL import Image

from model_components import maskCNNModel, classificationHybridModel

np.random.seed(10)
random.seed(10)

for sf in range(8, 11):
    data_dir = f'/path/to/NeLoRa_Dataset/{sf}/'
    name = f'sf{sf}_model'
    namelist=['HFFT','LoRaPhy','NeLoRa']
    bw = 125e3
    fs = 1e6
    n_classes = 2 ** sf
    nsamp = int(n_classes * fs / bw)
    snrrange = list(range(-10,-30,-1))
    normalization = False

    mask_CNN = maskCNNModel(conv_dim_lstm=nsamp, lstm_dim=400, fc1_dim=600, freq_size=n_classes)
    C_XtoY = classificationHybridModel(conv_dim_in=2, conv_dim_out=n_classes, conv_dim_lstm=nsamp)

    state_dict = torch.load(f'checkpoint/sf{sf}/100000_maskCNN.pkl', map_location=lambda storage, loc: storage)
    mask_CNN.load_state_dict(state_dict, strict=True)
    state_dict = torch.load(f'checkpoint/sf{sf}/100000_C_XtoY.pkl', map_location=lambda storage, loc: storage)
    C_XtoY.load_state_dict(state_dict, strict=True)
    mask_CNN.cuda() 
    C_XtoY.cuda()
    torch.no_grad()
    mask_CNN.eval()
    C_XtoY.eval()


    t = np.linspace(0, nsamp / fs, nsamp+1)[:-1]
    chirpI1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
    chirpQ1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
    downchirp = chirpI1 + 1j * chirpQ1


    def decode_loraphy_hfft(dataX, n_classes, downchirp):
        upsampling = 100
        chirp_data = dataX * downchirp
        fft_raw = fft(chirp_data, len(chirp_data) * upsampling)
        target_nfft = n_classes * upsampling

        cut0 = np.concatenate((fft_raw[:target_nfft//2],fft_raw[-target_nfft//2:]))
        est_hfft = round(np.argmax(abs(cut0)) / upsampling) % n_classes

        cut1 = np.array(fft_raw[:target_nfft])
        cut2 = np.array(fft_raw[-target_nfft:])
        est_loraphy = round(np.argmax(abs(cut1)+abs(cut2)) / upsampling) % n_classes
        return est_loraphy, est_hfft

    def decode_model(dataX):
        stft_window = n_classes // 2
        stft_overlap = stft_window // 2
        x = torch.stft(input=torch.tensor(dataX, dtype=torch.cfloat), n_fft=nsamp,
                       hop_length=stft_overlap, win_length=stft_window, pad_mode='constant',
                       return_complex=True)
        y = torch.concat((x[-n_classes // 2:, :], x[0:n_classes // 2, :]), axis=0)
        y = torch.stack((y.real, y.imag), 0)
        y = y.unsqueeze(0)
        y = y.cuda()
        mask_Y = mask_CNN(y)
        outputs = C_XtoY(mask_Y)
        _, est = torch.max(outputs, 1)
        est = est.cpu().item()
        return est




    acc = np.zeros((len(namelist), len(snrrange),n_classes,))
    cnt = np.zeros((len(snrrange),n_classes,))
    files = [[] for i in range(n_classes)]
    for subfolder in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, subfolder)):
            truth_idx = int(filename.split('_')[1])
            files[truth_idx].append(os.path.join(data_dir, subfolder,filename))
    minl = min([len(x) for x in files])
    datas = []
    for truth_idx, filelist in enumerate(files):
        for filepath in filelist:
            with open(filepath, 'rb') as fid:
                chirp_raw = np.fromfile(fid, np.complex64, nsamp)
                assert len(chirp_raw) == nsamp
                datas.append((chirp_raw, truth_idx))

    random.shuffle(datas)
    for snridx, snr in enumerate(snrrange):
        print(f"FILES {n_classes}*{minl} START EVALUATION {snr}\n")

        for dataY, truth_idx in tqdm(datas):

                amp = math.pow(0.1, snr / 20) * np.mean(np.abs(dataY))
                noise = amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp)
                dataX = dataY + noise
                if normalization: dataX = dataX / np.mean(np.abs(dataX))

                est_loraphy, est_hfft = decode_loraphy_hfft(dataX, n_classes, downchirp)
                est_model = decode_model(dataX)
                acc[namelist.index('NeLoRa')][snridx][truth_idx] += (est_model == truth_idx)
                acc[namelist.index('HFFT')][snridx][truth_idx] += (est_hfft == truth_idx)
                acc[namelist.index('LoRaPhy')][snridx][truth_idx] += (est_loraphy == truth_idx)
                cnt[snridx][truth_idx] += 1

                s = f"SNR: {snr} ACC: {np.sum(acc[namelist.index('NeLoRa')][snridx])/np.sum(cnt[snridx]):.3f} est_model:{est_model} est_loraphy:{est_loraphy} tru:{truth_idx}"
                sys.stdout.write("\x1b[s\x1b[1A\r" + s +  " " * (os.get_terminal_size().columns-len(s)) + "\x1b[u")

    print("\nFINISH EVALUATION")
    with open(name+'.pkl', 'wb') as g: pickle.dump((acc, cnt),g)



# Plot
    plt.rcParams['font.size'] = 15
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['lines.markersize'] = 12
    plt.figure(figsize=(8, 6)) 
    color = ['#FF1F5B', '#00CD6C', '#009ADE', '#AF58BA', '#FFC61E', '#F28522']
    linestyle=['-', '--', '-.', ':', '-', '--']
    marker=['x','.','^','v','*','+']
    plt.axhline(y=0.9,linestyle='--',color='black')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')

    for pidx, label in enumerate(namelist):
        div_result = np.where(cnt != 0, acc[pidx] / cnt, np.nan)
        res = np.nanmean(div_result, axis=1)
        plt.plot(snrrange, res, color=color[pidx], linestyle=linestyle[pidx], marker=marker[pidx], label=label)

    plt.legend()
    plt.savefig(name + '.pdf')
    plt.clf()


