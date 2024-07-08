import os
import sys
import math
import torch
import random
import numpy as np
from scipy.signal import chirp
from scipy.fft import fft 
from torch.utils.data import DataLoader, Subset,TensorDataset, random_split
from tqdm import tqdm

from model_components import maskCNNModel, classificationHybridModel

np.random.seed(10)
random.seed(10)

sf = 8
data_dir = f'/data/djl/NeLoRa_Dataset/NeLoRa_Dataset/{sf}/'
name = f'sf{sf}_model'
save_ckpt_dir = 'ckpt'
if not os.path.exists(save_ckpt_dir): os.mkdir(save_ckpt_dir)
bw = 125e3
fs = 1e6
n_classes = 2 ** sf
nsamp = int(n_classes * fs / bw)
snrrange = list(range(-30, 1))
test_snr = -22
batch_size = 16
scaling_for_imaging_loss = 1

mask_CNN = maskCNNModel(conv_dim_lstm=nsamp, lstm_dim=400, fc1_dim=600, freq_size=n_classes)
C_XtoY = classificationHybridModel(conv_dim_in=2, conv_dim_out=n_classes, conv_dim_lstm=nsamp)
mask_CNN.load_state_dict(torch.load(f'checkpoint/sf{sf}/100000_maskCNN.pkl', map_location=lambda storage, loc: storage), strict=True)
C_XtoY.load_state_dict(torch.load(f'checkpoint/sf{sf}/100000_C_XtoY.pkl', map_location=lambda storage, loc: storage), strict=True)
mask_CNN.cuda() 
C_XtoY.cuda()


t = np.linspace(0, nsamp / fs, nsamp+1)[:-1]
chirpI1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
chirpQ1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
downchirp = chirpI1 + 1j * chirpQ1


def decode_loraphy(dataX, n_classes, downchirp):
    upsampling = 100
    chirp_data = dataX * downchirp
    fft_raw = fft(chirp_data, len(chirp_data) * upsampling)
    target_nfft = n_classes * upsampling

    cut1 = np.array(fft_raw[:target_nfft])
    cut2 = np.array(fft_raw[-target_nfft:])
    return round(np.argmax(abs(cut1)+abs(cut2)) / upsampling) % n_classes

def decode_model(dataX):
    stft_window = n_classes // 2
    stft_overlap = stft_window // 2
    x = torch.stft(input=dataX, n_fft=nsamp,
                   hop_length=stft_overlap, win_length=stft_window, pad_mode='constant',
                   return_complex=True)
    y = torch.concat((x[:, -n_classes // 2:, :], x[:, 0:n_classes // 2, :]), axis=1)
    y = torch.stack((y.real, y.imag), 1).cuda()
    mask_Y = mask_CNN(y)
    outputs = C_XtoY(mask_Y)
    return mask_Y, outputs, y




files = [[] for i in range(n_classes)]
for subfolder in os.listdir(data_dir):
    for filename in os.listdir(os.path.join(data_dir, subfolder)):
        truth_idx = int(filename.split('_')[1])
        files[truth_idx].append(os.path.join(data_dir, subfolder,filename))
minl = min([len(x) for x in files])
datax = []
datay = []
for truth_idx, filelist in enumerate(files):
    for filepath in filelist:
        with open(filepath, 'rb') as fid:
            chirp_raw = np.fromfile(fid, np.complex64, nsamp)
            assert len(chirp_raw) == nsamp
            if decode_loraphy(chirp_raw, n_classes, downchirp) == truth_idx:
                datax.append(torch.tensor(chirp_raw, dtype=torch.cfloat))
                datay.append(truth_idx)

data = TensorDataset(torch.stack(datax),torch.tensor(datay, dtype=torch.long)) 
train_data, test_data = random_split(data, [len(data) - len(data)//10, len(data) // 10])
training_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
print(f'Len Training {len(training_loader)} Testing {len(test_loader)}')

g_params = list(mask_CNN.parameters()) + list(C_XtoY.parameters())
g_optimizer = torch.optim.Adam(g_params, 0.0001, [0.5, 0.999])
loss_spec = torch.nn.MSELoss(reduction='mean')
loss_class = torch.nn.CrossEntropyLoss()
while True:
    for iteration, data_train in enumerate(training_loader):
        dataY, truth_idx = data_train

        amp = math.pow(0.1, random.choice(snrrange) / 20) * torch.mean(torch.abs(dataY))
        noise = (amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp)).type(torch.cfloat)
        dataX = dataY + noise

        fake_Y_spectrum, labels_X_estimated, images_X_spectrum = decode_model(dataX)

        g_y_pix_loss = loss_spec(fake_Y_spectrum, images_X_spectrum)
        g_y_class_loss = loss_class(labels_X_estimated, truth_idx.cuda())
        g_optimizer.zero_grad()
        G_Y_loss = scaling_for_imaging_loss * g_y_pix_loss + g_y_class_loss
        G_Y_loss.backward()
        g_optimizer.step()

        if iteration % 100 == 0:
            torch.save(mask_CNN.state_dict(), os.path.join(save_ckpt_dir, str(iteration) + '_maskCNN.pkl'))
            torch.save(C_XtoY.state_dict(), os.path.join(save_ckpt_dir, str(iteration) + '_C_XtoY.pkl'))
            with torch.no_grad():
                mask_CNN.eval() 
                C_XtoY.eval()

                acc_model = 0
                for vdata in tqdm(test_loader):
                    dataYv, truthv = vdata

                    amp = math.pow(0.1, test_snr / 20) * torch.mean(torch.abs(dataY))
                    noisev = (amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp)).type(torch.cfloat)
                    dataXv = dataYv + noisev

                    _, estv, _ = decode_model(dataXv)
                    acc_model += torch.sum(torch.max(estv,1)[1].cpu() == truthv)

                print('SNR: %d TEST ACCMODEL: %.3f' % (test_snr, acc_model / (len(test_loader) * batch_size)))
                mask_CNN.train() 
                C_XtoY.train()


