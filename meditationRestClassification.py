# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:16:01 2017

@author: Amarantine
"""

#%% import relevant toolboxes
#clear all?
get_ipython().magic('reset -sf')
from IPython import get_ipython

import mne # a toolbox to use work with bio-signals
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import gc
import pywt


#from autoreject import LocalAutoRejectCV



#%% change working directory and set parameters

path="E:\Biomedical.master\Data\par071_1" #TODO : go back to input design
os.chdir(path)
montage=mne.channels.read_montage('standard_1020')
layout=mne.channels.read_layout('EEG1005')
layout2=mne.channels.read_layout('EGI256')
blockNum=3
samp_freq=float(1000)
resamp_freq=float(1000)

#%% load data files

fname=os.path.join(path, "meditation.vhdr") #TODO: go back to input design
# load the data from matlab interdace: sequence of stimuli, oddball stimuli, gc.collect()

#%%
meditation= mne.io.read_raw_brainvision (fname,preload=True)

gc.collect() 

#%% add event list to the brainvision raw object
exported_events=np.load("exported_events.npy")  # have manipulated \
#brainvision.py to save events as exported_events.npy from  read marker files
exported_events[:,0]=exported_events[:,0]/(samp_freq/resamp_freq)     
events=exported_events.tolist()
meditation.add_events(events)
#events: 1: baseline start, 3: mditation starts 
#meditation duration__track1 5:36 mins__track2 5:14 mins
gc.collect()


#%%
#meditation.notch_filter(np.arange(60, 302, 120), filter_length='auto')
meditation.resample(sfreq=resamp_freq) 
meditation.filter(2,30,None,method='iir') 
meditation.info['lowpass']=30

 #%%annotations 
##### saving annotation step
#onset=[] 
#duration=[]  
###########
#badchannels=['O1','O2','P8','Cz','Fpz'] #
#meditation_annot_params=dict(onset=onset,duration=duration,badchannels=badchannels)
#np.save(os.path.join(path, 'meditation_annot_params.npy'), meditation_annot_params)

##loading annotation step
meditation_annot_params= np.load(os.path.join(path, 'meditation_annot_params.npy'))
onset=meditation_annot_params.item().get('onset')
duration=meditation_annot_params.item().get('duration')
badchannels=meditation_annot_params.item().get('badchannels')
annotations=mne.Annotations(onset,duration,'bad')
meditation.annotations=annotations 

bads=badchannels
meditation.info['bads']=bads


#%% separate blocks 
event_time_baseline=[exported_events[i,0] for i in range(len(exported_events))\
                     if (exported_events[i,2] == 1 or exported_events[i,2] ==3)]
time_endOfExp=exported_events[len(exported_events)-1, 0]+314*resamp_freq 
#end of session(=end of last block) will be definded 5:14mins=314 seconds after last stimulus event is lasbat block
meditation_blocks=[None]*blockNum

# create blocks : a dictionary which its  values are cropped brainvision raw 
# objects
for i in range(blockNum):
#    print "preparing block#", i+1
    meditationReplica = meditation.copy()

    tmin = event_time_baseline[i] / resamp_freq                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    if i < blockNum - 1:                         
        tmax=event_time_baseline[i+1]/resamp_freq
    else:
        tmax=time_endOfExp/resamp_freq  # end of session(=end of last block) \
        #will be definded 10 seconds after last stimulus event is last block                   
    meditation_blocks[i] = meditationReplica.crop(tmin=tmin, tmax=tmax)
    
    del meditationReplica   

gc.collect()    

#%% 



numOfChans=17
numOfBestChans=numOfChans- len(bads) #one channel is stim (marker) channel
numOfFeatures=53
#tmin=0
#tmax=.8
decim=10
#baseline
baseline_startpoint=events[1][0]
baselineEventTimes=500*np.arange(60,dtype=int)+baseline_startpoint
baseline_events=np.zeros([baselineEventTimes.shape[0],3],dtype=np.int32)
for i in range(baseline_events.shape[0]):
    baseline_events[i]=[baselineEventTimes[i],0,19] #the marker for epoching the data is 19
    
baseline_events[:,0]=baseline_events[:,0]/(samp_freq/resamp_freq)     
baseline_events_list=baseline_events.tolist()

X_baseline=np.full([60,numOfBestChans,numOfFeatures],np.nan)
y_baseline=np.zeros([60,1],dtype=np.int8)

#meditation2
meditation2_startpoint=events[4][0]
meditation2EventTimes=500*np.arange(180,dtype=int)+meditation2_startpoint+ 60000 #500=epoch length , 60000: 1 min after start point
meditation2_events=np.zeros([meditation2EventTimes.shape[0],3],dtype=np.int32)
for i in range(meditation2_events.shape[0]):
    meditation2_events[i]=[meditation2EventTimes[i],0,19] #the marker for epoching the data is 19    
    
meditation2_events[:,0]=meditation2_events[:,0]/(samp_freq/resamp_freq)     
meditation2_events_list=meditation2_events.tolist() 


X_meditation2=np.full([180,numOfBestChans,numOfFeatures],np.nan)
y_meditation2=np.ones([180,1],dtype=np.int8)


#%%


print "********preparing data for the baseline *********************"
baseline=meditation_blocks[0] 
baseline.n_times
baseline.load_data()
baseline.info['bads']=bads
baseline.set_montage(montage)
baseline_rerefrenced, _= mne.set_eeg_reference(baseline,[])
baseline.resample(sfreq=resamp_freq)
baseline.add_events(baseline_events_list) 
baseline.drop_channels(baseline.info['bads'])
baseline_data=baseline.get_data()
baseline_Epoch=mne.Epochs(baseline_rerefrenced, baseline_events, tmin=0,
                       tmax=0.5, decim=decim, reject_by_annotation=True)
baseline_Epoch_data= baseline_Epoch.get_data()

for e in range (baseline_Epoch_data.shape[0]):
    for c in range(numOfBestChans):
        cA3,cD3,cD2,cD1=pywt.wavedec(baseline_Epoch_data[e][c],'db1',level=3)
        X_baseline[e][c][0:numOfFeatures]=np.concatenate([cD1,cD2,cD3,cA3],axis=0)
        
X_baseline=X_baseline.reshape([60,numOfBestChans*numOfFeatures])  

#trial_Epoch.drop_bad() 
#%%
print "********preparing data for the meditation2 *********************"
meditation2=meditation_blocks[2] 
meditation2.n_times
meditation2.load_data()
meditation2.info['bads']=bads
meditation2.set_montage(montage)
meditation2_rerefrenced, _= mne.set_eeg_reference(meditation2,[])
meditation2.resample(sfreq=resamp_freq)
meditation2.add_events(meditation2_events_list) 
meditation2.drop_channels(meditation2.info['bads'])
meditation2_data=meditation2.get_data()
meditation2_Epoch=mne.Epochs(meditation2_rerefrenced, meditation2_events, tmin=0,
                       tmax=0.5, decim=decim, reject_by_annotation=True)

meditation2_Epoch_data= meditation2_Epoch.get_data()


for e in range (meditation2_Epoch_data.shape[0]):
    for c in range(numOfBestChans):
        cA3,cD3,cD2,cD1=pywt.wavedec(meditation2_Epoch_data[e][c],'db1',level=3)
        X_meditation2[e][c][0:numOfFeatures]=np.concatenate([cD1,cD2,cD3,cA3],axis=0)
        
X_meditation2=X_meditation2.reshape([180,numOfBestChans*numOfFeatures])        
            
#trial_Epoch.drop_bad() 

X=np.concatenate([X_baseline,X_meditation2],axis=0)
y=np.concatenate([y_baseline,y_meditation2],axis=0)



#%%% visualization

import pywt
import random
from scipy import signal
import matplotlib.pyplot as plt

#plt.close('all')
#
#cmap=random.choice(plt.colormaps())
#plt.figure(1)
#x_wavelet=baseline_Epoch_data[23,8,:]
#
#t = np.linspace(0, 60, 1, endpoint=False)
#widths = np.arange(2, 20)
#cwtmatr = signal.cwt(x_wavelet, signal.ricker, widths)
#plt.imshow(cwtmatr, extent=[0, 60, 1, 20],cmap=cmap, aspect='auto',
#vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
#plt.show()
#
#plt.figure(2)
#x_wavelet_meditation2=meditation2_Epoch_data[23,8,:]
#t = np.linspace(0, 1, 60, endpoint=False)
#widths = np.arange(2, 20)
#cwtmatr = signal.cwt(x_wavelet_meditation2, signal.ricker, widths)
#plt.imshow(cwtmatr, extent=[0, 60, 1, 20], cmap=cmap, aspect='auto',
#vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
#plt.show()
#
##%%
#plt.figure(3)
#freqs = np.arange(7,30, 1)
#n_cycles=2
#from mne.time_frequency import tfr_morlet  # noqa
#power, itc = tfr_morlet(baseline_Epoch, freqs=freqs, n_cycles=n_cycles,
#                        return_itc=True, decim=3, n_jobs=1)
#power.plot([power.ch_names.index('Cz')])
#
#plt.figure(4)
#freqs = np.arange(7,30, 1)
#n_cycles=2
#from mne.time_frequency import tfr_morlet  # noqa
#power, itc = tfr_morlet(meditation2_Epoch, freqs=freqs, n_cycles=n_cycles,
#                        return_itc=True, decim=3, n_jobs=1)
#power.plot([power.ch_names.index('Cz')])


#%%
plt.close('all')
#cmap=random.choice(plt.colormaps())
#cmap='BrBG_r' #'Spectral_r', cubehelix_r, Pastel1,RdYlBu_r, YlGnBu_r,PuBuGn, summer_r, bwr_r, pink_r, ocean, Set2
cmap='Spectral_r'
#channelNumFz=baseline.ch_names.index('Fz')
#med_spectrogram_Fz=plt.figure(1)
#ax1 = plt.subplot(121)
#Pxx, freqs, bins, im = plt.specgram(baseline.get_data()[channelNumFz,15000:25000], NFFT=1024, Fs=100, noverlap=900,cmap=cmap)
#plt.xlabel('Time')
#plt.ylabel('Frequnecy')  
#plt.title('Baseline(Fz)')
#plt.ylim(0,20)
#plt.colorbar()
#plt.show()
#plt.subplot(122, sharex=ax1)
#Pxx, freqs, bins, im = plt.specgram(meditation2.get_data()[channelNumFz,15000:25000], NFFT=1024, Fs=100, noverlap=900,cmap=cmap)
#plt.xlabel('Time')
#plt.ylabel('Frequnecy')  
#plt.title('Meditation(Fz)')
#plt.ylim(0,20)
#plt.colorbar()
#plt.show()
#med_spectrogram_Fz.tight_layout()
#plt.savefig('figFz')
#
#channelNumF3=baseline.ch_names.index('F3')
#med_spectrogram_F3=plt.figure(2)
#ax1 = plt.subplot(121)
#Pxx_baseline, freqs, bins, im = plt.specgram(baseline.get_data()[channelNumF3,15000:25000], NFFT=1024, Fs=100, noverlap=900,cmap=cmap,vmin=-300,vmax=-100)
#plt.xlabel('Time')
#plt.ylabel('Frequnecy')  
#plt.title('Baseline(F3)')
#plt.ylim(0,20)
#plt.colorbar()
#plt.show()
#plt.subplot(122, sharex=ax1)
#Pxx_meditation, freqs, bins, im = plt.specgram(meditation2.get_data()[channelNumF3,15000:25000], NFFT=1024, Fs=100, noverlap=900,cmap=cmap,vmin=-300,vmax=-100)
#plt.xlabel('Time')
#plt.ylabel('Frequnecy')  
#
#plt.title('Meditation(F3)')
#plt.ylim(0,20)
#plt.colorbar()
#plt.show()
#med_spectrogram_F3.tight_layout()
#plt.savefig('figF3')


channelNumF4=baseline.ch_names.index('F4')
med_spectrogram_F4=plt.figure(3)
ax1 = plt.subplot(121)
Pxx_baseline, freqs, bins, im = plt.specgram(baseline.get_data()[channelNumF4,15000:25000], NFFT=1024, Fs=100, noverlap=900,cmap=cmap,vmin=-300,vmax=-100)
plt.xlabel('Time')
plt.ylabel('Frequnecy')  
plt.title('Baseline(F4)')
plt.ylim(0,20)
plt.colorbar()
plt.show()

plt.subplot(122, sharex=ax1)
Pxx_meditation, freqs, bins, im = plt.specgram(meditation2.get_data()[channelNumF4,15000:25000], NFFT=1024, Fs=100, noverlap=900,cmap=cmap,vmin=-300,vmax=-100)
plt.xlabel('Time')
plt.ylabel('Frequnecy')  
plt.title('Meditation(F4)')
plt.ylim(0,20)
plt.colorbar()
plt.show()
med_spectrogram_F4.tight_layout()
plt.savefig('figF4')
#baseline_Epoch.plot_psd_topomap(bands=[(1,15,'b')])
#meditation2_Epoch.plot_psd_topomap(bands=[(1,15,'b')])
#%%



#delta
delta_low=20
delta_high=40
print "low delta", delta_low, "high delta", delta_high
psds_baseline = 10 * np.log10(Pxx_baseline)
psds_mean_delta = np.mean(psds_baseline[delta_low:delta_high])
psds_std_delta = np.std(np.mean(psds_baseline[delta_low:delta_high],axis=0))
print " baseline delta mean", psds_mean_delta, "baseline delta std", psds_std_delta


psds_meditation = 10 * np.log10(Pxx_meditation)
psds_mean_delta = np.mean(psds_meditation[delta_low:delta_high])
psds_std_delta = np.std(np.mean(psds_meditation[delta_low:delta_high],axis=0))
print " meditation delta mean", psds_mean_delta, "meditation delta std", psds_std_delta
print scipy.stats.wilcoxon(np.mean(psds_baseline[delta_low:delta_high],axis=0),np.mean(psds_meditation[delta_low:delta_high],axis=0))

#theta
theta_low=57
theta_high=63
print "low theta", theta_low, "high theta", theta_high
psds_baseline = 10 * np.log10(Pxx_baseline)
psds_mean_theta = np.mean(psds_baseline[theta_low:theta_high])
psds_std_theta = np.std(np.mean(psds_baseline[theta_low:theta_high],axis=0))
print " baseline theta mean", psds_mean_theta, "baseline theta std", psds_std_theta


psds_meditation = 10 * np.log10(Pxx_meditation)
psds_mean_theta = np.mean(psds_meditation[theta_low:theta_high])
psds_std_theta = np.std(np.mean(psds_meditation[theta_low:theta_high],axis=0))
print " meditation theta mean", psds_mean_theta, "meditation theta std", psds_std_theta
print scipy.stats.wilcoxon(np.mean(psds_baseline[theta_low:theta_high],axis=0),np.mean(psds_meditation[theta_low:theta_high],axis=0))

#alpha

alpha_low=80
alpha_high=120
print "low alpha", alpha_low, "high alpha", alpha_high
psds_baseline = 10 * np.log10(Pxx_baseline)
psds_mean_alpha = np.mean(psds_baseline[alpha_low:alpha_high])
psds_std_alpha = np.std(np.mean(psds_baseline[alpha_low:alpha_high],axis=0))
print " baseline alpha mean", psds_mean_alpha, "baseline alpha std", psds_std_alpha


psds_meditation = 10 * np.log10(Pxx_meditation)
psds_mean_alpha = np.mean(psds_meditation[alpha_low:alpha_high])
psds_std_alpha = np.std(np.mean(psds_meditation[alpha_low:alpha_high],axis=0))
print " meditation alpha mean", psds_mean_alpha, "meditation alpha std", psds_std_alpha
print scipy.stats.wilcoxon(np.mean(psds_baseline[alpha_low:alpha_high],axis=0),np.mean(psds_meditation[alpha_low:alpha_high],axis=0))

#beta
beta_low=138
beta_high=148
print "low beta", beta_low, "high beta", beta_high
psds_baseline = 10 * np.log10(Pxx_baseline)
psds_mean_beta = np.mean(psds_baseline[beta_low:beta_high])
psds_std_beta = np.std(np.mean(psds_baseline[beta_low:beta_high],axis=0))
print " baseline beta mean", psds_mean_beta, "baseline beta std", psds_std_beta

psds_meditation = 10 * np.log10(Pxx_meditation)
psds_mean_beta = np.mean(psds_meditation[beta_low:beta_high])
psds_std_beta = np.std(np.mean(psds_meditation[beta_low:beta_high], axis=0))
print " meditation beta mean", psds_mean_beta, "meditation beta std", psds_std_beta
print scipy.stats.wilcoxon(np.mean(psds_baseline[beta_low:beta_high],axis=0),np.mean(psds_meditation[beta_low:beta_high],axis=0))

meditation_tf_analysis=np.full((2,4,2), np.nan)
