import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import mne
from eeg_positions import get_elec_coords
from utils.filters import bandpass_torch
from sklearn.metrics import confusion_matrix
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms
from .adversarial_attacks_utils import get_channel_nb_or_id,get_attenuated_perturbation
import itertools

def get_successfully_attacked_indices(prediction, true_label, attack_label, target,device):
    
    # 1 Get where true label matches prediction
    correctly_classified = true_label == prediction
    
    # 2 Correctly classified, not as target
    true_label_LR = true_label != target
    cc_LR =  correctly_classified & true_label_LR 

    # 3 Correctly classified as L/R which are now resting
    cc_LR_2_rest = attack_label * cc_LR
    cc_LR_2_rest = cc_LR_2_rest == target

#     asr = cc_LR_2_rest.sum()/cc_LR.sum()
    return cc_LR_2_rest


def plot_sample_perturbation_attack_comparison(sample,attack_1,attack_2,save_name='_',w=12,h=6,ylimit=50,both_plots=True):
    fs_eeg=160
    t_eeg = np.arange(0, sample.shape[-1]/ fs_eeg, step=1/fs_eeg)
    low_f,high_f= 0.1,40
    
    fig = plt.figure(figsize=(w,h))
    gs = GridSpec(nrows=2, ncols=3)
    
    # Electrodes to plot
    alpha=0.3
    chan_list = [42,42]
    
    # Perturbation
    perturbation_1 = attack_1-sample
    perturbation_2 = attack_2-sample

    # Filters
    sample_filt = bandpass_torch(sample,low_f,high_f,fs_eeg=160).cpu()
    perturbation_1_filt = bandpass_torch(perturbation_1,low_f,high_f,fs_eeg=160).cpu()
    perturbation_2_filt = bandpass_torch(perturbation_2,low_f,high_f,fs_eeg=160).cpu()
    attack_filt_1 = bandpass_torch(attack_1,low_f,high_f,fs_eeg=160).cpu()
    attack_filt_2 = bandpass_torch(attack_2,low_f,high_f,fs_eeg=160).cpu()

        
    # lists 
    perturbation_list =[perturbation_1,perturbation_2]
    perturbation_filt_list =[perturbation_1_filt,perturbation_2_filt]
    attack_list =[attack_1,attack_2]
    attack_filt_list= [attack_filt_1,attack_filt_2]
    
    # Add subplots
    axes_list= []
    for i in range(6):
        axes_list.append(fig.add_subplot(gs[i//3, i%3])) #T09
        axes_list[i].set_ylim(-ylimit,ylimit)
        axes_list[i].grid(True, axis='y')
        axes_list[i].spines['top'].set_visible(False)
        axes_list[i].spines['right'].set_visible(False)
        axes_list[i].set_facecolor('whitesmoke')

    #Define columns 
    sample_col=[0,3]
    pert_col = [1,4]
    att_col = [2,5]
    bottom_row = [3,4,5]
    group_1 = [1,2]
    group_2 = [0]
    group_3 = [4,5]
    
    # Samples column
    trans = mtransforms.ScaledTranslation(-0.9, -0.5, fig.dpi_scale_trans) #12,4.8

    for ind,i in enumerate(sample_col):
        axes_list[i].plot(t_eeg,sample[chan_list[ind]], 'black',alpha=alpha)
        axes_list[i].plot(t_eeg,sample_filt[chan_list[ind]], 'black')
        axes_list[i].set_ylabel('A [mV]')
        axes_list[i].text(0.0, 1.0, f'{get_channel_nb_or_id(chan_list[ind])}', rotation=90,transform=axes_list[i].transAxes + trans, fontsize='large')

    # Perturbation column
    for ind,i in enumerate(pert_col):
        axes_list[i].plot(t_eeg,perturbation_list[ind][chan_list[ind]], 'black',alpha=alpha)
        axes_list[i].plot(t_eeg,perturbation_filt_list[ind][chan_list[ind]], 'black')

    # Attack column
    for ind,i in enumerate(att_col):
        axes_list[i].plot(t_eeg,attack_list[ind][chan_list[ind]], 'k',alpha=alpha)
        axes_list[i].plot(t_eeg,attack_filt_list[ind][chan_list[ind]], 'k')   
    
    for ind, i in enumerate(bottom_row):
        axes_list[i].set_xlabel('t [s]')
        axes_list[i].xaxis.set_ticks([0,1,2,3])

    for ind, i in enumerate(group_1):
        axes_list[i].yaxis.set_ticklabels([])
        axes_list[i].xaxis.set_visible(False)
        axes_list[i].spines['left'].set_visible(False)
        axes_list[i].spines['bottom'].set_visible(False)
    
    for ind, i in enumerate(group_2):
        axes_list[i].xaxis.set_visible(False)
        axes_list[i].spines['bottom'].set_visible(False)
    
    for ind, i in enumerate(group_3):
        axes_list[i].yaxis.set_ticklabels([])
        axes_list[i].spines['left'].set_visible(False)
    axes_list[0].set_title('Original EEG',fontsize=15)  
    axes_list[1].set_title('Adversarial perturbation',fontsize=15)  
    axes_list[2].set_title('Attacked EEG',fontsize=15)  

    plt.tight_layout()
#     plt.savefig(datapath+'4-Plots/'+save_name+".pdf",bbox_inches='tight')   
    plt.show()

def plot_sample_perturbation_attack(sample,attack,save_name='_'):
    plt.rcParams["figure.figsize"] = (18, 2) # (w, h)
    fs_eeg=160
    t_eeg = np.arange(0, sample.shape[-1]/ fs_eeg, step=1/fs_eeg)
    low_f,high_f= 0.1,40
    
    fig = plt.figure(figsize=(12,4.8))
    gs = GridSpec(nrows=4, ncols=3)#,width_ratios=[0,2,0])
    
    # Electrodes to plot
    alpha=0.3
    chan_list = [42,8,12,43]
    
    # Perturbation
    perturbation = attack-sample
    
    # Filters
    sample_filt = bandpass_torch(sample,low_f,high_f,fs_eeg).cpu()
    perturbation_filt = bandpass_torch(attack-sample,0.1,40,fs_eeg).cpu()
    attack_filt = bandpass_torch(attack,0.1,40,fs_eeg).cpu()
    
    # Add subplots
    axes_list= []
    for i in range(12):
        axes_list.append(fig.add_subplot(gs[i//3, i%3])) #T09
        axes_list[i].set_ylim(-150,150)
        axes_list[i].grid(True, axis='y')
        axes_list[i].spines['top'].set_visible(False)
        axes_list[i].spines['right'].set_visible(False)
        axes_list[i].set_facecolor('whitesmoke')

    #Define columns 
    sample_col=[0,3,6,9]
    pert_col = [1,4,7,10]
    att_col = [2,5,8,11]
    bottom_row = [9,10,11]
    group_1 = [1,2,4,5,7,8]
    group_2 = [0,3,6]
    group_3 = [10,11]
    
    # Samples column
    trans = mtransforms.ScaledTranslation(-0.9, -0.5, fig.dpi_scale_trans) #12,4.8

    for ind,i in enumerate(sample_col):
        axes_list[i].plot(t_eeg,sample[chan_list[ind]], 'k',alpha=alpha)
        axes_list[i].plot(t_eeg,sample_filt[chan_list[ind]], 'k')
        axes_list[i].set_ylabel('A [mV]')
        axes_list[i].text(0.0, 1.0, f'{get_channel_nb_or_id(chan_list[ind])}', rotation=90,transform=axes_list[i].transAxes + trans, fontsize='large')

    # Perturbation column
    for ind,i in enumerate(pert_col):
        axes_list[i].plot(t_eeg,perturbation[chan_list[ind]], 'k',alpha=alpha)
        axes_list[i].plot(t_eeg,perturbation_filt[chan_list[ind]], 'k')
    
    # Attack column
    for ind,i in enumerate(att_col):
        axes_list[i].plot(t_eeg,attack[chan_list[ind]], 'k',alpha=alpha)
        axes_list[i].plot(t_eeg,attack_filt[chan_list[ind]], 'k')   
    
    for ind, i in enumerate(bottom_row):
        axes_list[i].set_xlabel('t [s]')
        axes_list[i].xaxis.set_ticks([0,1,2,3])

    for ind, i in enumerate(group_1):
        axes_list[i].yaxis.set_ticklabels([])
        axes_list[i].xaxis.set_visible(False)
        axes_list[i].spines['left'].set_visible(False)
        axes_list[i].spines['bottom'].set_visible(False)
    
    for ind, i in enumerate(group_2):
        axes_list[i].xaxis.set_visible(False)
        axes_list[i].spines['bottom'].set_visible(False)
    
    for ind, i in enumerate(group_3):
        axes_list[i].yaxis.set_ticklabels([])
        axes_list[i].spines['left'].set_visible(False)
        
    plt.tight_layout()
    plt.show()

def plot_9hm(Nnet,dataset,datapath,eps_list,restarts,channel='T9'):
    
    #dataset: ["PhysioNet","BCI-Competition"]
    
    runs=5 if dataset=='PhysioNet' else 9
        
    asr_d = np.empty((restarts,1,9,len(eps_list),runs))
    asr_nd = np.empty((restarts,1,9,len(eps_list),runs))

    asr_d_uap = np.empty((restarts,1,9,len(eps_list),runs))
    asr_nd_uap = np.empty((restarts,1,9,len(eps_list),runs))

    for ind in range(restarts):
        asr_d[ind] = np.load(datapath + f'ASR_{dataset}_{Nnet}__{channel}_PGD_Derivative_r{ind}__eps{eps_list}_l[[1, 0.1], [1, 0.3], [1, 0.563], [5, 0.1], [5, 0.3], [5, 0.563], [15, 0.1], [15, 0.3], [15, 0.563]]_epochs1.npy')
        asr_nd[ind] = np.load(datapath + f'ASR_{dataset}_{Nnet}__{channel}_PGD_noDerivative_r{ind}__eps{eps_list}_l[[1, 0.1], [1, 0.3], [1, 0.563], [5, 0.1], [5, 0.3], [5, 0.563], [15, 0.1], [15, 0.3], [15, 0.563]]_epochs1.npy')

        asr_d_uap[ind] = np.load(datapath + f'ASR_{dataset}_{Nnet}_UAP_{channel}_PGD_Derivative_r{ind}__eps{eps_list}_l[[1, 0.1], [1, 0.3], [1, 0.563], [5, 0.1], [5, 0.3], [5, 0.563], [15, 0.1], [15, 0.3], [15, 0.563]]_epochs10.npy')
        asr_nd_uap[ind] = np.load(datapath + f'ASR_{dataset}_{Nnet}_UAP_{channel}_PGD_noDerivative_r{ind}__eps{eps_list}_l[[1, 0.1], [1, 0.3], [1, 0.563], [5, 0.1], [5, 0.3], [5, 0.563], [15, 0.1], [15, 0.3], [15, 0.563]]_epochs10.npy')

    asr_d=asr_d.mean(axis=0).squeeze()
    asr_nd=asr_nd.mean(axis=0).squeeze()

    asr_d_uap=asr_d_uap.mean(axis=0).squeeze()
    asr_nd_uap=asr_nd_uap.mean(axis=0).squeeze()

    label_list=['PGD-Derivative','PGD-NoDerivative','UAP-Derivative','UAP-NoDerivative']

    plot_HM_ASR(datapath,eps_list=eps_list, 
                asr = [asr_d,asr_nd,asr_d_uap,asr_nd_uap],
                label_list = label_list,
                suptitle='')
    
def plot_HM_ASR(datapath,eps_list, asr, label_list,suptitle):
    lambda_list = [[1,0.1],[1,0.3],[1,0.563],
                [5,0.1],[5,0.3],[5,0.563],
                [15,0.1],[15,0.3],[15,0.563]]
    plt.rcParams["figure.figsize"] = (32, 12) # (w, h)
    
    fig, axs = plt.subplots(3, 3,sharey=True)
 
    for i in range(9):
        x = i//3
        y= i%3
        for run in asr:

            axs[x,y].errorbar(eps_list,np.mean(run[i],axis=1)*100,np.nanstd(run[i],axis=1)*100,
                                     capsize=5,linewidth=5,linestyle=':')
        axs[x,y].set_title(f'$\lambda_m={lambda_list[i][0]}$ , $\lambda_d={lambda_list[i][1]}$',size=25)

        axs[x,y].set_ylim(0,110)
        axs[x,y].grid()
        if (x!=2): 
            axs[x,y].xaxis.set_ticklabels([])
        else:
            
            axs[x,y].xaxis.set_ticklabels([0,0,10,20,30,40,50],size=25)    

            if y==1:
                axs[x,y].set_xlabel('Max Amplitude [mV]',size=25)    
                
        if (y==0):            
            axs[x,y].yaxis.set_ticklabels(np.arange(0,101,20),size=25)     

            if (y==0)& (x==1):
                axs[x,y].set_ylabel('ASR [%]',size=25)
                axs[x,y].legend(label_list,fontsize=20)
    plt.tight_layout()

    fig.suptitle(suptitle,size=35)

def plot_head_model_asr(asr_array, label_list):
    plt.rcParams["figure.figsize"] = (16, 8) # (w, h)
    n_channels = asr_array.shape[0]

    sampling_freq = 160  if n_channels==64 else 250
    channel_loc_file_name = 'channel_locs' if n_channels==64 else 'channel_locs_BCI-Competition'
    bci_system='1010' if n_channels==64 else '1020'
    nb_heads = asr_array.shape[-1]
    
    channels = pd.read_table(channel_loc_file_name, names=['index','x','y','z','name'])
    channels_names = list(channels['name'].str.lstrip())
    info = mne.create_info(ch_names = channels_names,
                           ch_types=n_channels*['eeg'],
                           sfreq=sampling_freq)

    montage = get_elec_coords(system=bci_system,
                              elec_names=channels_names,
                              as_mne_montage=True)

    info.set_montage(montage)
    channels_bool = np.zeros((n_channels,1),dtype=bool)
    mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4, markeredgewidth=0)

    channels_bool[:,0] = asr_array[:,0]#[0,:,0]
    fig,ax1 = plt.subplots(nrows=nb_heads)
    
    for ind in range(nb_heads):
        
        asr = asr_array[:,ind]*100
        im,cm = mne.viz.plot_topomap(asr, 
                                     info,
                                     show=False,
                                     sensors=False,
                                     mask=channels_bool,
                                     mask_params=mask_params,
                                     vmin=30,
                                     vmax=90,
                                     axes=ax1[ind])
        ax1[ind].set_title(label_list[ind],fontsize=33)

    ax_x_start = 1
    ax_x_width = 0.03
    ax_y_start = 0.19
    ax_y_height = 0.6
    ax_ = plt.gca()
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im,cax=cbar_ax,ticks=[30,50,70,90])
    fig.tight_layout()
    clb.ax.set_title('    ASR [%]',fontsize=25,loc='center',pad=20) 
    clb.ax.tick_params(labelsize=20)

def plot_head_model(lambda_value, distances, attacked_channels, device,dataset='PhysioNet',given_channels=t.tensor(0)):
    plt.rcParams["figure.figsize"] = (18, 5) # (w, h)

    if dataset=='PhysioNet':
        sampling_freq = 160  # in Hertz
        datapoints=480
        n_channels = 64
        channels = pd.read_table('channel_locs', names=['index','x','y','z','name'])
        system='1010'
    elif dataset=='BCI-Competition':
        sampling_freq = 250  # in Hertz
        datapoints=1125
        n_channels = 22
        channels = pd.read_table('channel_locs_10-20', names=['index','x','y','z','name'])
        system='1020'
    
    channels_bool = np.zeros((n_channels,1),dtype=bool)

    if len(given_channels.shape)<1:

        v = t.ones(datapoints)
        v = get_attenuated_perturbation(perturbation=v.to(device),
                                          distances = distances,
                                          attacked_channels=attacked_channels,
                                          device=device,
                                          lambd=lambda_value).unsqueeze(0).to('cpu')
        channels_bool[:,0] = v[0,:,0]
        v=v[0, :,0]

    else:
        v=given_channels
        channels_bool[:,0] = v

    channels_names = list(channels['name'].str.lstrip())
    info = mne.create_info(ch_names = channels_names,
                           ch_types=n_channels*['eeg'],
                           sfreq=sampling_freq)

    montage = get_elec_coords(system=system,
                              elec_names=channels_names,
                              as_mne_montage=True)

    info.set_montage(montage)
    mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=7, markeredgewidth=0)
    mne.viz.plot_topomap(v, 
                         info,
                         vmin=0.6,
                         vmax=0.85,
                         show=False,
                         sensors=False,
                         names=channels_names,
                         show_names=False,
                         mask=channels_bool,
                         mask_params=mask_params)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          climit=(None,None)):
    """
    Plots the confusion matrix.
    """
    plt.rcParams["figure.figsize"] = (6, 6) # (w, h)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.clim(climit)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, ['Left','Right','Rest'],size=15)
    plt.yticks(tick_marks, ['Left','Right','Rest'],size=15)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{100*cm[i,j]:.1f}%',
                 horizontalalignment="center",fontsize=23,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label',size=20)
    plt.xlabel('Predicted label',size=20)
    plt.tight_layout()