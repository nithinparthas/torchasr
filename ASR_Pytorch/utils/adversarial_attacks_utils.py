from cmath import nan
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from utils.data_utils import TransformSubset, PhysionetMMMI,ReshapeTensor
from utils.data_importers import *
from sklearn.model_selection import KFold
from .filters import bandpass_torch
import mne
from eeg_positions import get_elec_coords
from utils.data_importers import get_BCIcomp_data
import torch.nn as nn
import torch.nn.functional as F

"""
Helper functions for practical adversarial attacks.
"""
def add_run_constraints(run_params,eps,iterations,lambdas):
    # additional parameters of run
    run_params['Epsilon'] = eps 
    run_params['Lambdas']=lambdas
    run_params['target'] = 2 if run_params['Dataset']=='PhysioNet' else 3
    run_params['frequency_band'] = [0.1, 40]
    
    # If it's not running with UAP update MaxEpochs with only one epoch
    if not run_params['UAP']:
        run_params['MaxEpochs']=1
        
    # If it's FGSM then run only one iteration 
    if run_params['PGD']:
        run_params['Iterations'] = iterations
    else:
        run_params['Iterations']=1
                  
    # Gaussian kernels for smoothing
    if run_params['Naturalism']=='gaussian':
        factor=480//19 if run_params['Dataset']=='PhysioNet' else 1125//19
        weights,sizes = generate_gaussian_kernels(factor=1)
        # weights=[weight/0.016 for weight in weights]
    else:
        weights,sizes=0,0
    run_params['weights']=weights
    run_params['sizes']=sizes
    
    if run_params['loss function']=='Liu et al':
        run_params['loss regularizer']='l1'# 'l1','l2','l1+l2'
        run_params['loss alpha']=1e0
    
    return run_params

def add_spatial_constraints(perturbation,
                            distances,
                            fs,
                            attacked_channels,
                            device,
                            lambdas=[1,0.563]):
    
    lambd,lambda_2=lambdas
    
    # do attenuation
    perturbation = get_attenuated_perturbation(perturbation,distances,attacked_channels,device,lambd=lambd)
    
    #do phase shift
    perturbation = get_shifted_perturbation(perturbation,distances, fs, attacked_channels, lambd=lambda_2)
    # perturbation = get_shifted_perturbation_(perturbation, delays)

    return perturbation

def evaluate_attack(model, loader, perturbation, target=2, frequency_band = [0.1,40],fs_eeg=160, device=t.device('cpu')):
    
    #1 Initialize tensors to gather results
    prediction_result = t.empty(0).to(device)
    true_labels = t.empty(0).to(device)
    attack_labels = t.empty((0)).to(device)

    for sample,label in loader:
        
        # 1 Prepare sample and get original classification
        sample = sample.to(device)
        y_pred = bandpass_torch(sample.detach().clone(),frequency_band[0],frequency_band[1],fs_eeg=fs_eeg,device=device)
        y_pred = model(y_pred).argmax(dim=1) # get the index

        # 2 Filter to the indices of samples which are going to be attacked
        sample, label, y_pred = get_samples_to_attack(sample, label, y_pred, target)

        # 3 Perform attack
        attack = sample + perturbation

        # 4 Collect classification of original sample
        prediction_result = t.cat((prediction_result,y_pred))
        
        # 5 Collect classification of attack
        attack_classification = bandpass_torch(attack, frequency_band[0],frequency_band[1], fs_eeg=fs_eeg,device=device)
        attack_classification = model(attack_classification.detach()).argmax(dim=1)
        attack_labels = t.cat((attack_labels,attack_classification))
        
        # 6 Collect the true labels
        true_labels = t.cat((true_labels,label.to(device)))

    asr= get_attack_success_rate(prediction_result, true_labels, attack_labels,target, device)
    
    return asr 

def generate_gaussian_kernels(factor=1):
    sizes = [5, 7, 11, 15, 19]
    sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
    sigmas = [x*factor for x in sigmas]
    sizes=[int(x*factor) for x in sizes]
    crafting_sizes = []
    crafting_weights = []
    for size in sizes:
        for sigma in sigmas:
            crafting_sizes.append(size)
            weight = np.arange(size) - size//2
            weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
            weight = t.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(t.FloatTensor)
            crafting_weights.append(weight)
    return crafting_weights, crafting_sizes


def get_attack_success_rate(prediction, true_label, attack_label, target,device):
    
    # 1 Get where true label matches prediction
    correctly_classified = true_label == prediction
    
    # 2 Correctly classified, not as target
    true_label_LR = true_label != target
    cc_LR =  correctly_classified & true_label_LR 

    # 3 Correctly classified as L/R which are now resting
    cc_LR_2_rest = attack_label * cc_LR
    cc_LR_2_rest = cc_LR_2_rest == target
    
    asr = cc_LR_2_rest.sum()/cc_LR.sum()

    return asr

def get_attenuated_perturbation(perturbation,distances,attacked_channels,device,lambd=1e2):
    #Returns a tensor of size N_channel X N_datapoints 
    distances = t.from_numpy(distances.astype('float32')).unsqueeze(1).to(device)
    a = 1-lambd*distances
    a[a<0]=0
    v2 = a*perturbation
    return v2

def get_channel_nb_or_id(channel):
    #Get the channel number if the id is provided or viceversa
    channels = pd.read_table('channel_locs', names=['index','x','y','z','name'])
    if type(channel)==str:        
        channels_names = list(channels['name'].str.lstrip())
        return channels_names.index(channel)
    
    elif type(channel)==int:
        return channels.name[channel].lstrip()

def get_cosine_similarity(samples,attack_samples,fs_eeg,device, attacked_channels=False):
    
    cos = nn.CosineSimilarity(dim=-1)
    distance=cos(bandpass_torch(attack_samples,0.1,40,fs_eeg).to(device),bandpass_torch(samples,0.1,40,fs_eeg).to(device))

    if attacked_channels:
        distance = distance[attacked_channels]
    else:
        distance = distance.mean(axis=-1)
    
    return distance.mean(axis=-1).cpu().item()

def get_dataset_model_and_loaders(run,device,dataset_params, model_net, params,train_idx=0, valid_idx=0):
    '''
    run: fold# for PhysioNet, subject# for BCI-Competition  
    '''
    if dataset_params['name']=='PhysioNet':
        transform = transforms.Compose([ReshapeTensor()])
        train_set = TransformSubset(dataset_params['dataset'], train_idx, transform)
        valid_set = TransformSubset(dataset_params['dataset'], valid_idx, transform)

        # Load the model 'EEGNetX.net' or 'EEGTCNetX.net'
        dataset_params['model'] = t.load(dataset_params['data_path'] +f'/1-Model/PhysioNet_{model_net}_fold{run}.net',map_location = device).eval()
        dataset_params['train_loader'] = t.utils.data.DataLoader(train_set, **params)
        dataset_params['val_loader'] = t.utils.data.DataLoader(valid_set, **params)
    
    elif dataset_params['name'] == 'BCI-Competition':
        
        # Load the model 'EEGNetX.net', 'EEGTCNetX.net' or 'ShallowConvNet'
        dataset_params['model'] = t.load(dataset_params['data_path'] + f'/1-Model/{model_net}{run+1}.net',map_location = device)#.eval()
        dataset_params['model'].train(False)

        samples, labels = get_BCIcomp_data(subject=run+1,data_path=dataset_params['data_path']+'/2-Data/', training=True)
        dataset_params['train_loader'] = as_data_loader(samples,labels,device,batch_size=params['batch_size'],scale=True)
        
        samples, labels = get_BCIcomp_data(subject=run+1,data_path=dataset_params['data_path']+'/2-Data/', training=False)
        dataset_params['val_loader'] = as_data_loader(samples,labels,device,batch_size=params['batch_size'],scale=True)
    
    return dataset_params    

def get_distances(attacked_channel_list,dataset='PhysioNet'):
    ''''
    Read the file with xyz coordinates and names and get list of remaining channels 
    (to which the distances will be calculated)
    '''
    if dataset=='PhysioNet':
        channels = pd.read_table('channel_locs', names=['index','x','y','z','name'])
        remaining = np.array([channel for channel in list(range(64)) if channel not in attacked_channel_list]) 
    elif dataset=='BCI-Competition':
        channels = pd.read_table('channel_locs_BCI-Competition', names=['index','x','y','z','name'])
        remaining = np.array([channel for channel in list(range(22)) if channel not in attacked_channel_list]) 

    #radius,conductivity and coordinates
    radius= 8.7e-2 #meters

    x = radius*channels['x'].values
    y = radius*channels['y'].values
    z = radius*channels['z'].values
    
    #only one channel
    attacked_channel = attacked_channel_list[0]
    
    
    distances_np= np.empty(0)

    for to_channel in remaining:
        ch1_pos = np.array([x[attacked_channel],y[attacked_channel],z[attacked_channel]])
        ch2_pos = np.array([x[to_channel],y[to_channel],z[to_channel]])
        distance = radial_dist(ch1_pos,ch2_pos, radius)
        distances_np = np.append(distances_np,distance)  
    distances_np = np.insert(distances_np,attacked_channel,0)

    return distances_np

def get_euclidean_distance(samples,attack_samples,fs_eeg,attacked_channels=False):
    
    distance= bandpass_torch(attack_samples,0.1,40,fs_eeg)-bandpass_torch(samples,0.1,40,fs_eeg)
    
    if attacked_channels:
        distance= t.linalg.norm(distance,dim=(-1))[attacked_channels].mean(-1).cpu() # Only for the attacked electrode
    else:
        distance= t.linalg.norm(distance,dim=(-1)).mean(-1).mean(-1).cpu()
    return distance.item()

def get_indices_to_keep(y_pred, label, target):
    
    # 1 Return indices where attack matches prediction
    correctly_classified = label == y_pred
    return [i for i, j in enumerate(correctly_classified) if j == True]

def get_key_from_value(dictionary, key,multiple_channels=False):
    if key=='PGD':
        if dictionary['Perturbation type']=='attack':
            return key if dictionary[key] else 'FGSM'
        else:
            return dictionary['Perturbation type']
    elif key=='Head Model':
        return key if dictionary[key] else 'noHM'
    elif key =='UAP':
        if  dictionary[key]:
            return 'UAP_Liu_et_al' if dictionary['loss function']=='Liu et al' else 'UAP'
        else:
            return ''
    elif key =='AblatedHM':
        return key if dictionary[key] else ''
    elif key=='Naturalism':
        if dictionary[key]=='derivative':
            return 'Derivative'
        elif dictionary[key]=='gaussian':
            return dictionary[key] 
        else:
            return 'noDerivative' 
    elif key=='Attacked channels':
        return get_channel_nb_or_id(dictionary[key][0]) if not multiple_channels else 'Allchannels'
    else:
        return key if dictionary[key] else 'no'+ key

def get_samples_to_attack(sample, label, y_pred, target=2):
    
    indices_to_keep = get_indices_to_keep(y_pred.cpu(), label.cpu(), target)
    
    if indices_to_keep:
        sample = sample[indices_to_keep]
        label = label[indices_to_keep]
        y_pred = y_pred[indices_to_keep]

    return sample, label, y_pred

def get_shifted_perturbation(perturbation, distances, fs, attacked_channels, lambd=1e2):
    d0 = (1./360) #~3ms/10cm 
    
    # If lambda is negative, then there is no delay at all, otherwise it's the slope, int(1/360 *160) is 
    if lambd>0:
        b = d0-lambd*0.1
        channel_delays = lambd * distances + b
    elif lambd==0:
        b = d0
        channel_delays = lambd * distances + b
        channel_delays[attacked_channels]=0
    elif lambd<0:
        lambd,b =0,0
        channel_delays = np.zeros(distances.shape[0])
    #Clip negative values to zero
    channel_delays[np.where(channel_delays<0)]=0
    channel_delays = (channel_delays*fs).astype(int)
    for channel, delay in enumerate(channel_delays):   
        # define the shift as a proportion of the total signal 
        # shift the signal and delete the first elements of the shift
        perturbation[...,channel,:]=t.roll(perturbation[...,channel,:],delay)
        perturbation[...,channel,:delay]=0
    
    return perturbation

def get_x_correlation(samples,attack_samples,fs_eeg,attacked_channels=False,freq_domain=False):
    
    # Bandpass
    attack_samples = bandpass_torch(attack_samples,0.1,40,fs_eeg).cpu()
    samples = bandpass_torch(samples,0.1,40,fs_eeg).cpu()
    
    # Initialize array to gather correlations
    if attacked_channels:
        CC_array = np.zeros((attack_samples.shape[0],len(attacked_channels)))
    else:   
        CC_array = np.zeros((attack_samples.shape[0],attack_samples.shape[1]))
    
    # Iterate through all samples
    for i in range(attack_samples.shape[0]):      
        
        if attacked_channels: # Code only for one attacked channel
            if not freq_domain:

                # Cross correlation in time
                cross_correlation = np.correlate(samples[i,attacked_channels].squeeze().numpy(),attack_samples[i,attacked_channels].squeeze().numpy())
                reference = np.correlate(samples[i,attacked_channels].squeeze().numpy(),samples[i,attacked_channels].squeeze().numpy())

            else:
                # Cross correlation in frequency domain
                cross_correlation = np.correlate(np.fft.fft(attack_samples[i,samples].squeeze().numpy()),np.fft.fft(attack_samples[i,attacked_channels].squeeze().numpy()))
                reference = np.correlate(np.fft.fft(samples[i,attacked_channels].squeeze().numpy()),np.fft.fft(samples[i,attacked_channels].squeeze().numpy()))
                
            CC_array[i]=abs(reference-cross_correlation)
        
        else: # All channels
            for channel in range(attack_samples.shape[-2]):     
                if not freq_domain:
                    # Cross correlation in time
                    cross_correlation = np.correlate(samples[i,channel].numpy(),attack_samples[i,channel].numpy())
                    reference = np.correlate(samples[i,channel].numpy(),samples[i,channel].numpy())
                
                else:
                    # Cross correlation in frequency domain
                    cross_correlation = np.correlate(np.fft.fft(np.fft.fft(samples[i,channel].numpy(),attack_samples[i,channel].numpy())))
                    reference = np.correlate(np.fft.fft(samples[i,channel].numpy()),np.fft.fft(samples[i,channel].numpy()))
                CC_array[i,channel]=abs(reference-cross_correlation)
    
    return CC_array.mean(axis=-1).mean(axis=-1)
 
def loss_function_LiuEtAl(model,
                            sample,
                            v,
                            run_params,
                            device,
                            fs=160):
    # Regularization
    if run_params['loss regularizer']=='l1':
        reg_loss = run_params['loss alpha']*t.abs(v).mean()
    elif run_params['loss regularizer']=='l2':
        reg_loss = run_params['loss alpha']*(v**2).mean()
    elif run_params['loss regularizer']=='l1+l2':
        reg_loss = run_params['loss alpha']*(10*(v**2).mean()+t.abs(v).mean())
        
    # FFT filtering
    xbp = bandpass_torch(sample + v, run_params['frequency_band'][0],run_params['frequency_band'][1], fs_eeg=fs,device=device) 
    
    # CE Loss Forward pass
    ce_loss = nn.CrossEntropyLoss()
    classification_result = model(xbp)
    CE_Loss = ce_loss(classification_result,t.tensor([run_params['target']]*sample.shape[0]).to(device))
    loss = CE_Loss

    return t.sum(loss)+reg_loss

def radial_dist(ch1, ch2,r):
    angle = np.arccos((ch1 @ ch2) / (np.linalg.norm(ch1) * np.linalg.norm(ch2)))
    return angle * r

def smooth_perturbation(perturbation,weights_list,sizes_list,device):
    channels = perturbation.shape[-2]
    unsqueeze_back =False
    if len(perturbation.shape)>3:
        unsqueeze_back = True
        perturbation=perturbation.squeeze()
    smoothed_perturbation=0
    for i in range(len(sizes_list)):
        smoothed_perturbation = smoothed_perturbation + F.conv1d(perturbation, weights_list[i].to(device).repeat(channels,1,1), padding = sizes_list[i]//2,groups=channels)
    smoothed_perturbation = smoothed_perturbation/float(len(sizes_list))
    if unsqueeze_back:
        smoothed_perturbation=smoothed_perturbation.unsqueeze(1)
    return smoothed_perturbation

def plot_head_model(lambda_value, distances, attacked_channels, device):
    sampling_freq = 160  # in Hertz

    v = t.ones(480)
    n_channels = 64
    channels = pd.read_table('channel_locs', names=['index','x','y','z','name'])
    channels_names = list(channels['name'].str.lstrip())
    info = mne.create_info(ch_names = channels_names,
                           ch_types=n_channels*['eeg'],
                           sfreq=sampling_freq)

    montage = get_elec_coords(system='1010',
                              elec_names=channels_names,
                              as_mne_montage=True)

    info.set_montage(montage)
    channels_bool = np.zeros((n_channels,1),dtype=bool)
    mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4, markeredgewidth=0)

    v = get_attenuated_perturbation(perturbation=v.to(device),
                                          distances = distances,
                                          attacked_channels=attacked_channels,
                                          device=device,
                                          lambd=lambda_value).unsqueeze(0).to('cpu')
    channels_bool[:,0] = v[0,:,0]

    mne.viz.plot_topomap(v[0, :,0], 
                         info,
                         show=False,
                         sensors=False,
                         mask=channels_bool,
                         mask_params=mask_params)

    plt.savefig(data_path + '/4-Plots/' + f"Headmodel_Attenuated_{lambda_value}.pdf",bbox_inches='tight')