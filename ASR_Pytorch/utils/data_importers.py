#!/usr/bin/env python3

import os
import numpy as np
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib as edf
import statistics as stats
import random
import scipy.io as sio
import torch as t

from filters import highpass
from sklearn.preprocessing import StandardScaler

def get_data(path, long = False, normalization = 0,subjects_list=range(1,110), n_classes=4):
    '''
    Loads '.edf' MI data from Physionet 
    Load data samples and return it as one array 

    Parameters:
    -----------
    path:   string
        path to .edf data of 
    normalization   int {0,1}

        normalization per trial 
        0: no normalization; 1: normalized per channel
    long:    bool 
        length of read time window
        True: Trials of length 6s returned; False: Trials of length 3s returned
    subjects_list   list [1, .. , 109] 
        list of subject numbers to be loaded
    n_classes:      int 
        number of classes
        2: L/R, 3: L/R/0, 4
    
    Return: X:  numpy array (n_sub*n_trials, 64, n_samples) 
                EEG data 
            y:  numpy array (n_sub*n_trials, n_samples)
                labels 
    '''
    # Define subjects whose data is not taken, for details see data tester added 106 again to analyze it, deleted from the exluded list
    excluded_subjects = [88,92,100,104]
    # Define subjects whose data is taken, namely from 1 to 109 excluding excluded_subjects
    subjects = [x for x in subjects_list if (x not in excluded_subjects)]
   

    mi_runs = [1, 4, 6, 8, 10, 12, 14]
    # Extract only requested number of classes
    if(n_classes == 3):
        print('Returning 3 Class data')
        mi_runs.remove(6) # feet
        mi_runs.remove(10) # feet
        mi_runs.remove(14) # feet
    elif(n_classes == 2):
        print('Returning 2 Class data')
        mi_runs.remove(6) # feet
        mi_runs.remove(10) # feet
        mi_runs.remove(14) # feet
        mi_runs.remove(1) #rest 
    print(f'Data from runs: {mi_runs}')

    X, y = read_data(subjects = subjects,runs = mi_runs, path=path, long=long)
   
    # do normalization if wanted
    if(normalization == 1):
        #TODO: declare std_dev, mean arrays to return
        for i in range(X.shape[0]):
            for ii in range(X.shape[1]):
                std_dev = stats.stdev(X[i,ii])
                mean = stats.mean(X[i,ii])
                X[i,ii] = (X[i,ii] - mean) / std_dev
        
    return X, y

    
def read_data(subjects , runs, path, long=False):
    '''
    Load data samples and return it as one array 

    Parameters:
    -----------
    subjects   list [1, .. , 109] 
        list of subject numbers to be loaded
    path:   string
        path to .edf data of 
    runs    list 
        runs to read from 
    long:    bool 
        length of read time window
        True: Trials of length 6s returned; False: Trials of length 3s returned
    
    
    Return: X:  numpy array (n_sub*n_trials, 64, n_samples) 
                EEG data 
            y:  numpy array (n_sub*n_trials, n_samples)
                labels 
    '''

    """
    DATA EXPLANATION:
        
        LABELS:
        both first_set and second_set
            T0: rest
        first_set (real motion in runs 3, 7, and 11; imagined motion in runs 4, 8, and 12)
            T1: the left fist 
            T2: the right fist
        second_set (real motion in runs 5, 9, and 13; imagined motion in runs 6, 10, and 14)
            T1: both fists
            T2: both feet
        
        Here, we get data from the first_set (rest, left fist, right fist), 
        and also data from the second_set (rest, both feet).
        We ignore data for T1 from the second_set and thus return data for 
        four classes/categories of events: Rest, Left Fist, Right Fist, Both Feet.
    """
    base_file_name = 'S{:03d}R{:02d}.edf'
    base_subject_directory = 'S{:03d}'
    
    # Define runs where the two different sets of tasks were performed
    baseline = np.array([1])
    first_set = np.array([4,8,12])
    second_set = np.array([6,10,14])
    
    # Number of EEG channels
    NO_channels = 64
    # Number of Trials extracted per Run
    NO_trials = 7
    
    # Define Sample size per Trial 
    if not long:
        n_samples = int(160 * 3) # 3s Trials: 480 samples
    else:
        n_samples = int(160 * 6) # 6s Trials: 960 samples 
    
    # initialize empty arrays to concatanate with itself later
    X = np.empty((0,NO_channels,n_samples))
    y = np.empty(0)
    
    for subject in subjects:

        for run in runs:
            #For each run, a certain number of trials from corresponding classes should be extracted
            counter_0 = 0
            counter_L = 0
            counter_R = 0
            counter_F = 0
            
            # Create file name variable to access edf file
            filename = base_file_name.format(subject,run)
            directory = base_subject_directory.format(subject)
            file_name = os.path.join(path,directory,filename)
            # Read file
            f = edf.EdfReader(file_name)
            # Signal Parameters - measurement frequency
            fs = f.getSampleFrequency(0)
            # Number of eeg channels = number of signals in file
            n_ch = f.signals_in_file
            # Initiate eg.: 64*20000 matrix to hold all datapoints
            sigbufs = np.zeros((n_ch, f.getNSamples()[0]))
            
            for ch in np.arange(n_ch):
                # Fill the matrix with all datapoints from each channel
                sigbufs[ch, :] = f.readSignal(ch)
            
            # Get Label information
            annotations = f.readAnnotations()
            
            # close the file
            f.close()
            
            # Get the specific label information
            labels = annotations[2]
            points = fs*annotations[0]
            
            labels_int = np.empty(0)
            data_step = np.empty((0,NO_channels, n_samples))             
            
            if run in second_set:
                for ii in range(0,np.size(labels)):
                    if(labels[ii] == 'T0' and counter_0 < NO_trials):
                        continue
                        counter_0 += 1
                        labels_int = np.append(labels_int,[2])
                        
                    elif(labels[ii] == 'T2' and counter_F < NO_trials):
                        counter_F += 1
                        labels_int = np.append(labels_int,[3])
                        # change data shape and seperate events
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+n_samples])[None]))        
                
            elif run in first_set:
                for ii in range(0,np.size(labels)):
                    if(labels[ii] == 'T0' and counter_0 < NO_trials):
                        continue
                        counter_0 += 1
                        labels_int = np.append(labels_int, [2])
                        
                    elif(labels[ii] == 'T1' and counter_L < NO_trials):
                        counter_L += 1
                        labels_int = np.append(labels_int, [0])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+n_samples])[None]))
                        
                    elif(labels[ii] == 'T2' and counter_R < NO_trials):
                        counter_R += 1
                        labels_int = np.append(labels_int, [1])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+n_samples])[None]))
                        
            elif run in baseline:
                for ii in range(0,20):
                    if(counter_0 < 20):  
                        counter_0 += 1
                        labels_int = np.append(labels_int, [2])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,(ii*n_samples):((ii+1)*n_samples)])[None]))
                # randomly choose resting trials
                np.random.seed(7)
                index = random.randint(0*fs,57*fs)
                labels_int = np.append(labels_int, [2])
                data_step = np.vstack((data_step, np.array(sigbufs[:,(index):(index+n_samples)])[None]))
               
            # concatenate arrays in order to get the whole data in one input array    
            X = np.concatenate((X,data_step))
            y = np.concatenate((y,labels_int))
        
    return X, y

def get_BCIcomp_data(subject, training, data_path=None, do_filter=False):
    """
    Loads the dataset 2a of the BCI Competition IV
    available on http://bnci-horizon-2020.eu/database/data-sets

    arguments:
     - subject:   number of subject in [1, .. ,9]
     - training:  if True, load training data
                  if False, load testing data
     - data_path: String, path to the BCI IV 2a dataset (.mat files)
     - do_filter: bool, apply highpass filter at fc= 4Hz if true

    Returns: data_return,  numpy matrix, size = NO_valid_trial x 22 x 1750
             class_return, numpy matrix, size = NO_valid_trial
    """

    if data_path is None:
        data_path = DATA_PATH

    NO_channels = 22
    NO_tests = 6 * 48
    Window_Length = 7 * 250

    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests, NO_channels, Window_Length))

    n_valid_trials = 0
    if training:
        a = sio.loadmat(os.path.join(data_path, 'A0' + str(subject) + 'T.mat'))
    else:
        a = sio.loadmat(os.path.join(data_path, 'A0' + str(subject) + 'E.mat'))
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_fs = a_data3[3]
        # a_classes = a_data3[4]
        a_artifacts = a_data3[5]
        # a_gender = a_data3[6]
        # a_age = a_data3[7]

        # apply filter
        if do_filter:
            for ch in range(a_X.shape[1]):
                a_X[:, ch] = highpass(a_X[:, ch], a_fs, 4)

        for trial in range(0, a_trial.size):
            if a_artifacts[trial] == 0:
                range_a = int(a_trial[trial])
                range_b = range_a + Window_Length
                data_return[n_valid_trials, :, :] = np.transpose(a_X[range_a:range_b, :22])
                class_return[n_valid_trials] = int(a_y[trial])
                n_valid_trials += 1

    data_return = data_return[0:n_valid_trials, :, :]
    data_return = _use_time_window_post_cue(data_return)
    return data_return, class_return[0:n_valid_trials]


def as_tensor(samples, labels):
    """
    Returns the data as t.tensor, with the correct data type

    Parameters:
     - samples: np.ndarray, size = [s, C, T]
     - labels:  np.ndarray, size = [s]

    Returns:
     - samples: t.tensor, size = [s, C, T], dtype = float
     - labels:  t.tensor, size = [s],       dtype = long
    """

    x = t.tensor(samples).to(dtype=t.float)
    y = t.tensor(labels).to(dtype=t.long) - 1  # labels are from 1 to 4, but torch expects 0 to 3

    return x, y


def as_data_loader(samples, labels, device,batch_size=32, shuffle=True, scale=False):
    """
    Returns the data as a t.utils.data.DataLoader.
    Moves data to device if available

    arguments:
     - samples: np.ndarray, size = [s, C, T]
     - labels:  np.ndarray, size = [s]

    Returns: t.utils.data.Dataloader
    """
    if scale:
        for j in range(22):
            scaler = StandardScaler()
            scaler.fit(samples[:,j,:])
            samples[:,j,:] = scaler.transform(samples[:,j,:])
    x, y = as_tensor(samples, labels)
    dataset = t.utils.data.TensorDataset(x.to(device), y)
    loader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def _use_time_window_post_cue(x, fs=250, t1_factor=1.5, t2_factor=6):
    """
    Prepares the input data to only use the post-cue range.

    Parameters:
     - x:         np.ndarray, size = [s, C, T], where T should be 1750
     - fs:        integer, sampling rate
     - t1_factor: float, window will start at t1_factor * fs
     - t2_factor: float, window will end at t2_factor * fs

    Returns np.ndarray, size = [s, C, T'], where T' should be 1125 with default values
    """

    assert t1_factor < t2_factor

    t1 = int(t1_factor * fs)
    t2 = int(t2_factor * fs)

    return x[:, :, t1:t2]
