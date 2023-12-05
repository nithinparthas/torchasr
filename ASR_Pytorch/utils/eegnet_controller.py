import numpy as np
import torch as t
from tqdm import tqdm
from .models import EEGNet,EEGTCNet,ShallowConvNet
from .early_stopping import EarlyStopping
from .data_importers import get_BCIcomp_data,as_data_loader,as_tensor
from .metrics import get_metrics_from_model
import os
from .filters import bandpass_torch
from .functions import MaxNormDefaultConstraint,set_random_seeds

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../1-Data/2-BCI-Competition/2-Data/'))

def train_subject_specific(subject, epochs=500, batch_size=32, Nnet='EEGNet',lr=0.001, silent=False, plot=True,
                           **kwargs):
    """
    Trains a subject specific model for the given subject

    Parameters:
     - subject:    Integer in the Range 1 <= subject <= 9
     - epochs:     Number of epochs to train
     - batch_size: Batch Size
     - lr:         Learning Rate
     - silent:     bool, if True, hide all output including the progress bar
     - plot:       bool, if True, generate plots
     - kwargs:     Remaining arguments passed to the EEGnet model

    Returns: (model, metrics)
     - model:   t.nn.Module, trained model
     - metrics: t.tensor, size=[1, 4], accuracy, precision, recall, f1
    """
    # cuda_av = t.cuda.is_available() 
    # set_random_seeds(seed=20190706, cuda=cuda_av)


    # load the data
    train_samples, train_labels = get_BCIcomp_data(subject, training=True,data_path=DATA_PATH)
    test_samples, test_labels = get_BCIcomp_data(subject, training=False,data_path=DATA_PATH)

    scale_data=False
    if Nnet=='EEGTCNet' or Nnet=='ShallowConvNet':
        scale_data=True
    train_loader = as_data_loader(train_samples, train_labels, batch_size=batch_size,device=t.device('cuda'),scale=scale_data)
    test_loader = as_data_loader(test_samples, test_labels, batch_size=batch_size,device=t.device('cuda'),scale=scale_data)

    # prepare the model
    eps=1e-7
    if Nnet=='EEGNet':
        model = EEGNet(T=train_samples.shape[2],C=22,N=4, **kwargs) 
        # model.initialize_params()
    elif Nnet=='EEGTCNet':
        eps=1e-8
        model = EEGTCNet(T=train_samples.shape[2],pt=0.3,Ft=12,p_dropout=0.2,activation='elu', **kwargs)
    elif Nnet == 'ShallowConvNet':
        model=ShallowConvNet()
        model.initialize_params()
    if t.cuda.is_available():
        model = model.cuda()

    # prepare loss function and optimizer
    loss_function = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr, eps=eps)
    scheduler = None

    # prepare progress bar
    with tqdm(desc=f"Subject {subject}", total=epochs, leave=False, disable=silent,
              unit='epoch', ascii=True) as pbar:
        
        # Early stopping is not allowed in this mode, because the testing data cannot be used for
        # training!
        model, metrics, _, history = _train_net(subject, model, train_loader, test_loader,
                                                loss_function, optimizer, scheduler=scheduler,
                                                epochs=epochs, early_stopping=False, plot=plot,
                                                pbar=pbar)
    
    if not silent:
        print(f"Subject {subject}: accuracy = {metrics[0, 0]}")
    
    # Save model 
    data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../1-Data/2-BCI-Competition/1-Model/'))
    # t.save(model, data_path+f'/{Nnet}{subject}.net')
    # print("Model saved as: " + f'{Nnet}{subject}.net')
    return model, metrics, history

def _train_net(subject, model, train_loader, val_loader, loss_function, optimizer, scheduler=None,
               epochs=500, early_stopping=True, plot=False, track_lr=True, pbar=None,do_modelconstr=True):
    """
    Main training loop

    Parameters:
     - subject:        Integer, subject ID
     - model:          t.nn.Module (is set to training mode)
     - train_loader:   t.utils.data.DataLoader: training data
     - val_loader:     t.utils.data.DataLoader: validation data
     - loss_function:  function
     - optimizer:      t.optim.Optimizer
     - scheduler:      t.optim.lr_scheduler or None
     - epochs:         Integer, number of epochs to train
     - early_stopping: boolean, if True, store models for all epochs and select the one with the
                       highest validation accuracy
     - plot:           boolean, if True, generate all plots and store on disk
     - pbar:           tqdm progress bar or None, in which case no progress will be displayed
                       (not closed afterwards)

    Returns: (model, metrics, epoch, history)
     - model:   t.nn.Module, trained model
     - metrics: t.tensor, size=[1, 4], accuracy, precision, recall, f1
     - epoch:   integer, always equal to 500 if early stopping is not used
     - history: tuple: (loss, accuracy), where both are t.tensor, size=[2, epochs]

    Notes:
     - Model and data will not be moved to gpu, do this outside of this function.
     - When early_stopping is enabled, this function will store all intermediate models
    """
    if do_modelconstr:
        model_constraint = MaxNormDefaultConstraint()
    else: 
        model_constraint = None

    # prepare result
    loss = t.zeros((2, epochs))
    accuracy = t.zeros((2, epochs))
    lr = None
    if track_lr:
        lr = t.zeros((epochs))

    # prepare early_stopping
    if early_stopping:
        early_stopping = EarlyStopping()

    use_cuda = model.is_cuda()

    # train model for all epochs
    for epoch in range(epochs):
        # train the model
        train_loss, train_accuracy = _train_epoch(model, train_loader, loss_function, optimizer,
                                                  scheduler=scheduler, use_cuda=use_cuda,model_constraint=model_constraint)

        # collect current loss and accuracy
        validation_loss, validation_accuracy = _test_net(model, val_loader, loss_function,
                                                         train=False, use_cuda=use_cuda)
        loss[0, epoch] = train_loss
        loss[1, epoch] = validation_loss
        accuracy[0, epoch] = train_accuracy
        accuracy[1, epoch] = validation_accuracy
        if track_lr:
            lr[epoch] = optimizer.param_groups[0]['lr']

        # do early stopping
        if early_stopping:
            early_stopping.checkpoint(model, loss[1, epoch], accuracy[1, epoch], epoch)

        if pbar is not None:
            pbar.update()

    # get the best model
    if early_stopping:
        model, best_loss, best_accuracy, best_epoch = early_stopping.use_best_model(model)
    else:
        best_epoch = epoch

    metrics = get_metrics_from_model(model, val_loader)

    return model, metrics, best_epoch + 1, (loss, accuracy)

def _train_epoch(model, loader, loss_function, optimizer, scheduler=None, use_cuda=None,model_constraint=None):
    """
    Trains a single epoch

    Parameters:
     - model:         t.nn.Module (is set to training mode)
     - loader:        t.utils.data.DataLoader
     - loss_function: function
     - optimizer:     t.optim.Optimizer
     - scheduler:     t.optim.lr_scheduler or None
     - use_cuda:      bool or None, if None, use cuda if the model is on cuda

    Returns: loss: float, accuracy: float
    """

    if use_cuda is None:
        use_cuda = model.is_cuda()

    model.train(True)
    n_samples = 0
    running_loss = 0.0
    accuracy = 0.0
    for x, y in loader:
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        # Forward step
        optimizer.zero_grad()
        
        output = model(bandpass_torch(x,low_f=0.1,high_f=40,fs_eeg=250))

        loss = loss_function(output, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # constraint weights if necessary
        if model_constraint is not None:
            model_constraint.apply(model)
        
        # prepare loss and accuracy
        n_samples += x.shape[0]
        running_loss += loss * x.shape[0]
        # decision = class_decision(output)
        decision = output.argmax(axis=1)
        accuracy += (decision == y).sum().item()

    running_loss = running_loss / n_samples

    if scheduler:
        scheduler.step(running_loss)

    accuracy = accuracy / n_samples
    return running_loss, accuracy

def _test_net(model, loader, loss_function, train=False, use_cuda=None, attack_samples=False):
    """
    Tests the model for accuracy

    Parameters:
     - model:         t.nn.Module (is set to testing mode)
     - loader:        t.utils.DataLoader
     - loss_function: function or None.
     - train:         boolean, if the model is to be set into training or testing mode
     - use_cuda:      bool or None, if None, use cuda if the model is on cuda
     - attack_samples bool, True if testing the attack 

    Returns: loss: float, accuracy: float, results: vector of integers
    """
    if use_cuda is None:
        use_cuda = model.is_cuda()

    # set the model into testing mode
    model.train(train)
    with t.no_grad():

        running_loss = 0.0
        running_acc = 0.0
        n_total = 0
        results=[]

        # get the data from the loader (only one batch will be available)
        for x, y in loader:
            if use_cuda:
                x = x.cuda()
                y = y.cuda()

            # compute the output
            output = model(bandpass_torch(x,low_f=0.1,high_f=40,fs_eeg=250))

            # compute accuracy
            yhat = output.argmax(axis=1)
            prediction_correct = yhat == y
            num_correct = prediction_correct.sum().item()
            running_acc += num_correct

            # compute the loss function
            loss = loss_function(output, y)
            running_loss += loss * x.shape[0]

            # increment sample counter
            n_total += x.shape[0]

    return running_loss / n_total, running_acc / n_total
