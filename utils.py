import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange
import time
import numpy as np

def evaluate(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            _, predicted = torch.max(out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.shape[0]
            loss = criterion(out, labels)
            test_loss += loss.item()
    test_loss = test_loss / len(testloader)
    accuracy = correct / total
    return test_loss, accuracy


def train(cfg, model, optimizer, criterion, trainloader, testloader, device, scheduler=None, transforms=None, batch_transform=None, log_file=None):
    st = time.time()
    train_losses = []
    test_losses = []
    accuracies = []
    train_times = []
    lr = []
    times = []
    if log_file:
        f = open(log_file, 'w+')
    
    pbar = range(cfg.epochs)
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            step_start_time = time.time()
            labels = labels.to(device)
            if transforms:
                inputs = transforms(inputs)
            
            
            optimizer.zero_grad()
            if batch_transform:
                augmented = torch.cat([batch_transform(inputs) for _ in range(cfg.m)])
                labels = torch.cat([labels for _ in range(cfg.m)])
                out = model(augmented)
                loss = criterion(out, labels)
                loss.backward()
            
            else:
                out = model(inputs)
                loss = criterion(out, labels)
                loss.backward()
            
            optimizer.step()
            train_loss += loss.item()
            train_losses.append(loss.item())
            train_times.append(time.time() - step_start_time)
            if scheduler:
                scheduler.step()
                lr.append(scheduler.get_last_lr())

        train_loss_epoch = train_loss / len(trainloader)
#         train_losses.append(train_loss_epoch)

        model.eval()
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                _, predicted = torch.max(out, 1)
                correct += (predicted == labels).sum().item()
                total += labels.shape[0]
                loss = criterion(out, labels)
                test_loss += loss.item()
        test_loss_epoch = test_loss / len(testloader)
        test_losses.append(test_loss_epoch)
        accuracies.append(correct / total)
    #     pbar.set_postfix(test_loss=f"{test_loss_epoch:.4f}",train_loss=f"{train_loss/(i+1):.4f}")
        if epoch % cfg.log_interval == 0:
#             print (f"Epoch {epoch}, Train Loss: {train_loss/(i+1):.4f}, Test Loss: {test_loss_epoch:.4f}")
            print (f"Epoch {epoch}, Train Loss: {train_loss_epoch:.4f}, Test Loss: {test_loss_epoch:.4f}, "+
                    f"Test Accuracy: {correct / total:.4f}, avg_epoch_time: {(time.time() - st)/(epoch+1):.2f}s")
        if log_file:
            f.write(f"Epoch {epoch}, Train Loss: {train_loss_epoch:.4f}, Test Loss: {test_loss_epoch:.4f}, \
                    Test Accuracy: {correct / total:.4f}, avg_epoch_time: {(time.time() - st)/(epoch+1):.2f}s\n")
        times.append(time.time() - st)
        

    elapsed_time = time.time() - st
    res = {}
    res['model'] = model
    res['train_losses'] = train_losses
    res['test_losses'] = test_losses
    res['test_accuracies'] = accuracies
    res['time'] = times
    res['lr'] = lr
    res['train_times'] = train_times

    return res

def get_results(file_name):
    accs = []
    with open(file_name, 'r') as f:
        line = f.readline()
        loss, acc =  map(float, (line[line.index('Loss') + 6:line.index(', Accuracy')], line[line.index(', Accuracy')+12:]))
        accs.append(acc)
        f.readline()
        for line in f.readlines():
            global_sparsity,conv_sparsity,fc_sparsity,loss,acc,inference_time = line.strip().split(',')
            accs.append(float(acc)*100)
    return accs
