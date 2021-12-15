import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

import utils
import models
import prune_utils

import time
import numpy as np
import matplotlib.pyplot as plt



def run(cfg, device):
    
    #########################################################
    if cfg.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        
    if cfg.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)

    if cfg.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        
    if cfg.model == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        
    if cfg.model == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        
    ############################################################    
       
    if cfg.dataset == 'mnist':
        if 'resnet' in cfg.model:
            model.conv1 = torch.nn.Conv2d(1, model.conv1.out_channels, 
                                      kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
        elif 'vgg' in cfg.model:
            model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 10)
    
    elif cfg.dataset == 'cifar100':
        if 'resnet' in cfg.model:
            model.fc = torch.nn.Linear(model.fc.in_features, 100)
        elif 'vgg' in cfg.model:
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 100)


    ############################################################################
    
    if cfg.load_model:
        model.load_state_dict(torch.load(cfg.model_path))
        model = model.to(device)
    else:  

        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, \
                                    weight_decay=cfg.l2_regularization)

        criterion = nn.CrossEntropyLoss() 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(cfg.trainloader) * 60, gamma=0.1)

        res = utils.train(cfg, model, optimizer, criterion, cfg.trainloader, cfg.testloader, device,
                      scheduler, transforms=cfg.train_transforms)
        torch.save(model.state_dict(), cfg.model_path)



    criterion = nn.CrossEntropyLoss() 
    loss, acc = utils.evaluate(model, cfg.testloader, criterion, device)
    print (f"Loss: {loss:.4f}, Accuracy: {acc*100:.2f}")


    file_name = cfg.file_name

    f = open(file_name, 'w+')
    f.write(f"Initial Model -- Loss: {loss:.4f}, Accuracy: {acc*100:.2f}\n")
    f.write("global_sparsity,conv_sparsity,fc_sparsity,loss,acc,inference_time\n")

    if cfg.prune_strategy == 'l1':
        prune_strategy = prune.L1Unstructured
    elif cfg.prune_strategy == 'l1structured':
        prune_strategy = prune_utils.L1Structured
        
    elif cfg.prune_strategy == 'l2unstructured':
        prune_strategy = prune_utils.L2Unstructured
    
    elif cfg.prune_strategy == 'random':
        prune_strategy = prune.RandomUnstructured



    if cfg.finetune:
        accs = []
        for t in range(9):
            params = prune_utils.get_child("resnet", model)
            if cfg.prune_layers == 'conv':
                prune_layers = [torch.nn.modules.conv.Conv2d]
            elif cfg.prune_layers == 'fc':
                prune_layers = [torch.nn.modules.linear.Linear]
            else:
                prune_layers = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]
            filtered = list(filter(lambda x: type(x[1]) in prune_layers, params))
            
            
            if cfg.prune_strategy == 'l1structured':
                conv_sparsity, fc_sparsity, global_sparsity = \
                    prune_utils.prune_model_structured(filtered, t / 10, (t+1)/10)
            else:
                conv_sparsity, fc_sparsity, global_sparsity = \
                    prune_utils.prune_model(filtered, t / 10, (t+1)/10, prune_strategy)


            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, \
                        weight_decay=cfg.l2_regularization)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(cfg.trainloader) * 60, gamma=0.1)

            res = utils.train(cfg, model, optimizer, criterion, cfg.trainloader, cfg.testloader, device, scheduler, transforms=cfg.train_transforms)

            st = time.time()
            loss, acc = utils.evaluate(model, cfg.testloader, criterion, device)
            et = time.time() - st
            print (f"Conv Sparsity: {100. * conv_sparsity:.2f}%")
            print (f"FC Sparsity: {100. * fc_sparsity:.2f}%")
            print (f"Global Sparsity: {100. * global_sparsity:.2f}%")
            print (f"Loss: {loss:.4f}, Accuracy: {acc*100:.2f}, Inference Time: {et:.2f}s")
            print ("")
            f.write(f"{global_sparsity},{conv_sparsity},{fc_sparsity},{loss},{acc},{et}\n")
            accs.append(acc*100)

        f.close()
        plt.title(f"{cfg.dataset} with {cfg.model} - Finetuned")
        plt.xlabel("Prune Percentage %")
        plt.ylabel("Accuracy")
        plt.plot(np.arange(10, 100, 10), accs)
        plt.show()

    else:
        params = prune_utils.get_child("resnet", model)
        if cfg.prune_layers == 'conv':
            prune_layers = [torch.nn.modules.conv.Conv2d]
        elif cfg.prune_layers == 'fc':
            prune_layers = [torch.nn.modules.linear.Linear]
        else:
            prune_layers = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]
        filtered = list(filter(lambda x: type(x[1]) in prune_layers, params))
        accs = []
        for t in range(9):
            conv_sparsity, fc_sparsity, global_sparsity = \
                prune_utils.prune_model(filtered, t / 10, (t+1)/10, prune_strategy)

            st = time.time()
            loss, acc = utils.evaluate(model, cfg.testloader, criterion, device)
            et = time.time() - st
            print (f"Conv Sparsity: {100. * conv_sparsity:.2f}%")
            print (f"FC Sparsity: {100. * fc_sparsity:.2f}%")
            print (f"Global Sparsity: {100. * global_sparsity:.2f}%")
            print (f"Loss: {loss:.4f}, Accuracy: {acc*100:.2f}, Inference Time: {et:.2f}s")
            print ("")
            f.write(f"{global_sparsity},{conv_sparsity},{fc_sparsity},{loss},{acc},{et}\n")
            accs.append(acc*100)

        f.close()
        plt.title(f"{cfg.dataset} with {cfg.model}")
        plt.xlabel("Prune Percentage %")
        plt.ylabel("Accuracy")
        plt.plot(np.arange(10, 100, 10), accs)
        plt.show()

        
        
        
        