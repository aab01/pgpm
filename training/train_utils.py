
import torch as T
import torch.nn as nn
import torch.nn.functional as F

def dseg_train_epoch(net, train_ldr, criterion, optimizer, device):
    
    net.train()
    epoch_loss = 0
    loss = criterion
    
    for (_, batch) in enumerate(train_ldr):
    
        signals = batch['signals']
        gt = batch['gt'].float()

        if signals.dim()==2:
            signals = signals[:,None].to(device) # For single channel signals
        else:
            signals = signals.to(device)
            
        targets = gt.permute((0,2,1)).to(device)
        targets_as_class = T.argmax(targets, dim=1)
        
        outputs = net(signals)
        if isinstance(criterion, nn.CrossEntropyLoss):
            ClassProbabilityChannels = 1.0*(outputs[0] - 0.0)
            # Note: this ONLY works for BCE loss
            L = loss(ClassProbabilityChannels, targets_as_class)
        
        elif isinstance(criterion, nn.MSELoss):
            Outputs = outputs[0]
            L = loss(Outputs, targets)

        epoch_loss += L.item()/len(train_ldr)
        
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        
    return epoch_loss

def dseg_val_epoch(net, val_ldr, criterion, device):
    
    net.eval()
    loss = criterion
    L_v = 0
    
    for (_, batch) in enumerate(val_ldr):
       
        signals = batch['signals']
        gt = batch['gt'].float() 
    
        # If there is no "batch" dimension (only one signal)...
        if signals.dim()==2: 
        # Appends a dummy dimension to signals, representing batch dimension
            signals = signals[:,None].to(device) 
        else:
            signals = signals.to(device)
    
#       signals = signals.to(device)
        targets = gt.permute((0,2,1)).to(device)
    
        targets_as_class = T.argmax(targets, dim=1)

        # Perhaps a bit of amplification (1.1) helps 
        # the targets be reached by the sigmoidal....? 
        outputs = net(signals)
        if isinstance(criterion, nn.CrossEntropyLoss):
            ClassProbabilityChannels = (outputs[0])
            # Note: this ONLY works for BCE loss
            L_v += loss(ClassProbabilityChannels, targets_as_class)
        
        elif isinstance(criterion, nn.MSELoss):
            Outputs = outputs[0]
            L_v += loss(Outputs, targets)

    return L_v