def train_epoch(epoch, train_ldr, criterion, optimizer, batch_size, scheduler):
    net.train()
    epoch_loss = 0
    loss = criterion

    for (batch_idx, batch) in enumerate(train_ldr):
        signals = batch['signals']
        gt = batch['gt'].float()

        if signals.dim() == 2:
            signals = signals[:, None].to(device)  # For single channel signals
        else:
            signals = signals.to(device)

        targets = gt.permute((0, 2, 1)).to(device)
        targets_as_class = T.argmax(targets, dim=1)

        outputs = net(signals)
        if isinstance(criterion, nn.CrossEntropyLoss):
            ClassProbabilityChannels = 1.0 * (outputs[0] - 0.0)
            L = loss(ClassProbabilityChannels, targets_as_class)

        elif isinstance(criterion, nn.MSELoss):
            Outputs = outputs[0]
            L = loss(Outputs, targets)

        epoch_loss += L.item() / len(train_ldr)

        optimizer.zero_grad()
        L.backward()
        optimizer.step()

    return epoch_loss


def val_epoch(nepoch, val_loader, batch_size, criterion):
    net.eval()
    loss = criterion
    L_v = 0

    for (batch_idx, batch) in enumerate(val_loader):
        signals = batch['signals']
        gt = batch['gt'].float()

        if signals.dim() == 2:
            signals = signals[:, None].to(device)
        else:
            signals = signals.to(device)

        targets = gt.permute((0, 2, 1)).to(device)
        targets_as_class = T.argmax(targets, dim=1)

        outputs = net(signals)
        if isinstance(criterion, nn.CrossEntropyLoss):
            ClassProbabilityChannels = (outputs[0])
            L_v += loss(ClassProbabilityChannels, targets_as_class)

        elif isinstance(criterion, nn.MSELoss):
            Outputs = outputs[0]
            L_v += loss(Outputs, targets)

    return L_v