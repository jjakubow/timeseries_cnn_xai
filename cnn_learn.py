from tqdm import tqdm
import torch

def fit(model, data_loader, device, optimizer, loss_function):
    running_loss = .0
    model.train()
    
    for idx, (inputs, labels) in tqdm(enumerate(data_loader), total=data_loader.__len__(), disable=True):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        #print(inputs.float()[:, 0].shape )
        
        preds = model(inputs.float())[:, 0]        
        loss = loss_function(preds ,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(data_loader)
    train_loss = train_loss.detach().numpy()
    return train_loss

def validate(model, data_loader, device, optimizer, loss_function):
    running_loss = .0
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())[:, 0]
            loss = loss_function(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(data_loader)
        valid_loss = valid_loss.detach().numpy()
        
        return valid_loss


def run_training(model, optimizer, loss_function, train_loader, test_loader, epochs=1, device_name="cpu", min_loss_decrease=1e-5, epochs_wait_max=5):
    device = torch.device(device_name)
    
    train_losses = []
    valid_losses = []

    # iniatlize loss to extremely high value (for early stopping)
    current_loss = 1e9

    t = tqdm(range(epochs), desc='Training for %i epochs' % epochs, leave=True)

    for epoch in t:
        # train
        train_loss = fit(model, train_loader, device, optimizer, loss_function)
        train_losses.append(train_loss)
        # validate
        valid_loss = validate(model, test_loader, device, optimizer, loss_function)
        valid_losses.append(valid_loss)
        
        # the code below ensures that we do not overfit the model
        if (current_loss - valid_loss) < min_loss_decrease:
            epochs_wait +=1
        else:
            current_loss = valid_loss
            epochs_wait = 0
            
        if epochs_wait == epochs_wait_max:
            print("Not enough progress in last %i epochs, end of training." % epochs_wait_max)
            break
            
        # update progress bar
        t.set_description("Current Loss: train = %.3g, validation = %.3g)" % (train_loss, valid_loss))
        t.refresh() 

    return train_losses, valid_losses

