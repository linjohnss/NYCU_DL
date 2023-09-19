from dataLoader import dataloader
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import models.EEGNet
import models.DeepConvNet
import models.ShallowConvNet
from utils import utils
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, train_label, test_data, test_label = dataloader.read_bci_data()
# Convert the data to PyTorch TensorDataset
train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_label).long())
test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_label).long())
# Create DataLoader for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256,num_workers=4)

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    
    train_accs = []
    test_accs = []
    for epoch in range(epochs):
        model.train(True)
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            outputs = model(inputs)
            # l1 regularization
            l1_lambda = 0.001
            l1_regularization = torch.tensor(0., requires_grad=True)
            l1_regularization = l1_regularization.to(device, dtype=torch.float)
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1_regularization += torch.norm(param, p=1)
            # loss = cross entropy loss + l1 regularization
            loss = criterion(outputs, labels) + l1_lambda * l1_regularization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted_train = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted_train.eq(labels).sum().item()

        scheduler.step()

        train_acc = 100. * correct_train / total_train
        train_accs.append(train_acc)

        # Validation evaluation
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            for inputs, labels in test_loader:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(inputs)
                _, predicted_val = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted_val.eq(labels).sum().item()

            test_acc = 100. * correct_val / total_val
            test_accs.append(test_acc)
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}')
    
    return train_accs, test_accs

if __name__ == '__main__':
    activations = ['ReLU', 'LeakyReLU', 'ELU']
    results = {}
    for act in activations:
        print(f'Training models with {act} activation function...')
        # EEGNet model
        eeg_net = models.EEGNet.EEGNet(activation = act).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(eeg_net.parameters(), lr=0.005, weight_decay=0.001) # weight_decay is L2 regularization
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        eeg_train_accs, eeg_test_accs = train(eeg_net, train_loader, test_loader, criterion, optimizer, scheduler, epochs=800)

        # DeepConvNet model
        deep_conv_net =models.DeepConvNet.DeepConvNet(activation = act).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(deep_conv_net.parameters(), lr=0.005, weight_decay=0.001) # weight_decay is L2 regularization
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        deep_conv_train_accs, deep_conv_test_accs = train(deep_conv_net, train_loader, test_loader, criterion, optimizer, scheduler, epochs=800)

        results[act] = {
            'EEGNet_train': eeg_train_accs,
            'EEGNet_test': eeg_test_accs,
            'DeepConvNet_train': deep_conv_train_accs,
            'DeepConvNet_test': deep_conv_test_accs,
        }


    
    utils.plot_accuracy_trends(results, 'EEGNet')
    utils.plot_accuracy_trends(results, 'DeepConvNet')

    utils.plot_table(results)

    # ShallowConvNet model
    shallow_net = models.ShallowConvNet.ShallowConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(shallow_net.parameters(), lr=0.005, weight_decay=0.001) # weight_decay is L2 regularization
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    shallow_train_accs, shallow_test_accs = train(shallow_net, train_loader, test_loader, criterion, optimizer, scheduler, epochs=800)
    shallow_results = {
        'shallowNet_train': shallow_train_accs,
        'shallowNet_test': shallow_test_accs,
    }

    epochs = range(1, len(shallow_results['shallowNet_train']) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, shallow_results['shallowNet_train'], label='train')
    plt.plot(epochs, shallow_results['shallowNet_test'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('Acvation function comparison(ShallowConvNet)')
    plt.legend()
    plt.grid(True)
    plt.savefig('activation_ShallowConvNet.png')
    plt.show()
    print(max(shallow_results['shallowNet_train']))
    print(max(shallow_results['shallowNet_test']))
    