from dataloader import LeukemiaLoader
from ResNet import ResNet18, ResNet50, ResNet152
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data.sampler import WeightedRandomSampler

def evaluate(model, test_loader, device):
    model.eval()
    predict_result = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device, dtype=torch.float)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predict_result.append(predicted)
    predict_result = torch.cat(predict_result, dim=0)
    predict_result = predict_result.cpu().numpy()
    return predict_result

def train(model_type, model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device):
    train_accs = []
    valid_accs = []
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train(True)
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted_train = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted_train.eq(labels).sum().item()
        if epoch > 100:
            scheduler.step()

        train_acc = 100. * correct_train / total_train
        train_accs.append(train_acc)

        # Validation evaluation
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            for inputs, labels in valid_loader:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(inputs)
                _, predicted_val = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted_val.eq(labels).sum().item()

            valid_acc = 100. * correct_val / total_val
            valid_accs.append(valid_acc)
        epoch_end = time.time()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.2f}, Valid Accuracy: {valid_acc:.2f}, epoch time duration: {(epoch_end - epoch_start):.2f}s')
        # save model if validation accuracy is improved 
        if epoch == 0:
            best_acc = valid_acc
            torch.save(model.state_dict(), "./" + model_type + "_best_model.pth")
        else:
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), "./" + model_type + "_best_model.pth")
    return train_accs, valid_accs

def save_result(csv_path, predict_result, model_type):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./312554027_" + model_type + ".csv", index=False)

def plot_random_10_images(dataset):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        num = np.random.randint(0, len(dataset))
        image, label = dataset[num]
        # convert to image
        image = (image * 255).to(torch.uint8)
        plt.imshow(image.permute(1, 2, 0))
        plt.title(label)
        plt.xticks([])
        plt.yticks([])
    plt.savefig("./random_10_images.png")
    plt.close()

def get_weighted_sampler(train_dataset):
    target = np.array(train_dataset.label)
    class_sample_count = np.unique(target, return_counts=True)[1]
    print("class_sample_count: ", class_sample_count)
    weights = 1. / torch.Tensor(class_sample_count)
    samples_weight = weights[target]
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(train_dataset), replacement=True)
    return sampler

def plot_confusion_matrix(y_true, y_pred, title=None, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, square=True, cmap=cmap)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title + " confusion matrix")
    plt.savefig("./" + title + "_confusion_matrix.png")

def main(args):
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if args.model == "resnet18":
        model = ResNet18().to(device)
    elif args.model == "resnet50":
        model = ResNet50().to(device)
    elif args.model == "resnet152":
        model = ResNet152().to(device)
    else:
        raise ValueError("Invalid model name")
    
    if args.mode == "train":
        results = {}
        train_dataset = LeukemiaLoader(root=args.data_path, mode='train')
        valid_dataset = LeukemiaLoader(root=args.data_path, mode='valid')
        plot_random_10_images(train_dataset)
        # use weughtrandomsampler to balance the dataset
        sampler = get_weighted_sampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=12)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=12)
        if args.load_weight is not None:
            model.load_state_dict(torch.load(args.load_weight))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, last_epoch=-1)
        train_accs, valid_accs = train(args.model, model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs=args.epoch, device=device)
        print("Max valid accuracy: ", max(valid_accs))
        results["train_accs"] = train_accs
        results["valid_accs"] = valid_accs
        plt.plot(range(args.epoch), train_accs, label="train")
        plt.plot(range(args.epoch), valid_accs, label="valid")
        plt.legend()
        plt.savefig("./" + args.model + "_accs.png")
        # save acc per epoches to csv
        df = pd.DataFrame()
        df['epoch'] = range(args.epoch)
        df["train_accs"] = train_accs
        df["valid_accs"] = valid_accs
        df.to_csv("./" + args.model + "_accs.csv", index=False)

        # plot confusion matrix
        model.load_state_dict(torch.load("./" + args.model + "_best_model.pth"))
        model.eval()
        predict_result = []
        ground_truth = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                predict_result.append(predicted)
                ground_truth.append(labels)
        predict_result = torch.cat(predict_result, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)
        predict_result = predict_result.cpu().numpy()
        ground_truth = ground_truth.cpu().numpy()
        plot_confusion_matrix(ground_truth, predict_result, title=args.model)

    elif args.mode == "evaluate":
        if args.model == "resnet18":
            mode = "test18"
        elif args.model == "resnet50":
            mode = "test50"
        elif args.model == "resnet152":
            mode = "test152"
        else:
            raise ValueError("Invalid model name")
        test_dataset = LeukemiaLoader(root=args.data_path, mode=mode)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12)
        model.load_state_dict(torch.load("./" + args.model + "_best_model.pth"))
        
        predict_result = evaluate(model, test_loader, device)
        save_result(args.test_data_path, predict_result, args.model)
        
if __name__ == "__main__":
    # get arguments from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or evaluate")
    parser.add_argument("--model", type=str, default="resnet18", help="resnet18, resnet50 or resnet152")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--device", type=int, default=0, help="gpu number")
    parser.add_argument("--load_weight", type=str, default=None, help="load weight path")
    parser.add_argument("--test_data_path", type=str, default=None, help="test data path")
    parser.add_argument("--data_path", type=str, default='/home/linjohnss/NYCU_DL/lab3/dataset', help="data path")
    print("main.py: ", parser.parse_args())
    torch.manual_seed(20230808)
    np.random.seed(20230808)
    start = time.time()
    main(parser.parse_args())
    stop = time.time()
    print(f"time duration: {stop - start}s")
    print("Done!")