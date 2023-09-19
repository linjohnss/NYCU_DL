import os
import argparse
import torch
from model import ClassConditionedUnet
from dataloader import ICLEVRDataset
from evaluator import evaluation_model
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm.auto import tqdm


def train_one_epoch(args, model, noise_scheduler, train_loader, optimizer, criterion):
    model.train()
    losses = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        noise = torch.randn_like(images).to(args.device)
        timesteps = torch.randint(0, 1199, (images.shape[0],)).long().to(args.device)
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps).to(args.device)
        outputs = model(noisy_images, timesteps, labels)
        loss = criterion(outputs, noise)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses += loss.item()
    avg_loss = losses / len(train_loader)
    return avg_loss
    


def test(args, model, noise_scheduler, test_loader, writer):
    outputs_list = []
    labels_list = []
    progress_list = []
    model.eval()
    for i, (labels) in tqdm(enumerate(test_loader)):
        images = torch.randn(8, 3, 64, 64).to(args.device)
        labels = labels.to(args.device)
        for t in noise_scheduler.timesteps:
            t.to(args.device)
            with torch.no_grad():
                outputs = model(images, t, labels)
            images = noise_scheduler.step(outputs, t, images).prev_sample
            if i == 0 and t % 100 == 0:
                images_save = (images / 2 + 0.5).clamp(0, 1)
                progress_list.append(images_save[0])
        outputs_list.append(images)
        labels_list.append(labels)
    torchvision.utils.save_image(torch.stack(progress_list), os.path.join(args.logdir, 'progress.png'), nrow=10)
    return outputs_list, labels_list


def main(args):
    # Initialize model, loss function, and optimizer
    model = ClassConditionedUnet().to(args.device)
    evaluation = evaluation_model()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1200, beta_schedule='squaredcos_cap_v2')
    writer = SummaryWriter(args.logdir)
    # Train the model
    if args.mode == 'train':
        train_dataset = ICLEVRDataset(args.dataset, split='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_dataset = ICLEVRDataset(args.dataset, split='test')
        test_loader = DataLoader(test_dataset, batch_size=8, num_workers=args.num_workers)

        avg_loss_list = []
        test_acc_list = []
        for epoch in range(args.num_epoch):
            avg_loss = train_one_epoch(args, model, noise_scheduler, train_loader, optimizer, criterion)
            avg_loss_list.append(avg_loss)
            print(f'Epoch {epoch}/{args.num_epoch}. Average loss: {avg_loss:05f}')
            writer.add_scalar('Loss/avg', avg_loss, epoch)
            
            if (epoch + 1) % args.per_save == 0:
                torch.save(model.state_dict(), os.path.join(args.logdir, f'epoch_{epoch}.pth'))

            if epoch % args.per_test == 0:
                test_result, test_labels = test(args, model, noise_scheduler, test_loader, writer)
                test_result = torch.cat(test_result, dim=0)
                test_labels = torch.cat(test_labels, dim=0)
                images_save = (test_result / 2 + 0.5).clamp(0, 1)
                writer.add_images(f'test_images', images_save[:8], epoch + 1)
                test_acc = evaluation.eval(test_result, test_labels)
                test_acc_list.append(test_acc)
                print(f'Epoch {epoch}/{args.num_epoch}. Test accuracy: {test_acc:05f}')
                writer.add_scalar('Test/acc', test_acc, epoch)
        
            plt.plot(avg_loss_list)
            plt.savefig(os.path.join(args.logdir, 'loss.png'))
            plt.clf()
            plt.plot(test_acc_list)
            plt.savefig(os.path.join(args.logdir, 'acc.png'))
            plt.clf()
    
    else:
        test_dataset = ICLEVRDataset(args.dataset, split=args.mode)
        test_loader = DataLoader(test_dataset, batch_size=8, num_workers=args.num_workers)
        model.load_state_dict(torch.load(args.ckpt_path))
        test_result, test_labels = test(args, model, noise_scheduler, test_loader, writer)
        test_result = torch.cat(test_result, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        test_acc = evaluation.eval(test_result, test_labels)
        print(f'Test accuracy: {test_acc:05f}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=64)
    parser.add_argument('--lr',            type=float,  default=0.0001,     help="initial learning rate")
    parser.add_argument('--device',        type=str,    default='cuda:0')
    parser.add_argument('--mode',          type=str,    choices=["train", "test", "new_test"], default="train")
    parser.add_argument('--dataset',       type=str,    required=True,      help="Your Dataset Path")
    parser.add_argument('--num_workers',   type=int,    default=12)
    parser.add_argument('--num_epoch',     type=int,    default=300,            help="number of total epoch")
    parser.add_argument('--per_save',      type=int,    default=10,             help="Save checkpoint every seted epoch")
    parser.add_argument('--per_test',      type=int,    default=10,             help="Test every seted epoch")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    parser.add_argument('--logdir',        type=str,    default='log/test')
    args = parser.parse_args()
    
    main(args)
