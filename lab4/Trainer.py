import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size  
    return KLD

class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.args = args
        self.current_epoch = current_epoch
        self.beta = 0.0
        
    def update(self):
        if self.args.kl_anneal_type == 'Monotonic':
            self.beta += self.args.kl_anneal_ratio * 1 / self.args.kl_anneal_cycle
        elif self.args.kl_anneal_type == 'Cyclical':
            self.frange_cycle_linear(self.current_epoch, n_cycle=self.args.kl_anneal_cycle, ratio=self.args.kl_anneal_ratio)
        self.beta = min(self.beta, 1.0)
        self.current_epoch += 1
        
    def get_beta(self):
        if self.args.kl_anneal_type == 'without_anneal':
            return 1.0
        return self.beta
    
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        """
        Cyclical frange, a variant of learning rate annealing.
        """
        n_iter = (n_iter + 1) % n_cycle
        if n_iter == 0:
            self.beta = start
        else:
            self.beta += ratio * (stop - start) / n_cycle

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.kl_criterion = kl_criterion
        self.psnrs_criterion = Generate_PSNR
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        # list for plot
        self.losses_train = []
        self.teacher_forcing_ratios = []
        self.psnrs = []
        self.psnr_list = []
        self.mse_loss = []
        self.kld_loss = []
        self.bata_list = []
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            epoch_loss = 0
            epoch_mse = 0
            epoch_kld = 0
            train_loader = self.train_dataloader()
            # self.tfr = 0
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, mse, kld, beta  = self.training_one_step(img, label, adapt_TeacherForcing)
                epoch_loss += loss.detach().cpu()
                epoch_mse += mse.detach().cpu()
                epoch_kld += kld.detach().cpu()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()

            # store to list
            self.losses_train.append(epoch_loss / len(train_loader))
            self.teacher_forcing_ratios.append(self.tfr)
            self.bata_list.append(beta)
            self.mse_loss.append(epoch_mse / len(train_loader))
            self.kld_loss.append(epoch_kld / len(train_loader))

            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            self.plot_result()
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            psnr = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, psnr.detach().cpu(), lr=self.scheduler.get_last_lr()[0], psnr=psnr.detach().cpu())
        self.psnrs.append(psnr.detach().cpu())
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        output = []
        output.append(img[:, 0])
        mse = 0
        kld = 0
        self.label_transformation.zero_grad()
        self.frame_transformation.zero_grad()
        self.Gaussian_Predictor.zero_grad()
        self.Decoder_Fusion.zero_grad()
        self.Generator.zero_grad()
    
        for i in range(1, self.train_vi_len):
            if adapt_TeacherForcing:
                prev_img = img[:, i-1]
            else:
                prev_img = output[i-1].to(self.args.device)

            img_feature = self.frame_transformation(img[:, i])
            label_feature = self.label_transformation(label[:, i])
            z, mu, logvar = self.Gaussian_Predictor(img_feature, label_feature)
            
            prev_img_hat = self.frame_transformation(prev_img)
            parm = self.Decoder_Fusion(prev_img_hat, label_feature, z)
            out = self.Generator(parm)
            mse += self.mse_criterion(out, img[:, i])
            kld += self.kl_criterion(mu, logvar, self.batch_size)
            output.append(out)
        
        beta = self.kl_annealing.get_beta()
        loss = mse + beta * kld
        loss.backward()
        self.optimizer_step()
        return loss, mse, kld, beta
    
    def val_one_step(self, img, label):        
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        assert label.shape[0] == 630, "Testing pose seqence should be 630"
        assert img.shape[0] == 630, "Testing video seqence should be 630"
        
        decoded_frame_list = [img[0]]
        psnr = 0
        self.psnr_list.clear()
        out = img[0]
        
        for i in range(1, self.val_vi_len):
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_().to(self.args.device)
            label_feat = self.label_transformation(label[i])
            
            human_feat_hat = self.frame_transformation(out)
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)    
            out = self.Generator(parm)
            decoded_frame_list.append(out)            

        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        gt_frame = img.permute(1, 0, 2, 3, 4)
        for i in range(1, generated_frame.shape[1]):
            psnr_i = self.psnrs_criterion(gt_frame[0, i], generated_frame[0, i])
            psnr += psnr_i.cpu()
            self.psnr_list.append(psnr_i.cpu().numpy())
        # save image to gif
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f"epoch={self.current_epoch}_val.gif"))
        return psnr / (generated_frame.shape[1] - 1)
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch > self.args.tfr_sde:
            self.tfr -= self.args.tfr_d_step
            self.tfr = max(self.tfr, 0.0)  # ensuring tfr never goes below 0
            
    def tqdm_bar(self, mode, pbar, loss, lr, psnr=None):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        if mode == 'val':
            pbar.set_postfix(psnr=float(psnr), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

    def plot_result(self):
        # 1. Plot Teacher Forcing Ratio
        plt.figure()
        plt.plot(self.teacher_forcing_ratios, label='tfr')
        plt.title("Teacher Forcing Ratio over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Ratio")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(args.save_root, "tfr_plot.png"))
        plt.close()

        # 2. Plot Training
        plt.figure()
        plt.plot(self.losses_train, label='Training Loss')
        plt.title("Training and Validation Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(args.save_root, "loss_plot.png"))
        plt.close()

        # 3. Plot PSNR
        plt.figure()
        plt.plot(self.psnrs, label='PSNR')
        plt.title("Validation PSNR over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(args.save_root, "psnr_epoch_plot.png"))
        plt.close()

        # 4. Plot loss
        plt.figure()
        plt.plot(self.losses_train, label='Total Loss')
        plt.plot(self.mse_loss, label='MSE Loss')
        plt.plot(self.kld_loss, label='KLD Loss')
        plt.title("Loss over Steps")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(args.save_root, "total_loss_plot.png"))
        plt.close()

        # 5. Plot beta
        plt.figure()
        plt.plot(self.bata_list, label='beta')
        plt.title("beta over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("beta")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(args.save_root, "beta_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(self.bata_list, label='beta')
        plt.plot(self.teacher_forcing_ratios, label = 'tfr')
        plt.title("beta/tfr over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Ratio")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(args.save_root, "tfr_beta_plot.png"))
        plt.close()

        # 6. Plot PSNR
        plt.figure()
        plt.plot(self.psnr_list, label='Avg_PSNR: {:.2f}'.format(self.psnrs[-1]))
        plt.title("PSNR over Frame")
        plt.xlabel("Frame ndex")
        plt.ylabel("PSNR" )
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(args.save_root, "psnr_plot.png"))
        plt.close()

        # plot best psnr
        if self.psnrs[-1] == max(self.psnrs):
            plt.figure()
            plt.plot(self.psnr_list, label='Avg_PSNR: {:.2f}'.format(self.psnrs[-1]))
            plt.title("PSNR over Frame")
            plt.xlabel("Frame ndex")
            plt.ylabel("PSNR" )
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(args.save_root, "best_psnr_plot.png"))
            plt.close()

def main(args):
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=int,    default=0)
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=12)
    parser.add_argument('--num_epoch',     type=int, default=40,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=1,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=1,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.5,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=4,              help="")
    
    args = parser.parse_args()
    
    main(args)
