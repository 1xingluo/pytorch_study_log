import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.config import Config
import torch.nn.functional as F
from configs.seed import same_seeds
from dataset.predata import get_dataset
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.autograd import Variable
import os
from qqdm import qqdm
from model.model_2 import Generator,Discriminator
import torch.nn as nn
from torch.utils.data import  DataLoader
cfg=Config()
same_seeds(2021)

workspace_dir =cfg.workspace_dir
dataset = get_dataset(os.path.join(workspace_dir, 'faces'))

images = [(dataset[i]+1)/2 for i in range(16)]
grid_img = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()

# Training hyperparameters
batch_size = cfg.batch_size
z_dim = cfg.z_dim
z_sample = cfg.z_sample
lr = cfg.lr

""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
n_epoch =cfg.n_epoch # 50
n_critic = cfg.n_critic # 5
# clip_value = 0.01

log_dir = cfg.log_dir
ckpt_dir = cfg.ckpt_dir

# Model
G = Generator(in_dim=z_dim).cuda()
D = Discriminator(3).cuda()
G.train()
D.train()

# Loss
criterion = nn.BCELoss()

""" Medium: Use RMSprop for WGAN. """
# Optimizer
# opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
# opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
# opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)
#SNGAN
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))  # ✅ Adam + SNGAN betas
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))

# DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


steps = 0
for e, epoch in enumerate(range(n_epoch)):
    progress_bar = qqdm(dataloader)
    for i, data in enumerate(progress_bar):
        imgs = data
        imgs = imgs.cuda()

        bs = imgs.size(0)

        # ============================================
        #  Train D
        # ============================================
        z = Variable(torch.randn(bs, z_dim)).cuda()
        r_imgs = Variable(imgs).cuda()
        f_imgs = G(z)

        """ Medium: Use WGAN Loss. """
        # Label
        # r_label = torch.ones((bs)).cuda()
        # f_label = torch.zeros((bs)).cuda()

        # Model forwarding
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())
        
        # Compute the loss for the discriminator.
        # r_loss = criterion(r_logit, r_label)
        # f_loss = criterion(f_logit, f_label)
        # loss_D = (r_loss + f_loss) / 2

        # WGAN Loss
        # loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))
        #SNGAN
        loss_D = torch.mean(F.relu(1.0 - r_logit)) + torch.mean(F.relu(1.0 + f_logit))


        # Model backwarding
        D.zero_grad()
        loss_D.backward()

        # Update the discriminator.
        opt_D.step()

        """ Medium: Clip weights of discriminator. """
        # for p in D.parameters():
        #    p.data.clamp_(-clip_value, clip_value)

        # ============================================
        #  Train G
        # ============================================
        
        # Generate some fake images.
        z = Variable(torch.randn(bs, z_dim)).cuda()
        f_imgs = G(z)

        # Model forwarding
        f_logit = D(f_imgs)
            
        """ Medium: Use WGAN Loss"""
        # Compute the loss for the generator.
        # loss_G = criterion(f_logit, r_label)
        # WGAN Loss
        # loss_G = -torch.mean(D(f_imgs))
        loss_G = -torch.mean(f_logit)
        # Model backwarding
        G.zero_grad()
        loss_G.backward()

        # Update the generator.
        opt_G.step()

        steps += 1
        
        # Set the info of the progress bar
        #   Note that the value of the GAN loss is not directly related to
        #   the quality of the generated images.
        progress_bar.set_infos({
            'Loss_D': round(loss_D.item(), 4),
            'Loss_G': round(loss_G.item(), 4),
            'Epoch': e+1,
            'Step': steps,
        })

    G.eval()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(log_dir, f'Epoch_{epoch+1:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')
    
    # Show generated images in the jupyter notebook.
    # grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    # plt.figure(figsize=(10,10))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # plt.show()
    G.train()

    if (e+1) % 5 == 0 or e == 0:
        # Save the checkpoints.
        torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
