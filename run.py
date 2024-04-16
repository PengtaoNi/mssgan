import os
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform

import utils
from data import InstDataset
from models.Unet import Unet
from models.ConvDiscriminator import ConvDiscriminator

def train(opt):
    utils.set_seeds(opt)
    device = utils.get_device(opt.cuda)
    
    dataset = InstDataset(opt.dataset_path)
    dataloader = DataLoader(dataset, num_workers=2, batch_size=20, shuffle=True)

    noise_dist = Uniform(torch.Tensor([-1] * opt.z_dim), torch.Tensor([1] * opt.z_dim))

    G = Unet(opt).to(device)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=opt.lr,
                                   betas=(0.5, 0.999), weight_decay=0.0)
    D1 = ConvDiscriminator(opt).to(device)
    D1_optimizer = torch.optim.Adam(D1.parameters(), lr=opt.lr,
                                   betas=(0.5, 0.999), weight_decay=0.0)
    D2 = ConvDiscriminator(opt).to(device)
    D2_optimizer = torch.optim.Adam(D2.parameters(), lr=opt.lr,
                                   betas=(0.5, 0.999), weight_decay=0.0)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    G_loss_avg = []
    D1_loss_avg = []
    D2_loss_avg = []
    for epoch in range(opt.epochs):
        G_loss_total = 0
        D1_loss_total = 0
        D2_loss_total = 0

        for i, (insts, mixture) in enumerate(dataloader):
            inst1 = insts[0].to(device)
            inst2 = insts[1].to(device)
            mixture = mixture.to(device)

            noise = noise_dist.sample().to(device)
            fake = G([mixture, noise])
            fake1 = fake[0, 1]
            fake2 = fake[0, 2]

            # train discriminator
            real_label = torch.full((inst1.size(0),), 1, device=device)
            fake_label = torch.full((inst1.size(0),), 0, device=device)

            D1_optimizer.zero_grad()
            D2_optimizer.zero_grad()
            
            D1_real = D1(inst1)
            D2_real = D2(inst2)
            D1_fake = D1(fake1)
            D2_fake = D2(fake2)

            D1_real_loss = criterion(D1_real, real_label)
            D1_fake_loss = criterion(D1_fake, fake_label)
            D2_real_loss = criterion(D2_real, real_label)
            D2_fake_loss = criterion(D2_fake, fake_label)

            D1_loss = (D1_real_loss + D1_fake_loss) / 2
            D2_loss = (D2_real_loss + D2_fake_loss) / 2

            D1_loss.backward()
            D2_loss.backward()
            D1_optimizer.step()
            D2_optimizer.step()

            # train generator
            G_optimizer.zero_grad()

            generated = G([mixture, noise])
            generated1 = generated[0, 1]
            generated2 = generated[0, 2]

            D1_fake = D1(generated1)
            D2_fake = D2(generated2)

            G_loss = (criterion(D1_fake, real_label) +
                      criterion(D2_fake, real_label)) / 2
            
            G_loss.backward()
            G_optimizer.step()

            G_loss_total += G_loss.item()
            D1_loss_total += D1_loss.item()
            D2_loss_total += D2_loss.item()
        
        G_loss_avg.append(G_loss_total / len(dataloader))
        D1_loss_avg.append(D1_loss_total / len(dataloader))
        D2_loss_avg.append(D2_loss_total / len(dataloader))
        print(f"Epoch: {epoch}, G loss: {G_loss_avg[-1]},
                D1 loss: {D1_loss_avg[-1]}, D2 loss: {D2_loss_avg[-1]}")
        
    output_path = os.path.join(opt.output_path, "G_" + str(epoch))
    print("Saving generator at " + output_path)
    torch.save(G.state_dict(), output_path)

def eval(opt):
    utils.set_seeds(opt)
    device = utils.get_device(opt.cuda)

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--output_path', type=str)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--z_dim', type=int, default=50)

    parser.add_argument('--win_length', type=int, default=512)
    parser.add_argument('--hop_length', type=int, default=256)

    opt = parser.parse_args()
    print(opt)

    opt.input_h = opt.win_length // 2
    opt.input_w = opt.input_h // 2

    return opt

if __name__ == "__main__":
    opt = get_opt()

    if not opt.eval:
        train(opt)
    eval(opt)