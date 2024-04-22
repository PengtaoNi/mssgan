import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform

import utils
from data import InstDataset
from models.Unet import Unet
from models.ConvDiscriminator import ConvDiscriminator

def train(opt):
    utils.set_seeds(opt)
    device = utils.get_device()
    
    dataset = InstDataset(opt.dataset_path, opt.input_w)
    dataloader = DataLoader(dataset, num_workers=2, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    noise_dist = Uniform(torch.Tensor([-1] * opt.z_dim * opt.batch_size), torch.Tensor([1] * opt.z_dim * opt.batch_size))

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
    print('Start training...')
    for epoch in range(opt.epochs):
        G_loss_total = 0
        D1_loss_total = 0
        D2_loss_total = 0

        for i, sample in enumerate(dataloader):
            inst1 = sample[:, 0:1].to(device)
            inst2 = sample[:, 1:2].to(device)
            mixture = sample[:, 2:3].to(device)
            noise = noise_dist.sample().to(device)

            fake = G([mixture, noise])
            fake1 = fake[:, 0:1]
            fake2 = fake[:, 1:2]

            # train discriminator
            # real_label = torch.full((inst1.size(0),), 1.0, device=device)
            # fake_label = torch.full((inst1.size(0),), 0.0, device=device)

            D1_optimizer.zero_grad()
            D2_optimizer.zero_grad()
            
            D1_real = D1(inst1)
            D1_fake = D1(fake1)
            D2_real = D2(inst2)
            D2_fake = D2(fake2)

            D1_real_loss = criterion(D1_real, torch.full((inst1.size(0),), 1.0, device=device))
            D1_fake_loss = criterion(D1_fake, torch.full((inst1.size(0),), 0.0, device=device))
            D2_real_loss = criterion(D2_real, torch.full((inst1.size(0),), 1.0, device=device))
            D2_fake_loss = criterion(D2_fake, torch.full((inst1.size(0),), 0.0, device=device))

            D1_loss = (D1_real_loss + D1_fake_loss) / 2
            D2_loss = (D2_real_loss + D2_fake_loss) / 2

            D1_loss.backward(retain_graph=True)
            D2_loss.backward()
            D1_optimizer.step()
            D2_optimizer.step()

            # train generator
            G_optimizer.zero_grad()

            generated = G([mixture, noise])
            generated1 = generated[:, 0:1]
            generated2 = generated[:, 1:2]

            D1_generated = D1(generated1)
            D2_generated = D2(generated2)

            G_loss = (criterion(D1_generated, torch.full((inst1.size(0),), 1.0, device=device)) +
                      criterion(D2_generated, torch.full((inst1.size(0),), 1.0, device=device))) / 2
            
            G_loss.backward()
            G_optimizer.step()

            G_loss_total += G_loss.item()
            D1_loss_total += D1_loss.item()
            D2_loss_total += D2_loss.item()
        
        G_loss_avg.append(G_loss_total / len(dataloader))
        D1_loss_avg.append(D1_loss_total / len(dataloader))
        D2_loss_avg.append(D2_loss_total / len(dataloader))
        print(f"Epoch: {epoch}, G loss: {G_loss_avg[-1]}, \
                D1 loss: {D1_loss_avg[-1]}, D2 loss: {D2_loss_avg[-1]}")
    
    os.makedirs(opt.model_path, exist_ok=True)
    model_path = os.path.join(opt.model_path, "G_" + str(epoch))
    print("Saving generator at " + model_path)
    torch.save(G.state_dict(), model_path)

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='dataset')
    parser.add_argument('--model_path', type=str, default='G')

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
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

    train(opt)