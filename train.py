import torch
import torch.nn
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import os
from model import generate_model
from dataset import Dataset3D
from utils import DCL_Loss


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_threshold_auc(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    youden_index = np.argmax(tpr - fpr)
    return roc_auc, threshold[youden_index], youden_index


def main():
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = Dataset3D(r'./data/train.xlsx', train_transform)

    valset = Dataset3D(r'./data/val.xlsx', transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize,
    ]))
    testset = Dataset3D(r'./data/test.xlsx', transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            normalize,
    ]))

    dataloader = dict()

    dataloader['train'] = torch.utils.data.DataLoader(trainset,
                                                      batch_size=16,
                                                      shuffle=True,
                                                      pin_memory=True)
    dataloader['val'] = torch.utils.data.DataLoader(valset,
                                                    batch_size=16,
                                                    shuffle=False,
                                                    pin_memory=True)

    dataloader['test'] = torch.utils.data.DataLoader(testset,
                                                     batch_size=16,
                                                     shuffle=False,
                                                     pin_memory=True)

    num_classes = 2


    model = generate_model(no_cuda=False, gpu_id=[0],
                           pretrain_path='./pretrain/pretrain.pth',
                           nb_class=num_classes)

    net = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss()

    dataloader_tr = dataloader['train']
    dataloader_val = dataloader['val']

    save_path = 'result/'


    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    train_auc = []
    val_auc = []



    for epoch in range(200):

        avg_loss_value = train(net, dataloader_tr, optimizer, criterion, epoch + 1)
        _, train_acc_value, train_auc_value = val(net, dataloader_tr)
        val_loss_value, val_acc_value, val_auc_value = val(net, dataloader_val)

        train_loss.append(avg_loss_value)
        val_loss.append(val_loss_value)

        train_acc.append(train_acc_value)
        val_acc.append(val_acc_value)


        train_auc.append(train_auc_value)
        val_auc.append(val_auc_value)

        if len(val_auc) == 1:
            torch.save({'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), },
                       os.path.join(save_path, 'first_model' + '.pt'))
            df_acc = pd.DataFrame([],
                                  columns=['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_auc', 'val_auc'])
            df_acc['train_loss'] = train_loss
            df_acc['val_loss'] = val_loss
            df_acc['train_acc'] = train_acc
            df_acc['val_acc'] = val_acc
            df_acc['train_auc'] = train_auc
            df_acc['val_auc'] = val_auc
            df_acc.to_csv(os.path.join(save_path, 'first_result.csv'))
        else:
            if (val_auc[-1] > np.max(val_auc[:-1])):
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join(save_path, 'best_val_auc_model' + '.pt'))
                df_acc = pd.DataFrame([],
                                      columns=['train_loss', 'val_loss', 'train_acc', 'val_acc','train_auc', 'val_auc'])
                df_acc['train_loss'] = [train_loss[-1]]
                df_acc['val_loss'] = [val_loss[-1]]
                df_acc['train_acc'] = [train_acc[-1]]
                df_acc['val_acc'] = [val_acc[-1]]
                df_acc['train_auc'] = [train_auc[-1]]
                df_acc['val_auc'] = [val_auc[-1]]
                df_acc.to_csv(os.path.join(save_path, 'best_val_auc_result.csv'))

        torch.save({'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), },
                   os.path.join(save_path, 'final_model' + '.pt'))
        df_acc = pd.DataFrame([],
                              columns=['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_auc', 'val_auc'])
        df_acc['train_loss'] = train_loss
        df_acc['val_loss'] = val_loss
        df_acc['train_acc'] = train_acc
        df_acc['val_acc'] = val_acc
        df_acc['train_auc'] = train_auc
        df_acc['val_auc'] = val_auc
        df_acc.to_csv(os.path.join(save_path, 'final_result.csv'))
        torch.cuda.empty_cache()

def train(net, dataloader, optimizer, criterion, epoch):
    torch.cuda.empty_cache()
    net.train()
    loss_epoch = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_batch = len(dataloader.dataset) // 16
    pbar = tqdm(total=n_batch)
    dcl_loss = DCL_Loss()

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        diffusion = Diffusion(noise_steps=50,  device=device)
        time_step = diffusion.sample_timesteps(inputs.shape[0]).to(device)

        inputs, noise = diffusion.noise_images(inputs, time_step)
        outputs, feats = net(inputs, time_step)
        cross_loss = criterion(outputs, labels)
        con_loss = dcl_loss(feats, labels)
        loss = cross_loss + con_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch.append(float(loss.data.cpu().numpy()))
        pbar.update(1)

    pbar.close()
    avg_loss = np.mean(np.array(loss_epoch))
    return avg_loss

def val(net, dataloader):
    net.eval()
    loss_epoch = []
    n_batch = len(dataloader.dataset) // 16
    pbar = tqdm(total=n_batch)
    y_true = []
    y_score = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcl_loss = DCL_Loss()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            X_val, y_val = data
            X = X_val.to(device)
            y = y_val.to(device)
            pred, feats = net(X)

            cross_loss = nn.CrossEntropyLoss()(pred, y)
            con_loss = dcl_loss(feats, y)
            loss = cross_loss + con_loss

            score = nn.functional.softmax(pred, dim=1).cpu().detach().numpy()
            y_true.extend(y_val)
            y_score.extend(score[:, 1])
            loss_epoch.append(float(loss.data.cpu().numpy()))
            pbar.update(1)

    pbar.close()
    loss_value = np.mean(np.array(loss_epoch))
    roc_auc, best_threshold, _ = get_threshold_auc(y_true, y_score)
    y_pred = (np.array(y_score) > best_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    print(acc, roc_auc, best_threshold, loss_value)

    return loss_value, acc, roc_auc

if __name__ == '__main__':
    main()

