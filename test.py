import torch
import torch.nn
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys, os
import math
import matplotlib.pyplot as plt
from scipy import stats
from models.resnet_diffusion import Diffusion
from dataset import Dataset3D
from model import generate_model


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_threshold_auc(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    youden_index = np.argmax(tpr - fpr)
    return roc_auc, threshold[youden_index], youden_index



def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return acc, sensitivity, specificity, ppv, npv

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
                                                      # batch_size=16,
                                                      shuffle=True,
                                                      pin_memory=True)
    dataloader['val'] = torch.utils.data.DataLoader(valset,
                                                    batch_size=len(valset),
                                                    shuffle=False,
                                                    pin_memory=True)

    dataloader['test'] = torch.utils.data.DataLoader(testset,
                                                     batch_size=len(testset),
                                                     shuffle=False,
                                                     pin_memory=True)
    num_classes = 2


    model0 = generate_model(no_cuda=False, gpu_id=[0], nb_class=num_classes)
    net0 = model0.to(device)
    check_point = torch.load(r'./result/0.pt')
    net0.load_state_dict(check_point['model_state_dict'])
    net0.eval()

    model1 = generate_model(no_cuda=False, gpu_id=[0], nb_class=num_classes)
    net1 = model1.to(device)
    check_point = torch.load(r'./result/1.pt')
    net1.load_state_dict(check_point['model_state_dict'])
    net1.eval()

    model2 = generate_model(no_cuda=False, gpu_id=[0], nb_class=num_classes)
    net2 = model2.to(device)
    check_point = torch.load(r'./result/2.pt')
    net2.load_state_dict(check_point['model_state_dict'])
    net2.eval()



    models = [net0, net1, net2]



    dataloader_tr = dataloader['train']
    dataloader_val = dataloader['val']
    dataloader_test = dataloader['test']

    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(dataloader_val):
            X_val, y_val = data
            X = X_val.to(device)
            y = y_val.to(device)
            for j, model in enumerate(models):
                pred, feats = model(X)
                score = nn.functional.softmax(pred, dim=1).cpu().detach().numpy()
                y_score = score[:, 1]

                y_np = y.cpu().numpy()
                y_np = np.where(y_np == j, 1, 0)


                _, cut_off, _ = get_threshold_auc(y_np, y_score)
                print(j, 'cutoff', cut_off)

                pred = (np.array(y_score) > cut_off).astype(int)

                _, sensitivity, specificity, ppv, npv = calc_metrics(y_np, pred)

                print(j, 'sen', sensitivity)

                if j == 2:
                    pred[pred == 1] = 2  # 将预测为1的置为2
                    pred[pred == 0] = 3  # 将预测为0的置为3
                    y_pred.extend(pred)
                    y_true.extend(y.cpu().numpy())
                else:
                    pos_mask = (pred == 1)
                    if pos_mask.sum() > 0:
                        y_true.extend(y[pos_mask].cpu().numpy())
                        if j == 0:
                            y_pred.extend([0] * pos_mask.sum())  # 只保存预测为1的样本的预测值
                        else:
                            y_pred.extend([1] * pos_mask.sum())  # 只保存预测为1的样本的预测值
                    if pred.sum() > 0:  # 确保有阴性样本
                        mask = (pred == 0)
                        X_val = X_val[mask]
                        y_val = y_val[mask]
                        X = X_val.to(device)
                        y = y_val.to(device)
                print(j, y_true, y_pred)

    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')  # 对于二分类
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')




    # cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    #
    # plt.figure(figsize=(5, 5))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    #             annot_kws={"size": 12},
    #             xticklabels=['ccRCC', 'AML', 'pRCC', 'chRCC'],
    #             yticklabels=['ccRCC', 'AML', 'pRCC', 'chRCC'])
    #
    # plt.gca().xaxis.set_ticks_position('top')  # 设置 xticklabels 在顶部
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.xlabel('Predicted Label', fontsize=14, labelpad=8)
    # plt.ylabel('True Label', fontsize=14, labelpad=8)
    #
    # plt.show()
    # plt.savefig('heatmap.png', dpi=300)




def plt_roc():
    val_df1 = pd.read_csv(r'D:\gyji\subtype-classify\score_results\0_score.csv')
    val_true1 = val_df1['label']

    print(val_true1)
    val_score1 = val_df1['score']
    val_fpr1, val_tpr1, ths1 = roc_curve(val_true1, val_score1)
    val_auc1 = auc(val_fpr1, val_tpr1)
    print(val_fpr1, val_tpr1)

    val_df2 = pd.read_csv(r'D:\gyji\subtype-classify\score_results\1_score.csv')
    val_true2 = val_df2['label']
    val_score2 = val_df2['score']
    val_fpr2, val_tpr2, ths2 = roc_curve(val_true2, val_score2)
    val_auc2 = auc(val_fpr2, val_tpr2)
    print(val_fpr2, val_tpr2)

    val_df3 = pd.read_csv(r'D:\gyji\subtype-classify\score_results\2_score.csv')
    val_true3 = val_df3['label']
    val_score3 = val_df3['score']
    val_fpr3, val_tpr3, ths3 = roc_curve(val_true3, val_score3)
    val_auc3 = auc(val_fpr3, val_tpr3)
    print(val_fpr3, val_tpr3)

    val_df4 = pd.read_csv(r'D:\gyji\subtype-classify\score_results\3_score.csv')
    val_true4 = val_df4['label']
    val_score4 = val_df4['score']
    val_fpr4, val_tpr4, ths4 = roc_curve(val_true4, val_score4, pos_label=1)
    val_auc4 = auc(val_fpr4, val_tpr4)
    print(val_fpr4, val_tpr4)

     # 绘制y=x
    x = np.linspace(0, 1.0, 50)
    y = x

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.xlim((0, 1.05))
    plt.ylim((0, 1.05))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.rc('legend', fontsize=20)
    ax.set_xlabel('x', fontdict={'fontsize': 28})
    ax.set_ylabel('y', fontdict={'fontsize': 28})

    ax.plot(val_fpr1, val_tpr1, linewidth=4, color='#f76b61',
            label='ccRCC' + ' (auc=' + str("{:.2f}".format(val_auc1 * 100)) + '%)')
    ax.plot(val_fpr2, val_tpr2, linewidth=4, color='#0072B5',
            label='AML   ' + ' (auc=' + str("{:.2f}".format(val_auc2 * 100)) + '%)')
    ax.plot(val_fpr3, val_tpr3, linewidth=4, color='#99EE99',
            label='pRCC ' + ' (auc=' + str("{:.2f}".format(val_auc3 * 100)) + '%)')
    ax.plot(val_fpr4, val_tpr4, linewidth=4, color='#B38015',
            label='chRCC' + ' (auc=' + str("{:.2f}".format(val_auc4 * 100)) + '%)')


    ax.plot(x, y, color='gray', linestyle='--', linewidth=2)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.legend(loc="lower right")
    ax.set_xlabel('1-specificity')
    ax.set_ylabel('sensitivity')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.savefig('roc.png', dpi=300)




def calc_ci(arr):
    arr_sem = stats.sem(arr)
    arr_mean = np.mean(arr)
    ci = stats.t.interval(0.95, df=len(arr)-1, loc=arr_mean, scale=arr_sem)
    return format(arr_mean, '.4f'), [format(ci[0], '.4f'), format(ci[1], '.4f')]

def stastic_test(models, dataloader):

    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            X_val, y_val = data
            X = X_val.to(device)
            y = y_val.to(device)
            for j, model in enumerate(models):
                pred, feats = model(X)
                score = nn.functional.softmax(pred, dim=1).cpu().detach().numpy()
                y_score = score[:, 1]

                y_np = y.cpu().numpy()
                y_np = np.where(y_np == j, 1, 0)


                _, cut_off, _ = get_threshold_auc(y_np, y_score)
                print(j, 'cutoff', cut_off)

                pred = (np.array(y_score) > cut_off).astype(int)

                _, sensitivity, specificity, ppv, npv = calc_metrics(y_np, pred)

                print(j, 'sen', sensitivity)

                if j == 2:
                    pred[pred == 1] = 2  # 将预测为1的置为2
                    pred[pred == 0] = 3  # 将预测为0的置为3
                    y_pred.extend(pred)
                    y_true.extend(y.cpu().numpy())
                else:
                    pos_mask = (pred == 1)
                    if pos_mask.sum() > 0:
                        y_true.extend(y[pos_mask].cpu().numpy())
                        if j == 0:
                            y_pred.extend([0] * pos_mask.sum())  # 只保存预测为1的样本的预测值
                        else:
                            y_pred.extend([1] * pos_mask.sum())  # 只保存预测为1的样本的预测值



                    if pred.sum() > 0:  # 确保有阴性样本
                        mask = (pred == 0)
                        X_val = X_val[mask]
                        y_val = y_val[mask]
                        X = X_val.to(device)
                        y = y_val.to(device)
                print(j, y_true, y_pred)

    # 计算各项指标
    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred, average='weighted')  # 对于二分类
    # recall = recall_score(y_true, y_pred, average='weighted')
    # f1 = f1_score(y_true, y_pred, average='weighted')

    # print(f'Accuracy: {accuracy}')
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for i in range(1000):
        # 有放回采样
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        sampled_labels = y_true[indices]
        sampled_predictions = y_pred[indices]
        accuracy = accuracy_score(sampled_labels, sampled_predictions)
        precision = precision_score(sampled_labels, sampled_predictions, average='macro')  # 对于二分类
        recall = recall_score(sampled_labels, sampled_predictions, average='macro')
        f1 = f1_score(sampled_labels, sampled_predictions, average='macro')
        acc_list.append(accuracy)
        pre_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)


    acc_mean, acc_ci = calc_ci(acc_list)
    pre_mean, pre_ci = calc_ci(pre_list)
    rec_mean, rec_ci = calc_ci(rec_list)
    f1_mean, f1_ci = calc_ci(f1_list)

    print(
        'acc={}, pre={}, rec={}, f1={}'
        .format(acc_mean, pre_mean, rec_mean, f1_mean))
    print(
        'accci={}, preci={}, recci={}, f1ci={}'
        .format(acc_ci, pre_ci, rec_ci, f1_ci))


if __name__ == '__main__':
    main()
    # plt_roc()