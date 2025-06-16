import torch
from torch import nn, optim
import os
from pathlib import Path
from time import time
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
def evaluate_accuracy_frame( model, val_loader, device ):
    """验证模型预测结果的WA、UA与混淆矩阵"""

    model.eval()
    num_val = len( val_loader.dataset )
    acc = 0
    mat = np.zeros( (4,4) )
    class_num = np.zeros( 4 )
    acc_num = np.zeros( 4 )
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            ids,net_input,labels = batch["id"],batch["net_input"],batch["labels"]
            text_feat = net_input["text_feats"].to( device )
            audio_feat = net_input["audio_feats"].to( device )
            text_padding_mask = net_input["text_padding_mask"].to( device )
            audio_padding_mask = net_input["audio_padding_mask"].to( device )
            true = int( labels.item() ) 
            class_num[true] += 1
            output,outa1, outa2,attout1,attout2,outs,outt= model(text_feat, audio_feat)
            pred = int( torch.argmax( output, dim = -1 ).item() )
            all_preds.append(pred)
            all_labels.append(true)
            mat[true, pred] += 1 
            if true == pred:
                acc += 1
                acc_num[pred] += 1
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(4)), zero_division=0)
    wf1 = np.average(f1, weights=class_num)
    return acc / num_val, np.mean( acc_num / class_num ),  wf1, mat

# 验证集损失
def evaluate_loss_frame(model, val_loader, device):
    """计算验证集上的损失"""
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in val_loader:
            ids,net_input,labels = batch["id"],batch["net_input"],batch["labels"]
            text_feat = net_input["text_feats"].to( device )
            audio_feat = net_input["audio_feats"].to( device )
            text_padding_mask = net_input["text_padding_mask"].to( device )
            audio_padding_mask = net_input["audio_padding_mask"].to( device )
            labels = labels.to(device)
            output,outa1, outa2,attout1,attout2,outs,outt= model(text_feat, audio_feat)
            loss = criterion(output, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_one_fold_frame( logger,save_dir,fold,model, train_loader, val_loader,test_loader,
                optimizer, scheduler, device,cfg,continue_training=False):
    """模型训练，在每个epoch报道验证集和验证集的结果,在每折报告测试的结果"""

    best_val_wa = 0
    best_val_wa_ua = 0 
    best_val_wa_epoch = 0
    save_dir = os.path.join(str(Path.cwd()), f"{save_dir}/model_{fold}.pth")
    epoch = cfg["train"]["epoch"]
    if continue_training and os.path.exists(save_dir):
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_wa = checkpoint['best_val_wa']
        best_val_wa_ua = checkpoint['best_val_wa_ua']
        best_val_wa_epoch = checkpoint['best_val_wa_epoch']
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Continuing training from epoch {start_epoch}")
    else:
        start_epoch = 1

    for epoch in range( start_epoch, epoch + 1 ):
        torch.autograd.set_detect_anomaly(True)
        model.train()
        train_l_sum, train_acc_sum = 0, 0 
        start = time()
        for batch in train_loader:
            ids,net_input,labels = batch["id"],batch["net_input"],batch["labels"]
            text_feat = net_input["text_feats"]
            audio_feat = net_input["audio_feats"]
            text_padding_mask = net_input["text_padding_mask"]
            audio_padding_mask = net_input["audio_padding_mask"]
            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)
            text_padding_mask = text_padding_mask.to(device)
            audio_padding_mask = audio_padding_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output,outa1, outa2,outa11,outa22,outs,outt= model(text_feat, audio_feat)
            MSE = nn.MSELoss()
            cmd_loss_fn1 = CMDLoss(K=5)
            cmd_loss_fn2 = CMDLoss(K=5)
            loss_ts = cmd_loss_fn1(outa1, outa22)
            loss_st = cmd_loss_fn2(outa2, outa11)
            loss_p = F.cross_entropy( output, labels )
            loss_st = MSE(outa1, outa22)
            loss_ts = MSE(outa2, outa11)
            loss_s = F.cross_entropy(outs, labels)
            loss_t = F.cross_entropy(outt, labels)
            loss = 0.1*loss_p + 0.3*loss_st + 0.3*loss_ts + 0.15*loss_s + 0.15*loss_t
            if np.isnan( loss.item() ): 
                logger.info( "Error: loss diverges." )
                return
            loss.backward()
            optimizer.step()
            train_l_sum += loss.item()
            acc_num = ( torch.argmax( output, dim = 1 ) == labels ).sum().item()
            train_acc_sum += acc_num
        if scheduler: 
            scheduler.step()
        train_acc_sum /= len( train_loader.dataset )
        val_wa, val_ua, val_wf1,mat = evaluate_accuracy_frame( model, val_loader, device )
        if (val_wa > best_val_wa) or (val_wa == best_val_wa and val_ua > best_val_wa_ua):
            best_val_wa = val_wa
            best_val_wa_epoch = epoch
            best_val_wa_ua = val_ua
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_wa': best_val_wa,
                'best_val_wa_ua': best_val_wa_ua,
                'best_val_wa_epoch': best_val_wa_epoch
                },save_dir)

        # 计算验证集损失
        val_loss = evaluate_loss_frame(model, val_loader, device)

        # 记录每一次的WA, UA与混淆矩阵
        logger.info( 'epoch %d loss %.4f train_acc %.2f time %.2f s'
              % ( epoch, train_l_sum / len( train_loader ), train_acc_sum * 100, time() - start ) )
        logger.info( 'val_wa %f  val_ua %f val_wf1 %f val_loss %.2f' 
              % ( val_wa*100, val_ua*100, val_wf1*100, val_loss ) )
        logger.info(mat)  
    checkpoint  = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    test_wa,test_ua ,test_wf1,mat= evaluate_accuracy_frame(model,test_loader,device)
    logger.info(f"The {fold}th Fold at epoch {best_val_wa_epoch},test_wa {test_wa*100} test_ua {test_ua*100},test_wf1 {test_wf1*100}")
    return test_wa*100,test_ua*100,test_wf1*100


class CMDLoss(nn.Module):
    def __init__(self, K=5):
        super(CMDLoss, self).__init__()
        self.K = K
    
    def forward(self, X, Y):
        cmd_loss = 0
        for k in range(1, self.K + 1):
            # 计算X和Y的第 k 阶中心矩差异
            moment_x = self.central_moment(X, k)
            moment_y = self.central_moment(Y, k)
            cmd_loss += torch.norm(moment_x - moment_y, p=2) ** 2
        return cmd_loss
    
    # 第k阶中心矩计算
    def central_moment(self, X, k):
        mean = torch.mean(X, dim=0, keepdim=True)
        moment = torch.mean((X - mean) ** k, dim=0)
        return moment
