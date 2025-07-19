import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from models.modeling import CONFIGS, KUTS
import argparse
import warnings
from sklearn.utils.class_weight import compute_class_weight
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from utils.dataset import MyDataset, MIMICDataset
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset, random_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging
import os

# output file
weights_dir = 'weights'
results_dir = 'results'
log_name = 'KUTS_MIMIC_IV_ED'
# input file
dataset_name = 'mimic_iv_ed'
my_dataset = MIMICDataset(os.path.join('data', dataset_name + '.pkl'))
# learning rate
lr = 5e-5
lr_min = 5e-6
tmax = 20
# Other parameters
tk_lim = 160
grade_list = ['level 1', 'level 2', 'level 3', 'level 4', 'level 5']

# Configure logging
logging.basicConfig(
    filename=os.path.join(results_dir, log_name + '.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Redirect sys. stdout to log
class LogWriter:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)

    def flush(self):
        pass

sys.stdout = LogWriter(logging.getLogger(), logging.INFO)

def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)
    return (A + B)


def uncertainty_loss(predict, cls_label, epoch, criterion, device, test=False):
    evidences = [F.softplus(predict)]
    loss_un = 0
    alpha = dict()
    alpha[0] = evidences[0] + 1

    S = torch.sum(alpha[0], dim=1, keepdim=True)
    E = alpha[0] - 1
    b = E / (S.expand(E.shape))
    
    u = args.CLS / S

    Tem_Coef = epoch*(0.99/args.NUM_EPOCHS)+0.01

    loss_CE = criterion(b/Tem_Coef, cls_label)

    loss_un += ce_loss(cls_label, alpha[0], args.CLS, epoch, args.NUM_EPOCHS, device)
    loss_ACE = torch.mean(loss_un)
    loss = loss_CE+loss_ACE
    
    if test == False:
        return loss
    else:
        return u

# FocalLoss solves class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, epoch, criterion, device):
        # ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        ce_loss = uncertainty_loss(inputs, targets, epoch, criterion, device)
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Calculate the AUC for each category and the average AUC
def compute_AUROC (dataGT, dataPRED, multi_class = 'ovo'): 
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
        
    outAUROC = roc_auc_score(datanpGT, datanpPRED, average=None, multi_class=multi_class)
    auc = roc_auc_score(datanpGT, datanpPRED, average='macro', multi_class=multi_class)
            
    return outAUROC, auc


# Calculate specificity and NPV (as there are no ready-made library functions to calculate these two)
def calculate_specificity_npv(gt, pred, weight=[1,1,1,1], num_classes=4):
    gtnp = np.array(gt)
    prednp = np.array(pred)
    
    weight_new = [1 / x for x in weight]
    
    specificities = []
    npvs = []
    
    for i in range(num_classes):
        gt_new = np.copy(gtnp)
        gt_new[gtnp == i] = 1
        gt_new[gtnp != i] = 0
        
        pred_new = np.copy(prednp)
        pred_new[prednp == i] = 1
        pred_new[prednp != i] = 0
        
        # print(gt_new, pred_new)
        
        # 计算混淆矩阵
        cm = confusion_matrix(gt_new, pred_new)

        # 从混淆矩阵中提取 TP、FP、TN、FN
        TN, FP, FN, TP = cm.ravel()

        # 计算特异性、阴性预测值
        specificity = TN / (TN + FP)
        npv = TN / (TN + FN)
        
        specificities.append(specificity)
        npvs.append(npv)
        
    return np.average(specificities, weights=weight_new), np.average(npvs, weights=weight_new)


# Training (also including validation and testing)
def train(args):
    # torch.manual_seed(0)
    num_classes = args.CLS
    config = CONFIGS["KUTS"]
    cuda_device = int(args.DEVICE)
    device = torch.device('cuda:' + args.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    criterion = nn.CrossEntropyLoss().to(device)
    
    model = KUTS(config, num_classes=num_classes, device=device).to(device)
    best_acc = best_auc = 0
    arg_best_acc = arg_best_auc = -1
    
    # Obtain samples and labels
    data = my_dataset.data
    labels = []
    for index in range(len(my_dataset)):
        labels.append(my_dataset[index][0])

    # Get all categories
    classes = np.unique(labels)

    train_indices = []
    valid_indices = []
    test_indices = []

    # Index for segmenting data by category and dividing training and testing sets
    for cls in classes:
        # Get the index of the current category
        cls_indices = [i for i, label in enumerate(labels) if label == cls]

        # Index for dividing training and testing sets
        train_cls_size = int(0.7 * len(cls_indices))
        valid_cls_size = int(0.1 * len(cls_indices))
        test_cls_size = len(cls_indices) - train_cls_size - valid_cls_size
        train_cls_indices, valid_cls_indices, test_cls_indices = random_split(cls_indices, [train_cls_size, valid_cls_size, test_cls_size])

        # Add the training and testing set indexes of the current category to the overall training and testing set indexes
        train_indices.extend(train_cls_indices)
        valid_indices.extend(valid_cls_indices)
        test_indices.extend(test_cls_indices)

    # Create subsets of training and testing sets based on index partitioning
    train_dataset = Subset(my_dataset, train_indices)
    valid_dataset = Subset(my_dataset, valid_indices)
    test_dataset = Subset(my_dataset, test_indices)

    # Print the number of samples for the training and testing sets
    print(f"Train data size: {len(train_dataset)}")
    print(f"Valid data size: {len(valid_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    
    train_labels = [labels[i] for i in train_indices]
        
    num_classes = args.CLS

    # Calculate category weights based on training set labels
    class_weights = compute_class_weight('balanced', classes=range(num_classes), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    class_weights = class_weights / class_weights.sum()
    print(f"class_weights = {class_weights}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=lr_min)
    print(f'lr = {lr}, lr_min = {lr_min}, tmax = {tmax}')
    
    # Define loss function
    class_weights = class_weights.to(device)
    loss_fct = FocalLoss(alpha=class_weights, gamma=2, reduction='sum')
    
    trainloader = DataLoader(train_dataset, batch_size=args.BSZ, shuffle=True, collate_fn=my_dataset.collate_fn)
    validloader = DataLoader(valid_dataset, batch_size=args.BSZ, shuffle=False, collate_fn=my_dataset.collate_fn)
    testloader = DataLoader(test_dataset, batch_size=args.BSZ, shuffle=False, collate_fn=my_dataset.collate_fn)

    #----- Training ----------------------------------------------------------------------
    print('--------Start training-------')
    model.train()
    for epoch in range(args.NUM_EPOCHS):
        model.train()
        running_loss = torch.zeros(1).to(device)  # Accumulated losses
        all_pred_classes = []
        all_labels = []
        outGT = torch.FloatTensor().cuda(cuda_device, non_blocking=True)
        outPRED = torch.FloatTensor().cuda(cuda_device, non_blocking=True)
        for step, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            label, age, gender, dp, sp, sense, temp, spo, breath, hr, lai, input_ids, attention_mask, token_type_ids, text = data
            input_ids = input_ids.cuda(cuda_device, non_blocking=True)
            attention_mask = attention_mask.cuda(cuda_device, non_blocking=True)
            token_type_ids = token_type_ids.cuda(cuda_device, non_blocking=True)
            age = age.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            gender = gender.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            dp = dp.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            sp = sp.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            sense = sense.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            temp = temp.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            spo = spo.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            breath = breath.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            hr = hr.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            lai = lai.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
            label = label.cuda(cuda_device, non_blocking=True)
            
            # forward + backward + optimize
            logits = model(input_ids, attention_mask, token_type_ids, age, gender, dp, sp, sense, temp, spo, breath, hr, lai)
            loss = loss_fct(logits.view(-1, num_classes), label.view(-1).to(torch.long), epoch, criterion, device)
            probs = torch.sigmoid(logits)
            
            pred_classes = torch.max(logits, dim=1)[1]   
            all_pred_classes.extend(pred_classes.detach().cpu().tolist())
            all_labels.extend(label.detach().cpu().tolist())

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.detach()
            
            optimizer.step()
            
            label_onehot = torch.nn.functional.one_hot(label.to(torch.long), num_classes = num_classes)
            outGT = torch.cat((outGT, label_onehot.to(device)), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)

            # print statistics
            if step % 50 == 0:
                print("[train epoch {} step {}/{}] all_loss: {:.3f}, acc: {:.3f}, f1: {:.3f}".format(epoch, step, len(trainloader), running_loss.item() / (step + 1), accuracy_score(all_labels, all_pred_classes), f1_score(all_labels, all_pred_classes, average='weighted')))

        epoch_loss = running_loss.item() / (step + 1)
        epoch_acc = accuracy_score(all_labels, all_pred_classes)
        epoch_f1 = f1_score(all_labels, all_pred_classes, average='weighted')
        print('Train Epoch [{}/{}], Loss: {:.3f}, acc: {:.3f}, f1: {:.3f}'.format(epoch, args.NUM_EPOCHS, epoch_loss, epoch_acc, epoch_f1))
        
        aurocIndividual, aurocMean = compute_AUROC(outGT, outPRED, multi_class='ovo')
        print('mean AUROC:' + str(aurocMean))
        for i in range (0, len(aurocIndividual)):
            print(grade_list[i] + ': '+str(aurocIndividual[i]))
        
        scheduler.step()
        
        # -------valid----------------------------------------------------
        model.eval()
        with torch.no_grad():
            all_pred_classes = []
            all_labels = []
            outGT = torch.FloatTensor().cuda(cuda_device, non_blocking=True)
            outPRED = torch.FloatTensor().cuda(cuda_device, non_blocking=True)
            running_loss = torch.zeros(1).to(device)  # Accumulated losses
            for step, data in enumerate(validloader):
                # get the inputs; data is a list of [inputs, labels]
                label, age, gender, dp, sp, sense, temp, spo, breath, hr, lai, input_ids, attention_mask, token_type_ids, text = data
                input_ids = input_ids.cuda(cuda_device, non_blocking=True)
                attention_mask = attention_mask.cuda(cuda_device, non_blocking=True)
                token_type_ids = token_type_ids.cuda(cuda_device, non_blocking=True)
                age = age.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                gender = gender.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                dp = dp.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                sp = sp.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                sense = sense.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                temp = temp.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                spo = spo.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                breath = breath.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                lai = lai.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                hr = hr.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                
                label = label.cuda(cuda_device, non_blocking=True)
                
                # forward + backward + optimize
                logits = model(input_ids, attention_mask, token_type_ids, age, gender, dp, sp, sense, temp, spo, breath, hr, lai)
                loss = loss_fct(logits.view(-1, num_classes), label.view(-1).to(torch.long), epoch, criterion, device)
                probs = torch.sigmoid(logits)
                
                pred_classes = torch.max(logits, dim=1)[1]              
                all_pred_classes.extend(pred_classes.detach().cpu().tolist())
                all_labels.extend(label.detach().cpu().tolist())
                
                running_loss += loss.detach()
                
                label_onehot = torch.nn.functional.one_hot(label.to(torch.long), num_classes = num_classes)
                outGT = torch.cat((outGT, label_onehot), 0)
                outPRED = torch.cat((outPRED, probs.data), 0)

            epoch_loss = running_loss.item() / (step + 1)
            epoch_acc = accuracy_score(all_labels, all_pred_classes)
            epoch_f1 = f1_score(all_labels, all_pred_classes, average='weighted')
            print('Valid Epoch [{}/{}], Loss: {:.3f}, acc: {:.3f}, f1:{:.3f}'.format(epoch, args.NUM_EPOCHS, epoch_loss, epoch_acc, epoch_f1))
            
            aurocIndividual, aurocMean = compute_AUROC(outGT, outPRED, multi_class='ovo')
            print('mean AUROC:' + str(aurocMean))
            for i in range (0, len(aurocIndividual)):
                print(grade_list[i] + ': '+str(aurocIndividual[i]))
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                arg_best_acc = epoch
                torch.save(model.state_dict(), os.path.join(weights_dir, log_name + '_' + 'best_acc_params' + '.pth'))
                
            if aurocMean > best_auc:
                best_auc = aurocMean
                arg_best_auc = epoch
                torch.save(model.state_dict(), os.path.join(weights_dir, log_name + '_' + 'best_auc_params' + '.pth'))
                
            if epoch == args.NUM_EPOCHS - 1:
                torch.save(model.state_dict(), os.path.join(weights_dir, log_name + '_' + 'last_epoch_params' + '.pth'))
                
            print('Best acc is {:.3f} in epoch {}, best auc is {:.3f} in epoch {}.'.format(best_acc, arg_best_acc, best_auc, arg_best_auc))
            
        # ------test------------------------------------------------
        if epoch == args.NUM_EPOCHS - 1:
            # 各种指标
            acc_ = 0
            pre_ = 0
            recall_ = 0
            spe_ = 0
            npv_ = 0
            f1_ = 0
            f2_ = 0
            auc_ = 0
            auci_ = []
            
            params = ['best_acc_params', 'best_auc_params', 'last_epoch_params']
            for param in params:
                model.load_state_dict(torch.load(os.path.join(weights_dir, log_name + '_' + param + '.pth')))
                model.eval()
                with torch.no_grad():
                    outGT = torch.FloatTensor().cuda(cuda_device, non_blocking=True)
                    outPRED = torch.FloatTensor().cuda(cuda_device, non_blocking=True)
                    all_pred_classes = []
                    all_labels = []
                    running_loss = torch.zeros(1).to(device)  # Accumulated losses
                    for step, data in enumerate(testloader):
                        # get the inputs; data is a list of [inputs, labels]
                        label, age, gender, dp, sp, sense, temp, spo, breath, hr, lai, input_ids, attention_mask, token_type_ids, text = data
                        input_ids = input_ids.cuda(cuda_device, non_blocking=True)
                        attention_mask = attention_mask.cuda(cuda_device, non_blocking=True)
                        token_type_ids = token_type_ids.cuda(cuda_device, non_blocking=True)
                        age = age.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        gender = gender.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        dp = dp.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        sp = sp.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        sense = sense.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        temp = temp.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        spo = spo.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        breath = breath.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        hr = hr.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        lai = lai.view(-1, 1).cuda(cuda_device, non_blocking=True).float()
                        label = label.cuda(cuda_device, non_blocking=True)
                        
                        # forward + backward + optimize
                        logits = model(input_ids, attention_mask, token_type_ids, age, gender, dp, sp, sense, temp, spo, breath, hr, lai)
                        probs = torch.sigmoid(logits)
                        
                        pred_classes = torch.max(logits, dim=1)[1]                      
                        all_pred_classes.extend(pred_classes.detach().cpu().tolist())
                        all_labels.extend(label.detach().cpu().tolist())
                        
                        label_onehot = torch.nn.functional.one_hot(label.to(torch.long), num_classes = num_classes)
                        outGT = torch.cat((outGT, label_onehot), 0)
                        outPRED = torch.cat((outPRED, probs.data), 0)
              
                    acc = accuracy_score(all_labels, all_pred_classes)
                    precision = precision_score(all_labels, all_pred_classes, average='weighted')
                    recall = recall_score(all_labels, all_pred_classes, average='weighted')
                    specificity, npv = calculate_specificity_npv(all_labels, all_pred_classes, weight=class_weights.cpu().numpy(), num_classes=num_classes)
                    
                    beta = 2  # Set the weight of recall rate
                    f1 = f1_score(all_labels, all_pred_classes, average='weighted')
                    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                    
                    print(param)
                    print('Test, acc: {:.3f}, sensitivity: {:.3f}, specificity: {:.3f}, ppv: {:.3f}, npv: {:.3f}, f1: {:.3f}, f2: {:.3f}'.format(acc, recall, specificity, precision, npv, f1, f2))
                    
                    aurocIndividual, aurocMean = compute_AUROC(outGT, outPRED, multi_class='ovo')
                    print('mean AUROC:' + str(aurocMean))
                    for i in range (0, len(aurocIndividual)):
                        print(grade_list[i] + ': '+str(aurocIndividual[i]))
                    
                    if param == 'best_auc_params':
                        acc_ = acc 
                        pre_ = precision
                        recall_ = recall
                        spe_ = specificity
                        npv_ = npv
                        f1_ = f1
                        f2_ = f2
                        auc_ = aurocMean
                        auci_ = aurocIndividual
                        
            return acc_, pre_, recall_, spe_, npv_, f1_, f2_, auc_, auci_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str)
    parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', required=True, type=str)
    parser.add_argument('--NUM_EPOCHS', action='store', dest='NUM_EPOCHS', required=True, type=int)
    parser.add_argument('--DEVICE', action='store', dest='DEVICE', required=True, type=str)
    args = parser.parse_args()
    
    acc_list = []
    pre_list = []
    recall_list = []
    spe_list = []
    npv_list = []
    f1_list = []
    f2_list = []
    auc_list = []
    auci_list = []
    
    # Run 10 times to take the average value of each indicator
    for i in range(10):
        acc_, pre_, recall_, spe_, npv_, f1_, f2_, auc_, auci_ = train(args)
        acc_list.append(acc_)
        pre_list.append(pre_)
        recall_list.append(recall_)
        spe_list.append(spe_)
        npv_list.append(npv_)
        f1_list.append(f1_)
        f2_list.append(f2_)
        auc_list.append(auc_)
        auci_list.append(auci_)
    
    # Convert list to NumPy array
    # print(acc_list)
    acc_array = np.array(acc_list, dtype=float)
    pre_array = np.array(pre_list, dtype=float)
    recall_array = np.array(recall_list, dtype=float)
    spe_array = np.array(spe_list, dtype=float)
    npv_array = np.array(npv_list, dtype=float)
    f1_array = np.array(f1_list, dtype=float)
    f2_array = np.array(f2_list, dtype=float)
    auc_array = np.array(auc_list, dtype=float)
    auci_array = np.array(auci_list, dtype=float)
    

    # Calculate the mean and confidence space
    # auroc
    auc_mean = np.mean(auc_array)
    auc_percentile = np.percentile(auc_array, [2.5, 97.5])
    auci_mean = np.mean(auci_array, axis=0)
    auci_percentile = np.percentile(auci_array, [2.5, 97.5], axis=0)
    # accuracy
    acc_mean = np.mean(acc_array)
    acc_percentile = np.percentile(acc_array, [2.5, 97.5])
    # sensitivity(recall)
    recall_mean = np.mean(recall_array)
    recall_percentile = np.percentile(recall_array, [2.5, 97.5])
    # specificity
    spe_mean = np.mean(spe_array)
    spe_percentile = np.percentile(spe_array, [2.5, 97.5])
    # ppv(precision)
    pre_mean = np.mean(pre_array)
    pre_percentile = np.percentile(pre_array, [2.5, 97.5])
    # npv
    npv_mean = np.mean(npv_array)
    npv_percentile = np.percentile(npv_array, [2.5, 97.5])
    # f1
    f1_mean = np.mean(f1_array)
    f1_percentile = np.percentile(f1_array, [2.5, 97.5])
    # f2
    f2_mean = np.mean(f2_array)
    f2_percentile = np.percentile(f2_array, [2.5, 97.5])

    # Format output
    print('Final:')
    print(f'auroc: {auc_mean:.3f} [{auc_percentile[0]:.3f}, {auc_percentile[1]:.3f}]')
    for i in range(len(auci_mean)):
        print(f'level {i + 1}: {auci_mean[i]:.3f} [{auci_percentile[0, i]:.3f}, {auci_percentile[1, i]:.3f}]')
    
    print(f'accuracy: {acc_mean:.3f} [{acc_percentile[0]:.3f}, {acc_percentile[1]:.3f}]')
    print(f'sensitivity: {recall_mean:.3f} [{recall_percentile[0]:.3f}, {recall_percentile[1]:.3f}]')
    print(f'specificity: {spe_mean:.3f} [{spe_percentile[0]:.3f}, {spe_percentile[1]:.3f}]')
    print(f'ppv: {pre_mean:.3f} [{pre_percentile[0]:.3f}, {pre_percentile[1]:.3f}]')
    print(f'npv: {npv_mean:.3f} [{npv_percentile[0]:.3f}, {npv_percentile[1]:.3f}]')
    print(f'f1: {f1_mean:.3f} [{f1_percentile[0]:.3f}, {f1_percentile[1]:.3f}]')
    print(f'f2: {f2_mean:.3f} [{f2_percentile[0]:.3f}, {f2_percentile[1]:.3f}]')