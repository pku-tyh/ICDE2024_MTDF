import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
import time
from arguments import arg_parse
from data.utils_data import get_dataset,split_confident_data
from data.graph_aug import AugTransform
from torch.utils.data import SequentialSampler
from models.model import GNN, GNNWithDropout

from utils.utils import compute_mmd_batch, get_logger, setup_seed
import copy


from MGNN_Graph.preprocess import ScaledNormalize, MotifGraph, ConcatDataset
from torch_geometric.datasets import TUDataset
from MGNN_Graph.main import Net
from utils.mixup_utils import prepare_graphon,prepare_aligned_dataset,prepare_augmented_dataset

@torch.no_grad()
def test(loader, model):
    if (type(model) == list):
        model[0].eval()
        model[1].eval()
        total_correct = 0
        for data0,data1 in zip(loader[0],loader[1]):
            data0 = data0.to(device)
            motif0 = data1[0]
            motif0 = motif0.to(device)
            try:
                feat0, feat_proj0 = model[0](data0.x, data0.edge_index, data0.batch, data0.num_graphs)
                feat1, feat_proj1 = model[1](data1)
                pred = (model[0].classifier(feat0) + feat1).argmax(dim=-1)
            except:
                breakpoint()
            total_correct += int((pred == data0.y).sum())
        return total_correct / len(loader[0].dataset)
    else:
        model.eval()

        total_correct = 0
        loss_sum = 0
        for data in loader:
            data = data.to(device)
            try:
                feat, feat_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
            except:
                breakpoint()
            pred = model.classifier(feat).argmax(dim=-1)
            loss_sum += F.nll_loss(F.log_softmax(model.classifier(feat),dim=-1), data.y)
            total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)



@torch.no_grad()
def eval_train(loader, model):
    model.eval()

    total_correct = 0
    for data_dict in loader:
        data = data_dict.to(device)
        feat, feat_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(feat).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

@torch.no_grad()
def eval_train_motif(loader, model):
    model.eval()
    total_correct = 0
    for data_dict in loader:
        motif0 = data_dict[0]
        motif0 = motif0.to(device)
        motif_out,motif_embedding = model(data_dict)
        pred = motif_out.argmax(dim=-1)
        total_correct += int((pred == motif0.y).sum())
    return total_correct / len(loader.dataset)



def update_ema_variables(model, ema_model, alpha, global_step, speed = 1.0):
    alpha = min(1 - 1 / (global_step + 1), alpha) * speed
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)

def pretrain_motif(model,train_loader,val_loader,optimizer,criterion,device,args):
    model.train()
    best_val_acc = 0.0
    best_model = copy.deepcopy(model)
    total_loss = 0
    for epoch in range(1,args.motif_epochs+1):
        for data in train_loader:
            motif0 = data[0]
            motif0 = motif0.to(device)
            optimizer.zero_grad()
            motif_out,motif_embedding = model(data)
            loss = F.nll_loss(motif_out, data[0].y)
            total_loss = loss.item() * data[0].num_graphs
            loss.backward()
            optimizer.step()
        if epoch % args.motif_eval_interval == 0:
            model.eval()
            train_acc = eval_train_motif(train_loader, model)
            val_acc = eval_train_motif(val_loader, model)
            print(f'Epoch: {epoch:03d}, Loss: {total_loss / len(train_loader):.2f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')
            total_loss = 0
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
    return best_model

def run(seed):
    logger.info('seed:{}'.format(seed))

    epochs = args.epochs
    eval_interval = args.eval_interval
    sdfa_eval_interval = args.sdfa_eval_interval
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    source_dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset), \
        (source_train_index, source_val_index, target_train_index, target_test_index),(sDS,tDS) = get_dataset(DS, path, args)

    target_graphon = prepare_graphon(target_train_dataset)
    if args.train_mixup == 1:
        source_train_dataset = prepare_aligned_dataset(source_train_dataset)

    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    source_val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    train_transforms = AugTransform(args.aug,0.3)

    print('Calculating uniform targets...')
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    nll_criterion = nn.NLLLoss()

    dataset_num_features = source_train_dataset[0].x.shape[1]
    print(f'num_features: {dataset_num_features}')

    setup_seed(seed)
    model = GNN(dataset_num_features, args.hidden_dim, args.num_gc_layers, source_dataset.num_classes, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-2)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    best_val_acc = 0.0
    best_test_acc = 0.0
    final_test_acc = 0.0
    best_model = copy.deepcopy(model)
    for epoch in range(1, epochs + 1):
        time_start = time.time()

        loss_all = 0
        model.train()

        dataloader = source_train_loader
        
        for data_dict in dataloader:
            if args.train_mixup == 1:
                data_dict = prepare_augmented_dataset(data_dict,target_graphon,args.train_mixup_weight)
            data = data_dict.to(device)
            optimizer.zero_grad()

            x, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)

            pred = model.classifier(x)
            loss = ce_criterion(pred, data.y)

            loss.backward()

            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        if epoch == 1000:
            for data_dict in target_train_loader:
                data = data_dict.to(device)
                optimizer.zero_grad()
                x, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
                pred = model.classifier(x)
                random_label = torch.randint(0,2,(len(pred),)).to(device)
                loss = ce_criterion(pred, random_label)
                loss.backward()
                optimizer.step()


        if epoch % eval_interval == 0:
            model.eval()
            train_acc = eval_train(dataloader, model)
            val_acc = test(source_val_loader, model)
            test_acc = test(target_test_loader, model)
            if test_acc >= best_test_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                raw_target_acc = test_acc
                best_model = copy.deepcopy(model)
            print(f'Epoch: {epoch:03d}, Loss: {loss_all / len(dataloader):.2f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f} ')
      
    model = best_model

    confident_train_dataset,confident_val_dataset,inconfident_dataset,(confident_train_index,confident_val_index,inconfident_index) = split_confident_data(model,target_train_dataset,target_train_index,device,args)

    train_index = torch.randperm(len(confident_train_dataset))
    val_index = torch.randperm(len(confident_val_dataset))
    train_sampler = SequentialSampler(train_index)
    val_sampler = SequentialSampler(val_index)
    confident_unaligned_train_dataset = copy.deepcopy(confident_train_dataset)
    confident_train_unaligned_dataloder = DataLoader(confident_unaligned_train_dataset, batch_size=batch_size, num_workers=0, sampler=train_sampler)
    

    if args.use_mixup == 1:
        graphon = prepare_graphon(source_train_dataset)
        confident_train_dataset = prepare_aligned_dataset(confident_train_dataset)

    confident_train_dataloder = DataLoader(confident_train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    confident_val_dataloder = DataLoader(confident_val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)
    
    
    model_for_psuedo_label = copy.deepcopy(model)
    model_for_psuedo_label.eval()
    if args.use_motif_branch == 1:
        motif_model = Net(dataset_num_features, [16, source_train_dataset.num_classes], source_train_dataset.num_classes, args).to(device)
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'GraphMGNN_DATA')
        source_motif_dataset = TUDataset(path, name=sDS, pre_transform=ScaledNormalize(args.padto), use_node_attr=True)
        source_motif_train_datasets = [source_motif_dataset[source_train_index]]
        source_motif_val_datasets = [source_motif_dataset[source_val_index]]
        for i in range(1,14):
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'GraphMGNN_DATA', f'M{i}')
            dataset_motif = TUDataset(path, name=sDS, pre_transform=MotifGraph(f'M{i}', DS, pad=args.padto),use_node_attr=True)
            motif_train_dataset = dataset_motif[source_train_index]
            motif_val_dataset = dataset_motif[source_val_index]
            source_motif_train_datasets.append(motif_train_dataset)
            source_motif_val_datasets.append(motif_val_dataset)
        source_motif_train_loader = DataLoader(ConcatDataset(source_motif_train_datasets), batch_size,shuffle=True)
        source_motif_val_loader = DataLoader(ConcatDataset(source_motif_val_datasets), batch_size,shuffle=False)
        pretrain_optimizer = torch.optim.Adam(motif_model.parameters(), lr=0.01, weight_decay=5e-3)
        motif_model = pretrain_motif(motif_model,source_motif_train_loader,source_motif_val_loader,pretrain_optimizer,ce_criterion,device,args)
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'GraphMGNN_DATA')
        dataset = TUDataset(path, name=tDS, pre_transform=ScaledNormalize(args.padto), use_node_attr=True)
        motif_train_datasets = [dataset[confident_train_index]]
        motif_val_datasets = [dataset[confident_val_index]]
        motif_test_datasets = [dataset[target_test_index]]

        for i in range(1,14):
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'GraphMGNN_DATA', f'M{i}')
            dataset_motif = TUDataset(path, name=tDS, pre_transform=MotifGraph(f'M{i}', DS, pad=args.padto),use_node_attr=True)
            motif_train_dataset = dataset_motif[confident_train_index]
            motif_test_dataset = dataset_motif[target_test_index]
            motif_val_dataset = dataset_motif[confident_val_index]
            motif_train_datasets.append(motif_train_dataset)
            motif_test_datasets.append(motif_test_dataset)
            motif_val_datasets.append(motif_val_dataset)
        motif_train_loader = DataLoader(ConcatDataset(motif_train_datasets), batch_size, sampler=train_sampler)
        motif_val_loader = DataLoader(ConcatDataset(motif_val_datasets), batch_size, sampler=val_sampler)
        motif_test_loader = DataLoader(ConcatDataset(motif_test_datasets), batch_size, shuffle=False)

        optimizer = torch.optim.Adam(list(model.parameters()) + list(motif_model.parameters()), lr=args.sdfa_lr, weight_decay=5e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.sdfa_lr, weight_decay=5e-3)

        
    if args.use_teacher == 1:
        teacher_model = GNNWithDropout(dataset_num_features, args.hidden_dim, args.num_gc_layers, \
                                       source_train_dataset.num_classes, args, device, dropout_rate=args.teacher_dropout).to(device)
        teacher_model.load_state_dict(model.state_dict())
        for param in teacher_model.parameters():
            param.requires_grad = False
        if args.dual_teacher == 1:
            teacher_model_sta = copy.deepcopy(model)
            teacher_model_sta.eval()
            teacher_model_sta = GNNWithDropout(dataset_num_features, args.hidden_dim, args.num_gc_layers, \
                                               source_train_dataset.num_classes, args, device, dropout_rate=args.teacher_dropout).to(device)
            teacher_model_sta.load_state_dict(model.state_dict())
            for param in teacher_model_sta.parameters():
                param.requires_grad = False

    target_epochs = args.target_epochs
    best_val_acc = 0.0
    final_test_acc = 0.0
    best_test_acc = 0.0
    best_dual_teacher_acc = 0.0
    best_epoch = 0
    for epoch in range(1, target_epochs + 1):
        if args.use_motif_branch == 1:
            motif_train_iter = iter(motif_train_loader)
        unaligned_train_iter = iter(confident_train_unaligned_dataloder)
        loss_all = 0

        dataloader = confident_train_dataloder
        for data_dict in dataloader:
            if args.use_motif_branch == 1:
                motif_data = next(motif_train_iter)
            pseudo_data = next(unaligned_train_iter).to(device)

            if args.update_psuedo:
                model.eval()
                with torch.no_grad():
                    x_p,_ = model(pseudo_data.x, pseudo_data.edge_index, pseudo_data.batch, pseudo_data.num_graphs)
                    pred_p = model.classifier(x_p)
                    pseudo_label = pred_p.argmax(dim=-1)
            else:
                model_for_psuedo_label.eval()
                x_p,_ = model_for_psuedo_label(pseudo_data.x, pseudo_data.edge_index, pseudo_data.batch, pseudo_data.num_graphs)
                pred_p = model_for_psuedo_label.classifier(x_p)
                pseudo_label = pred_p.argmax(dim=-1)
            
            if args.use_mixup == 1:
                data_dict = prepare_augmented_dataset(data_dict,graphon,args.mixup_weight_source*(target_epochs - epoch)/target_epochs)
                data_dict = prepare_augmented_dataset(data_dict,target_graphon,args.mixup_weight_target*(epoch)/target_epochs)

            data = data_dict.to(device)
                
            

            optimizer.zero_grad()
            x, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
            pred = model.classifier(x)

            model.train()
            loss = ce_criterion(pred, pseudo_label)
            if args.use_motif_branch == 1:
                motif0 = motif_data[0]
                motif0 = motif0.to(device)
                motif_out,motif_embedding = motif_model(motif_data)
                cs_loss = mse_criterion(x_proj, motif_embedding)
                ce_loss2 = nll_criterion(motif_out, pseudo_label)
                loss += args.motif_weight * cs_loss + ce_loss2
            if args.use_teacher == 1:
                if args.dual_teacher == 0:
                    teacher_x, teacher_x_proj = teacher_model(data.x, data.edge_index, data.batch, data.num_graphs)
                    teacher_pred = teacher_model.classifier(teacher_x)
                    teacher_loss = mse_criterion(teacher_pred, pred)
                else:
                    teacher_x, teacher_x_proj = teacher_model(data.x, data.edge_index, data.batch, data.num_graphs)
                    teacher_pred_fast = teacher_model.classifier(teacher_x)
                    teacher_x_sta, teacher_x_proj_sta = teacher_model_sta(data.x, data.edge_index, data.batch, data.num_graphs)
                    teacher_pred_sta = teacher_model_sta.classifier(teacher_x_sta)
                    teacher_loss = mse_criterion(teacher_pred_fast, pred) + mse_criterion(teacher_pred_sta, pred)

                loss += args.cosistency_weight * teacher_loss
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
            if args.use_teacher == 1:
                update_ema_variables(model, teacher_model, args.ema_decay, epoch)
                    
        time_end = time.time()
        if epoch % sdfa_eval_interval == 0:
            model.eval()
            train_acc = eval_train(confident_train_dataloder, model)
            val_acc = test(confident_val_dataloder, model)
            if args.use_motif_branch == 1:
                test_acc = test(target_test_loader, model)
            else:
                test_acc = test(target_test_loader, model)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                final_test_acc = test_acc
            if args.use_teacher == 1 and args.dual_teacher == 1:
                dual_teacher_acc = test(target_test_loader, teacher_model)
                if best_dual_teacher_acc < dual_teacher_acc:
                    teacher_model_sta = copy.deepcopy(teacher_model)
                    teacher_model_sta.eval()
                    teacher_model_sta = GNNWithDropout(dataset_num_features, args.hidden_dim, args.num_gc_layers, \
                                                    source_train_dataset.num_classes, args, device, dropout_rate=args.teacher_dropout).to(device)
                    teacher_model_sta.load_state_dict(model.state_dict())
                    for param in teacher_model_sta.parameters():
                        param.requires_grad = False
                    best_dual_teacher_acc = dual_teacher_acc

            print(f'SFDA Epoch: {epoch:03d}, Loss: {loss_all / len(dataloader):.2f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')
            
    logger.info('final_test_acc: {:.2f}'.format(final_test_acc * 100))

    return raw_target_acc,final_test_acc


if __name__ == '__main__':
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = get_logger(args)

    acc_list = []
    raw_acc_list = []
    for seed in range(args.st_seed, args.st_seed+args.number_of_run):
        raw_acc,test_acc = run(seed)
        acc_list.append(test_acc)
        raw_acc_list.append(raw_acc)

    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    with open( "./csv/" + args.DS + "_accs_" +args.log_file.replace("txt","csv") ,'a') as f:
        f.write(f'{args.source_index}->{args.target_index},{acc_mean},{acc_std}\n')
