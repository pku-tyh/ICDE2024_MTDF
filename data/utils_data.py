from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset

import torch
from .data_splits import get_splits_in_domain, get_domain_splits
from torch_geometric.data import DataLoader
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def COX2_MD_transform(data):
    data.x = torch.cat([data.x,torch.zeros((data.x.shape[0],38-data.x.shape[1]))],dim=-1)
    return data

def BZR_MD_transform(data):
    data.x = torch.cat([data.x,torch.zeros((data.x.shape[0],56-data.x.shape[1]))],dim=-1)
    return data

def H_transform(data):
    data.x = torch.cat([data.x,torch.zeros((data.x.shape[0],1))],dim=-1)
    return data

def PTC_transform(data):
    data.x = torch.cat([data.x,torch.zeros((data.x.shape[0],20-data.x.shape[1]))],dim=-1)
    return data

def DD_transform(data):
    data.x = torch.cat([data.x,torch.zeros((data.x.shape[0],89-data.x.shape[1]))],dim=-1)
    return data

def no_node_label_transform(data):
    data.x = torch.ones((data.edge_index.max()+1,7))
    return data

two_dataset_mapping = {
    'COX2': 'COX2_MD',
    'BZR': 'BZR_MD',
    'NCI1': 'NCI109',
    'DHFR': 'DHFR_MD',
    'OVCAR-8':'OVCAR-8H',
    'PC-3':'PC-3H',
    'deezer_ego_nets':'twitch_egos',
    'PTC_FM':'PTC_FR',
    'PTC_MM':'PTC_MR',
    'PROTEINS_full':'DD'
}
multi_dataset_mapping = {
    'PTC': ['PTC_MR','PTC_MM','PTC_FM','PTC_FR']
}
def get_dataset(DS, path, args):
    setup_seed(0)

    if args.cross_dataset == 1: 
        DSS = [DS, two_dataset_mapping[DS]]
        source_split_index = args.source_index
        target_split_index = args.target_index
        source_dataset = TUDataset(path, name=DSS[source_split_index], use_node_attr=True,pre_transform=DD_transform)
        target_dataset = TUDataset(path, name=DSS[target_split_index], use_node_attr=True)
        split_dataset = [source_dataset, target_dataset]
        split_idx = [torch.arange(len(source_dataset)),torch.arange(len(target_dataset))]
        source_train_dataset, source_val_dataset, source_train_index, source_val_index = get_splits_in_domain(source_dataset,split_idx[0])
        target_train_dataset, target_test_dataset, target_train_index, target_test_index = get_splits_in_domain(target_dataset,split_idx[1])
        return source_dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset), (source_train_index, source_val_index, target_train_index, target_test_index),(DSS[source_split_index],DSS[target_split_index])
    elif args.cross_dataset == 2: 
        DSS = multi_dataset_mapping[DS]
        source_split_index = args.source_index
        target_split_index = args.target_index
        source_dataset = TUDataset(path, name=DSS[source_split_index], use_node_attr=True,pre_transform=PTC_transform)
        target_dataset = TUDataset(path, name=DSS[target_split_index], use_node_attr=True,pre_transform=PTC_transform)
        split_dataset = [source_dataset, target_dataset]
        split_idx = [torch.arange(len(source_dataset)),torch.arange(len(target_dataset))]
        source_train_dataset, source_val_dataset, source_train_index, source_val_index = get_splits_in_domain(source_dataset,split_idx[0])
        target_train_dataset, target_test_dataset, target_train_index, target_test_index = get_splits_in_domain(target_dataset,split_idx[1])
        return source_dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset), (source_train_index, source_val_index, target_train_index, target_test_index),(DSS[source_split_index],DSS[target_split_index])
    else:
        dataset = TUDataset(path, name=DS, use_node_attr=True)
        source_split_index = args.source_index
        target_split_index = args.target_index
        split = args.data_split
        split_dataset,split_idx = get_domain_splits(dataset, split)
        source_dataset = split_dataset[source_split_index]
        target_dataset = split_dataset[target_split_index]
        source_train_dataset, source_val_dataset, source_train_index, source_val_index = get_splits_in_domain(source_dataset,split_idx[source_split_index])
        target_train_dataset, target_test_dataset, target_train_index, target_test_index = get_splits_in_domain(target_dataset,split_idx[target_split_index])
    return source_dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset), (source_train_index, source_val_index, target_train_index, target_test_index),(DS,DS)

def split_confident_data(model,dataset,saved_index,device,args):
    model.eval()
    confident_percentage=args.confident_rate
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    confident_dataset = []
    inconfident_dataset = []
    confident_idx = []
    inconfident_idx = []
    correct_count = 0
    for data in loader:
        data = data.to(device)
        feat, feat_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        prob = torch.softmax(model.classifier(feat),dim=-1)
        confident_id = prob.max(dim=-1)[0].topk(int(len(prob)*confident_percentage))[1]
        confident_mask = torch.zeros(len(prob),dtype=torch.bool).to(device)
        confident_mask[confident_id] = True
        inconfident_id = torch.where(~confident_mask)[0]
        confident_dataset += data[confident_mask]
        inconfident_dataset += data[~confident_mask]
        confident_idx += saved_index[confident_id].tolist()
        inconfident_idx += saved_index[inconfident_id].tolist()
        correct_count += (prob.max(dim=-1)[1] == data.y).sum().item()


    train_ratio = 0.8
    confident_train_dataset, confident_val_dataset = confident_dataset[:int(len(confident_dataset)*train_ratio)],confident_dataset[int(len(confident_dataset)*train_ratio):]
    confident_train_idx, confident_val_idx = confident_idx[:int(len(confident_idx)*train_ratio)],confident_idx[int(len(confident_idx)*train_ratio):]
    correct_count = 0
    confident_dataloader = DataLoader(confident_train_dataset, batch_size=2048, shuffle=False)
    for data in confident_dataloader:
        data = data.to(device)
        feat, feat_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        prob = torch.softmax(model.classifier(feat),dim=-1)
        correct_count += (prob.max(dim=-1)[1] == data.y).sum().item()

    return confident_train_dataset,confident_val_dataset,inconfident_dataset,(confident_train_idx,confident_val_idx,inconfident_idx)