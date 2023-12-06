from typing import List, Tuple
import numpy as np
import copy
import torch_geometric.transforms as T
from torch_geometric.utils import degree, to_dense_adj
import torch.nn.functional as F
import torch
import random
from torch_geometric.datasets import TUDataset

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data,Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def prepare_synthetic_dataset(dataset):
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)

            data.x = F.one_hot(degs.to(torch.int64), num_classes=max_degree+1).to(torch.float)
            print(data.x.shape)


        return dataset


def prepare_dataset(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset




def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()



def align_graphs(graphs: List[np.ndarray],
                 padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  
        idx = idx[::-1]  

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        max_num = max(max_num, N)

        if padding:
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

    return aligned_graphs, normalized_node_degrees, max_num, min_num



def align_x_graphs(graphs: List[np.ndarray], node_x: List[np.ndarray], padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)
    aligned_graphs = []
    aligned_node_x = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  
        idx = idx[::-1]  
        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        node_x[i] = copy.deepcopy( node_x[i] )
        sorted_node_x = node_x[i][ idx, :]
        aligned_node_x.append(sorted_node_x)

        if padding:
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)

            aligned_node_x = np.zeros((max_num, 1))
            aligned_node_x[:num_i, :] = sorted_node_x


        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

            aligned_node_x = aligned_node_x[:N]

    return aligned_graphs, aligned_node_x, normalized_node_degrees, max_num, min_num





def two_graphons_mixup(two_graphons, la=0.5, num_sample=20):

    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]

    sample_graph_label = torch.from_numpy(label).type(torch.float32)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)
        
    return sample_graphs



def two_x_graphons_mixup(two_x_graphons, la=0.5, num_sample=20):

    label = la * two_x_graphons[0][0] + (1 - la) * two_x_graphons[1][0]
    new_graphon = la * two_x_graphons[0][1] + (1 - la) * two_x_graphons[1][1]
    new_x = la * two_x_graphons[0][2] + (1 - la) * two_x_graphons[1][2]

    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    sample_graph_x = torch.from_numpy(new_x).type(torch.float32)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.x = sample_graph_x
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)
    return sample_graphs



def graphon_mixup(dataset, la=0.5, num_sample=20):
    graphons = estimate_graphon(dataset, universal_svd)

    two_graphons = random.sample(graphons, 2)

    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]

    print("new label:", label)
    sample_graph_label = torch.from_numpy(label).type(torch.float32)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) < new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]

        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes

        sample_graphs.append(pyg_graph)
    return sample_graphs


def estimate_graphon(dataset, graphon_estimator):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
    num_classes = len(set(y_list))

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)


    graphons = []
    for class_label in set(y_list):
        c_graph_list = [ all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label ]

        aligned_adj_list, normalized_node_degrees, max_num, min_num = align_graphs(c_graph_list, padding=True, N=400)

        graphon_c = graphon_estimator(aligned_adj_list, threshold=0.2)

        graphons.append((np.array(class_label), graphon_c))

    return graphons



def estimate_one_graphon(aligned_adj_list: List[np.ndarray], method="universal_svd"):

    if method == "universal_svd":
        graphon = universal_svd(aligned_adj_list, threshold=0.2)
    else:
        graphon = universal_svd(aligned_adj_list, threshold=0.2)

    return graphon



def split_class_x_graphs(dataset):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
    num_classes = len(set(y_list))

    all_graphs_list = []
    all_node_x_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)
        all_node_x_list = [graph.x.numpy()]

    class_graphs = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        c_node_x_list = [all_node_x_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        class_graphs.append( ( np.array(class_label), c_graph_list, c_node_x_list ) )

    return class_graphs


def split_class_graphs(dataset):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
    num_classes = len(set(y_list))

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)

    class_graphs = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        class_graphs.append( ( np.array(class_label), c_graph_list ) )

    return class_graphs




def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    aligned_graphs = graph_numpy2tensor(aligned_graphs).to( "cuda" )
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :] 

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.cpu().numpy()
    torch.cuda.empty_cache()
    return graphon

def stat_graph(graphs_list: List[Data]):
    num_total_nodes = []
    num_total_edges = []
    for graph in graphs_list:
        num_total_nodes.append(graph.num_nodes)
        num_total_edges.append(  graph.edge_index.shape[1] )
    avg_num_nodes = sum( num_total_nodes ) / len(graphs_list)
    avg_num_edges = sum( num_total_edges ) / len(graphs_list) / 2.0
    avg_density = avg_num_edges / (avg_num_nodes * avg_num_nodes)

    median_num_nodes = np.median( num_total_nodes ) 
    median_num_edges = np.median(num_total_edges)
    median_density = median_num_edges / (median_num_nodes * median_num_nodes)

    return avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density


def mixup_augment(sample_graphs, graphon, la=0.5):
    augmented_graph = []
    for graph_adj in sample_graphs:
        sized_graphon = graphon[:graph_adj.shape[0], :graph_adj.shape[1]]
        if sized_graphon.shape < graph_adj.shape:
            sized_graphon = np.pad(sized_graphon, ((0, graph_adj.shape[0] - sized_graphon.shape[0]), (0, graph_adj.shape[1] - sized_graphon.shape[1])), 'constant', constant_values=0)
        
        
        new_graphon = la * sized_graphon  + (1 - la) * graph_adj
        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        augmented_graph.append(sample_graph)
    
    return augmented_graph

def remove_self_loop_adj(adj):
    adj = adj - np.diag(np.diag(adj))
    return adj

def build_pyg_dataset_x(graphs_list: List[np.ndarray],node_x,datay=None):
    pyg_graphs = []
    for i in range(len(graphs_list)):
        A = torch.from_numpy(graphs_list[i])
        edge_index, _ = dense_to_sparse(A)
        pyg_graph = Data()
        if (type(node_x[i]) == np.ndarray):
            pyg_graph.x = torch.from_numpy(node_x[i])
            pyg_graph.y = torch.tensor([datay[i].y])
        else:
            pyg_graph.x = node_x[i].x
            pyg_graph.y = torch.tensor(node_x[i].y)
        num_nodes = pyg_graph.x.shape[0]
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        pyg_graphs.append(pyg_graph)
    return pyg_graphs

def prepare_graphon(dataset):
    all_adj = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_adj.append(adj)
    aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(all_adj, padding=True, N=100)
    graphon = np.sum(np.array(aligned_graphs),axis=0)/len(aligned_graphs)
    return graphon

def prepare_aligned_dataset(dataset):
    all_adj = []
    for graph in dataset:
        graph.to('cpu')
        graph.edge_index = torch.cat((graph.edge_index, torch.stack((torch.arange(graph.num_nodes), torch.arange(graph.num_nodes)))), dim=1)
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_adj.append(adj)
    aligned_graphs, aligned_node_x, normalized_node_degrees, max_num, min_num = align_x_graphs(all_adj, [graph.x.numpy() for graph in dataset], padding=False)
    dataset = build_pyg_dataset_x(aligned_graphs, aligned_node_x, dataset)
    return dataset

def prepare_augmented_dataset(sample_graphs, graphon, la=0.1):
    all_adj = []
    for i in range(sample_graphs.ptr.shape[0]-1):
        graph = sample_graphs[i]
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_adj.append(remove_self_loop_adj(adj))
    agumented_adj = mixup_augment(all_adj, graphon, la=la)
    dataset = build_pyg_dataset_x(agumented_adj, sample_graphs)
    dataset = Batch.from_data_list(dataset)
    return dataset

if __name__ == '__main__':
    dataset1 = TUDataset(root='../data', name='PROTEINS', use_node_attr=True)
    dataset2 = TUDataset(root='../data', name='FRANKENSTEIN', use_node_attr=True)
    all_adj1 = []
    all_adj2 = []
    for graph in dataset1:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_adj1.append(adj)
    aligned_graphs1, normalized_node_degrees1, max_num1, min_num1 = align_graphs(all_adj1, padding=True, N=100)
    graphon1 = universal_svd(aligned_graphs1, threshold=0.2)
    print(graphon1)
    for graph in dataset2:
        graph.edge_index = torch.cat((graph.edge_index, torch.stack((torch.arange(graph.num_nodes), torch.arange(graph.num_nodes)))), dim=1)
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_adj2.append(adj)
    aligned_graphs2, aligned_node_x2, normalized_node_degrees2, max_num2, min_num2 = align_x_graphs(all_adj2, [graph.x.numpy() for graph in dataset2], padding=False)
    dataset2 = build_pyg_dataset_x(aligned_graphs2, aligned_node_x2)
    sample_graphs2 = list(dataset2)[:128]
    for graph in sample_graphs2:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_adj2.append(remove_self_loop_adj(adj))
    agumented_adj = mixup_augment(sample_graphs2, graphon1, la=0.5)
    dataset = build_pyg_dataset_x(agumented_adj, sample_graphs2)

