import torch
from torch_sparse import SparseTensor
from torch import Tensor, neg
import torch_sparse
from typing import List, Tuple
import numpy as np
from rbloom import Bloom
from torch_geometric.seed import seed_everything


class PermIterator:
    '''
    Iterator of a permutation
    '''
    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        # torch.manual_seed(0)
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret
    
def disable_training(model):
    for p in model.parameters():
        p.requires_grad = False
    
def enable_training(model):
    for p in model.parameters():
        p.requires_grad = True
        
def is_model_trainable(model):
    """
    Checks if the model is currently enabled for training.
    
    Args:
        model (nn.Module): The PyTorch model to check.
        
    Returns:
        bool: True if the model is currently trainable, False otherwise.
    """
    for param in model.parameters():
        if param.requires_grad:
            return True
    return False
    
def fairsample(dataset, split_edge, val_ratio: float=0.10, test_ratio: float=0.2):
    
    seed_everything(0)
    
    row, col = dataset.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    
    num_samples = row.size(0)
    
    group_index = dataset.y[dataset.edge_index].sum(0)
    groups = group_index.unique()
    proportions = []
    bf = Bloom(dataset.edge_index.shape[1] * 3, 0.01)
    
    print("[FAIR SAMPLER] SAMPLING POSITIVE PAIRS")
    pos_pairs = None
    for i in groups.tolist():
        group_mask = group_index == i
        group_proportion = (group_mask).sum() / group_index.shape[0]
        proportions.append(group_proportion)
        if pos_pairs is None:
            pos_pairs = dataset.edge_index[:, group_mask][:, torch.randperm(int(group_proportion * num_samples))]
        else:
            pos_pairs = torch.cat((pos_pairs, dataset.edge_index[:, group_mask][:, torch.randperm(int(group_proportion * num_samples))]), dim=-1)
            
    # Populates the bloom filter with the positive pairs 
    aux_edge_index = dataset.edge_index.numpy()
    for i in range(aux_edge_index.shape[1]):
        bf.add(aux_edge_index[:, i].data.tobytes())
    
    non_sens_nodes = (dataset.y == 0).nonzero().flatten().numpy()
    sens_nodes = (dataset.y == 1).nonzero().flatten().numpy()
    
    neg_nodes_groups = [(0, non_sens_nodes, non_sens_nodes), (1, non_sens_nodes, sens_nodes), (2, sens_nodes, sens_nodes)]
    
    print("[FAIR SAMPLER] SAMPLING NEGATIVE PAIRS")
    neg_pairs = []
    neg_samples_per_group = [0] * groups.shape[0]
    for cur_edge_group, group1, group2 in neg_nodes_groups:
        while neg_samples_per_group[cur_edge_group] < int(proportions[cur_edge_group] * num_samples):
            u = np.random.choice(group1)
            v = np.random.choice(group2)
            edge = np.array(sorted([u, v]))
            if u != v and edge.data.tobytes() not in bf:
                # cur_edge_group = int(dataset.y[edge].sum().item())
                neg_samples_per_group[cur_edge_group] += 1
                neg_pairs.append(edge)
                bf.add(edge.data.tobytes())
                
    neg_pairs = torch.tensor(neg_pairs).t()
    
    perm = torch.randperm(pos_pairs.size(1))
    pos_pairs = pos_pairs[:, perm]
    perm = torch.randperm(neg_pairs.size(1))
    neg_pairs = neg_pairs[:, perm]
            
    split_edge['train']['edge'] = pos_pairs[:, :int((1 - val_ratio - test_ratio) * num_samples)].t()
    split_edge['train']['edge_neg'] = neg_pairs[:, :int((1 - val_ratio - test_ratio) * num_samples)].t()
    
    split_edge['valid']['edge'] = pos_pairs[:, int((1 - val_ratio - test_ratio) * num_samples):int((1 - test_ratio) * num_samples)].t()
    split_edge['valid']['edge_neg'] = neg_pairs[:, int((1 - val_ratio - test_ratio) * num_samples):int((1 - test_ratio) * num_samples)].t()
    
    split_edge['test']['edge'] = pos_pairs[:, int((1 - test_ratio) * num_samples):].t()
    split_edge['test']['edge_neg'] = neg_pairs[:, int((1 - test_ratio) * num_samples):].t()
    
    dataset.edge_index = split_edge['train']['edge'].t()
    
    print(f"Proportions: {proportions}")
    print(f"Pos pairs: {pos_pairs.shape[1]}")
    print("Pos train pairs MM", (dataset.y[split_edge['train']['edge']].sum(0) == 0).sum(), 'MF', (dataset.y[split_edge['train']['edge']].sum(0) == 1).sum(), 'FF', (dataset.y[split_edge['train']['edge']].sum(0) == 2).sum())
    print("Pos valid pairs MM", (dataset.y[split_edge['valid']['edge']].sum(0) == 0).sum(), 'MF', (dataset.y[split_edge['valid']['edge']].sum(0) == 1).sum(), 'FF', (dataset.y[split_edge['valid']['edge']].sum(0) == 2).sum())
    print("Pos test pairs MM", (dataset.y[split_edge['test']['edge']].sum(0) == 0).sum(), 'MF', (dataset.y[split_edge['test']['edge']].sum(0) == 1).sum(), 'FF', (dataset.y[split_edge['test']['edge']].sum(0) == 2).sum())
    print(f"Neg pairs: {neg_pairs.shape[1]}")
    print("Pos train pairs MM", (dataset.y[split_edge['train']['edge_neg']].sum(0) == 0).sum(), 'MF', (dataset.y[split_edge['train']['edge_neg']].sum(0) == 1).sum(), 'FF', (dataset.y[split_edge['train']['edge_neg']].sum(0) == 2).sum())
    print("Pos valid pairs MM", (dataset.y[split_edge['valid']['edge_neg']].sum(0) == 0).sum(), 'MF', (dataset.y[split_edge['valid']['edge_neg']].sum(0) == 1).sum(), 'FF', (dataset.y[split_edge['valid']['edge_neg']].sum(0) == 2).sum())
    print("Pos test pairs MM", (dataset.y[split_edge['test']['edge_neg']].sum(0) == 0).sum(), 'MF', (dataset.y[split_edge['test']['edge_neg']].sum(0) == 1).sum(), 'FF', (dataset.y[split_edge['test']['edge_neg']].sum(0) == 2).sum())
    
    # import code
    # code.interact(local={**locals(), **globals()})

    return dataset, split_edge
        


def sparsesample(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > 0
    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand]

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask]

    ret = SparseTensor(row=samplerow.reshape(-1, 1).expand(-1, deg).flatten(),
                       col=samplecol.flatten(),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce().fill_value_(1.0)
    #print(ret.storage.value())
    return ret


def sparsesample2(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(
        row=torch.cat((samplerow, nosamplerow)),
        col=torch.cat((samplecol, nosamplecol)),
        sparse_sizes=adj.sparse_sizes()).to_device(
            adj.device()).fill_value_(1.0).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def sparsesample_reweight(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix. It will also scale the sampled elements.
    
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()
    samplevalue = (rowcount * (1/deg)).reshape(-1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(row=torch.cat((samplerow, nosamplerow)),
                       col=torch.cat((samplecol, nosamplecol)),
                       value=torch.cat((samplevalue,
                                        torch.ones_like(nosamplerow))),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def elem2spm(element: Tensor, sizes: List[int]) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
        element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem


def spmoverlap_(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    '''
    Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
    '''
    assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1

    idx = torch.searchsorted(element1[:-1], element2)
    mask = (element1[idx] == element2)
    retelem = element2[mask]
    '''
    nnz1 = adj1.nnz()
    element = torch.cat((adj1.storage.row(), adj2.storage.row()), dim=-1)
    element.bitwise_left_shift_(32)
    element[:nnz1] += adj1.storage.col()
    element[nnz1:] += adj2.storage.col()
    
    element = torch.sort(element, dim=-1)[0]
    mask = (element[1:] == element[:-1])
    retelem = element[:-1][mask]
    '''

    return elem2spm(retelem, adj1.sizes())


def spmnotoverlap_(adj1: SparseTensor,
                   adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    idx = torch.searchsorted(element1[:-1], element2)
    matchedmask = (element1[idx] == element2)

    maskelem1 = torch.ones_like(element1, dtype=torch.bool)
    maskelem1[idx[matchedmask]] = 0
    retelem1 = element1[maskelem1]

    retelem2 = element2[torch.logical_not(matchedmask)]
    return elem2spm(retelem1, adj1.sizes()), elem2spm(retelem2, adj2.sizes())


def spmoverlap_notoverlap_(
        adj1: SparseTensor,
        adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retoverlap = element1
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

        retoverlap = element2[matchedmask]
        retelem2 = element2[torch.logical_not(matchedmask)]
    sizes = adj1.sizes()
    return elem2spm(retoverlap,
                    sizes), elem2spm(retelem1,
                                     sizes), elem2spm(retelem2, sizes)


def adjoverlap(adj1: SparseTensor,
               adj2: SparseTensor,
               tarei: Tensor,
               filled1: bool = False,
               calresadj: bool = False,
               cnsampledeg: int = -1,
               ressampledeg: int = -1):
    # a wrapper for functions above.
    adj1 = adj1[tarei[0]]
    adj2 = adj2[tarei[1]]
    if calresadj:
        adjoverlap, adjres1, adjres2 = spmoverlap_notoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
        if ressampledeg > 0:
            adjres1 = sparsesample_reweight(adjres1, ressampledeg)
            adjres2 = sparsesample_reweight(adjres2, ressampledeg)
        return adjoverlap, adjres1, adjres2
    else:
        adjoverlap = spmoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
    return adjoverlap


if __name__ == "__main__":
    adj1 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 0, 1, 2, 3], [0, 1, 1, 2, 3]]))
    adj2 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 3, 1, 2, 3], [0, 1, 1, 2, 3]]))
    adj3 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 1,  2, 2, 2,2, 3, 3, 3], [1, 0,  2,3,4, 5, 4, 5, 6]]))
    print(spmnotoverlap_(adj1, adj2))
    print(spmoverlap_(adj1, adj2))
    print(spmoverlap_notoverlap_(adj1, adj2))
    print(sparsesample2(adj3, 2))
    print(sparsesample_reweight(adj3, 2))