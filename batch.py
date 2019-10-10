import torch
from torch_geometric.data import Data
import torch.utils.data
import pdb
import re
import numpy as np


class Dataset_mine(torch.utils.data.Dataset):
    def __init__(self, data_list):
        super(Dataset_mine, self).__init__()
        self.data = data_list
        # self.num_features = self.data[0].x.shape[1]
        # self.num_classes = len(np.unique(self.data[0].y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @property
    def num_features(self):
        return self.data[0].x.shape[1]
    @property
    def num_classes(self):
        return len(np.unique(self.data[0].y))




def __cat_dim__(key, value):
    r"""Returns the dimension for which :obj:`value` of attribute
    :obj:`key` will get concatenated when creating batches.

    .. note::

        This method is for internal use only, and should only be overridden
        if the batch concatenation process is corrupted for a specific data
        attribute.
    """
    # `*index*` and `*face*` should be concatenated in the last dimension,
    # everything else in the first dimension.
    return -1 if bool(re.search('(index|face|mask_link)', key)) else 0

def __cumsum__(key, value):
    r"""If :obj:`True`, :obj:`value` of attribute :obj:`key` is
    cumulatively summed up when creating batches.

    .. note::

        This method is for internal use only, and should only be overridden
        if the batch concatenation process is corrupted for a specific data
        attribute.
    """
    return bool(re.search('(index|face|mask_link)', key))

class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        # don't take "dists"
        keys = [key for key in keys if key!='dists']
        assert 'batch' not in keys

        batch = Batch()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum = 0
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in keys:
                item = data[key]
                item = item + cumsum if __cumsum__(key, item) else item
                batch[key].append(item)
            cumsum += num_nodes
        for key in keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(
                    batch[key], dim=__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            else:
                raise ValueError('Unsupported attribute type.')
        batch.batch = torch.cat(batch.batch, dim=-1)

        return batch.contiguous()

    @staticmethod
    def from_data_list_batch(data_list):
        # load one batch at a time
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        flatten = lambda l: [item for sublist in l for item in sublist]
        data_list = flatten(data_list)
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        # don't take "dists"
        keys = [key for key in keys if key != 'dists']
        assert 'batch' not in keys

        batch = Batch()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum = 0
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in keys:
                item = data[key]
                item = item + cumsum if __cumsum__(key, item) else item
                batch[key].append(item)
            cumsum += num_nodes
        for key in keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(
                    batch[key], dim=__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            else:
                raise ValueError('Unsupported attribute type.')
        batch.batch = torch.cat(batch.batch, dim=-1)

        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1






class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: Batch.from_data_list(data_list),
            **kwargs)

class DataLoader_batch(torch.utils.data.DataLoader):
    # each item is a batch of data
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: Batch.from_data_list_batch(data_list),
            **kwargs)