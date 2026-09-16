from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from lddm.utils import create_scatter_output_tensor
from lddm.model import gvp
from lddm.model.gvp import GVP, tuple_sum, tuple_cat, tuple_index, _split, _norm_no_nan
from lddm.scatter import scatter_mean, scatter_add, scatter_min, scatter_max, scatter_std


class ModuleDict(nn.ModuleDict):
    """ Extends the standard module dict to support any key types that can be converted to strings. """
    def __init__(self, modules: dict):
        super().__init__({str(k): v for k, v in modules.items()})

    def __getitem__(self, key):
        return super().__getitem__(str(key))

    def __setitem__(self, key, value):
        super().__setitem__(str(key), value)

    def __delitem__(self, key):
        super().__delitem__(str(key))

    def __contains__(self, key):
        return super().__contains__(str(key))


class GVPBlock(nn.Module):
    def __init__(
            self, 
            in_dims: tuple, 
            out_dims: tuple, 
            hidden_dims: tuple = None,
            n_layers: int = 1,
            activations=(F.silu, torch.sigmoid), 
            output_activations=(None, None),
            vector_gate=True,
            dropout=0.0, 
            layernorm=True,
        ):
        super(GVPBlock, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        hidden_dims = hidden_dims or out_dims

        GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)

        module_list = []
        for i in range(n_layers):
            layer_input_dim = in_dims if i == 0 else hidden_dims
            layer_output_dim = out_dims if i == n_layers - 1 else hidden_dims
            layer_activation = output_activations if i == n_layers - 1 else activations
            module_list.append(GVP_(layer_input_dim, layer_output_dim, activations=layer_activation))
            if layernorm:
                module_list.append(gvp.LayerNorm(layer_output_dim, learnable_vector_weight=True))

        self.layers = nn.Sequential(*module_list)
        self.dropout = gvp.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`
        """

        x = self.layers(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class SimpleResidualBlock(nn.Module):
    def __init__(
            self, 
            x_dim: tuple,
            dx_dim: tuple,
            **kwargs,
        ):
        super(SimpleResidualBlock, self).__init__()
        # self.ff_func = GVPBlock(tuple_sum(x_dim, dx_dim), x_dim, hidden_dims=(4 * x_dim[0], 2 * x_dim[1]), **kwargs)
        self.ff_func = GVPBlock(tuple_sum(x_dim, dx_dim), x_dim, **kwargs)

    def forward(self, x, dx):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param dx: tuple (s, V) of `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`
        """
        dx = self.ff_func(tuple_cat(x, dx))
        return tuple_sum(x, dx)


class ResidualBlock(nn.Module):
    """
    Inspired by the residual update in GVPConvLayer
    https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/__init__.py#L279

    This is more complex than SimpleResidualBlock. Note that here the node value and message are summed which halves the 
    number of trainable parameters in that layer because:
        W[x,y] = W1 x + W2 y (where W is split into two parts horizontally) and
        W(x+y) = W x + W y
    """
    def __init__(
            self, 
            x_dim: tuple,
            dx_dim: tuple,
            dropout=0.0, 
            layernorm=True,
            **kwargs,
        ):
        super(ResidualBlock, self).__init__()
        self.input_gvp = None if dx_dim == x_dim else \
            GVP(dx_dim, x_dim, activations=(None, None), vector_gate=True)
        self.pre_dropout = gvp.Dropout(dropout)
        self.pre_norm = gvp.LayerNorm(x_dim, learnable_vector_weight=True) if layernorm else None
        self.ff_func = GVPBlock(x_dim, x_dim, layernorm=False, dropout=False, **kwargs)
        self.post_dropout = gvp.Dropout(dropout)
        self.post_norm = gvp.LayerNorm(x_dim, learnable_vector_weight=True) if layernorm else None

    def forward(self, x, dx=None):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param dx: tuple (s, V) of `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`
        """
        dx = x if dx is None else dx
        if self.input_gvp is not None:
            dx = self.input_gvp(dx)
        x = self.pre_norm(tuple_sum(x, self.pre_dropout(dx)))
        dx = self.ff_func(x)
        x = self.post_norm(tuple_sum(x, self.post_dropout(dx)))
        return x


class SumAggregation:
    def __call__(self, x, index, dim=0, dim_size=None, scatter_fill_value=0):
        dim_size = dim_size or index.max() + 1
        s, v = x
        s = scatter_add(s, index, dim=dim, out=create_scatter_output_tensor(like=s, dim_size=dim_size, fill_value=scatter_fill_value))
        v = scatter_add(v, index, dim=dim, out=create_scatter_output_tensor(like=v, dim_size=dim_size, fill_value=scatter_fill_value))
        return (s, v)
    

class MeanAggregation:
    def __call__(self, x, index, dim=0, dim_size=None, scatter_fill_value=0):
        dim_size = dim_size or index.max() + 1
        s, v = x
        s = scatter_mean(s, index, dim=dim, out=create_scatter_output_tensor(like=s, dim_size=dim_size, fill_value=scatter_fill_value))
        v = scatter_mean(v, index, dim=dim, out=create_scatter_output_tensor(like=v, dim_size=dim_size, fill_value=scatter_fill_value))
        return (s, v)
    

class MultiAggregation(nn.Module):
    def __init__(self, d_in, d_out=None, sum=True, mean=True, std=True, min=True, max=True):
        """ Map features to global features """
        super().__init__()

        if d_out is None:
            d_out = d_in

        si, vi = d_in
        so, vo = d_out

        self._sum = sum
        self._mean = mean
        self._std = std
        self._min = min
        self._max = max

        assert self._sum or self._mean, "edge case not handled"

        si_concat = (self._sum + self._mean + self._std + self._min + self._max) * si + \
            (self._std + self._min + self._max) * vi
        vi_concat = (self._sum + self._mean) * vi

        self.gvp = GVPBlock((si_concat, vi_concat), d_out)

    def forward(self, x, index, dim=0, dim_size=None, scatter_fill_value=0):
        """ x: tuple (s, V) """
        s, v = x
        _s, _v = [], []

        if self._sum:
            _s.append(scatter_add(s, index, dim=dim, out=create_scatter_output_tensor(like=s, dim_size=dim_size, fill_value=scatter_fill_value)))
            _v.append(scatter_add(v, index, dim=dim, out=create_scatter_output_tensor(like=v, dim_size=dim_size, fill_value=scatter_fill_value)))

        if self._mean:
            _s.append(scatter_mean(s, index, dim=dim, out=create_scatter_output_tensor(like=s, dim_size=dim_size, fill_value=scatter_fill_value)))
            _v.append(scatter_mean(v, index, dim=dim, out=create_scatter_output_tensor(like=v, dim_size=dim_size, fill_value=scatter_fill_value)))

        vnorm = _norm_no_nan(v)

        if self._std:
            eps = 1e-6  # avoid nan in backward pass as derivative of sqrt is not defined at zero
            _s.append(scatter_std(s, index, dim=dim, out=create_scatter_output_tensor(like=s, dim_size=dim_size, fill_value=scatter_fill_value) + eps))
            _s.append(scatter_std(vnorm, index, dim=dim, out=create_scatter_output_tensor(like=vnorm, dim_size=dim_size, fill_value=scatter_fill_value) + eps))

        if self._min:
            _s.append(scatter_min(s, index, dim=dim, out=create_scatter_output_tensor(like=s, dim_size=dim_size, fill_value=scatter_fill_value))[0])
            _s.append(scatter_min(vnorm, index, dim=dim, out=create_scatter_output_tensor(like=vnorm, dim_size=dim_size, fill_value=scatter_fill_value))[0])

        if self._max:
            _s.append(scatter_max(s, index, dim=dim, out=create_scatter_output_tensor(like=s, dim_size=dim_size, fill_value=scatter_fill_value))[0])
            _s.append(scatter_max(vnorm, index, dim=dim, out=create_scatter_output_tensor(like=vnorm, dim_size=dim_size, fill_value=scatter_fill_value))[0])

        return self.gvp((torch.hstack(_s), torch.hstack(_v)))
    

class GVPMessagePassing(nn.Module):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings and returns new node and 
    edge embeddings.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    :param update_edge_attr: whether to compute an updated edge representation
    '''

    def __init__(self, in_dims_src, in_dims_dst, out_dims, edge_dims=None, *,
                 n_layers: int, aggr: str, activations: tuple, 
                 vector_gate: bool, update_edge_attr: bool):
        super(GVPMessagePassing, self).__init__()

        if edge_dims is None:
            update_edge_attr = False

        self.si_src, self.vi_src = in_dims_src
        self.si_dst, self.vi_dst = in_dims_dst
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims if edge_dims is not None else (0, 0)
        self.update_edge_attr = update_edge_attr
        
        self.message_func = GVPBlock(
            in_dims=(self.si_src + self.si_dst + self.se, self.vi_src + self.vi_dst + self.ve),
            out_dims=out_dims,
            n_layers=n_layers,
            activations=activations,
            vector_gate=vector_gate,
        )
        self.edge_func = GVPBlock(
            in_dims=(self.si_src + self.si_dst + self.se, self.vi_src + self.vi_dst + self.ve),
            out_dims=edge_dims,
            n_layers=n_layers,
            activations=activations,
            vector_gate=vector_gate,
        ) if self.update_edge_attr else None

        if aggr == "mean":
            self.aggr_fn = MeanAggregation()
        elif aggr == "sum":
            self.aggr_fn = SumAggregation()
        elif aggr == "multi":
            self.aggr_fn = MultiAggregation(out_dims)
        else:
            raise NotImplementedError(f"Aggregation '{aggr}' not implemented.")
    
    def _prepare_inputs(self, x_src, x_dst, edge_index, edge_attr=None):
        j, i = edge_index
        s_src, v_src = x_src
        s_dst, v_dst = x_dst
        s_i, v_i = s_dst[i], v_dst[i]
        s_j, v_j = s_src[j], v_src[j]
        return tuple_cat((s_j, v_j), edge_attr, (s_i, v_i)) if edge_attr is not None else tuple_cat((s_j, v_j), (s_i, v_i))
    
    def aggregate(self, m_ij, dst_index, num_dst_nodes):
        return self.aggr_fn(m_ij, dst_index, dim=0, dim_size=num_dst_nodes, scatter_fill_value=0)

    def forward(self, x_src: tuple, x_dst: tuple, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''

        x_ij = self._prepare_inputs(x_src, x_dst, edge_index, edge_attr)
        m_ij = self.message_func(x_ij)
        m_i = self.aggregate(m_ij, dst_index=edge_index[1], num_dst_nodes=x_dst[0].size(0))
        
        if self.update_edge_attr:
            edge_attr = self.edge_func(x_ij)

        return m_i, edge_attr


class GVPHeteroConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with Geometric Vector 
    Perceptrons. Residually updates node embeddings with aggregated incoming 
    messages, applies a pointwise feedforward network to node embeddings, and 
    returns updated node embeddings.

    :param conv_dims: (in_dims_src, in_dims_dst, out_dims, edge_dims)
    """
    def __init__(self, conv_dims, *, n_message: int, n_feedforward: int, 
                 drop_rate: float, aggr_nodes: str, aggr_types: str,
                 activations: tuple = (F.silu, torch.sigmoid), 
                 vector_gate: bool = True, update_edge_attr: bool = True, 
                 residual: str = 'SimpleResidualBlock'):

        super(GVPHeteroConvLayer, self).__init__()
        self.update_edge_attr = update_edge_attr

        gvp_conv = partial(GVPMessagePassing,
                           n_layers=n_message,
                           aggr=aggr_nodes,
                           activations=activations,
                           vector_gate=vector_gate,
                           update_edge_attr=update_edge_attr)
        
        self.convs = ModuleDict({k: gvp_conv(*dims) for k, dims in conv_dims.items()})
        self.edge_types = tuple(conv_dims.keys())  # ordered and immutable
        src_node_types = set(x[0] for x in self.edge_types)
        dst_node_types = set(x[-1] for x in self.edge_types)
        assert len(src_node_types - dst_node_types) == 0, "Some node types would not get updated."
        self.node_types = src_node_types | dst_node_types
        self.aggr = aggr_types

        node_dims = {k[-1]: dims[1] for k, dims in conv_dims.items()}
        message_dims = {}
        for k, dims in conv_dims.items():
            dst = k[-1]
            if self.aggr == "cat":
                message_dims[dst] = tuple_sum(message_dims[dst], dims[1]) if dst in message_dims else dims[1]
            else:
                assert dst not in message_dims or message_dims[dst] == dims[1]
                message_dims[dst] = dims[1]

        residual_update = SimpleResidualBlock if residual == 'SimpleResidualBlock' else ResidualBlock
        residual_update = partial(residual_update, n_layers=n_feedforward, activations=activations, vector_gate=vector_gate, dropout=drop_rate)
        self.node_residual_update = ModuleDict({k: residual_update(node_dims[k], message_dims[k]) for k in self.node_types})
        self.edge_residual_update = ModuleDict({k: residual_update(dims[3], dims[3]) for k, dims in conv_dims.items() if dims[3] is not None}) if self.update_edge_attr else None

    def aggregate_message_types(self, messages):
        if len(messages) == 1:
            return messages[0]  # no aggregation needed
        elif self.aggr == "sum":
            return gvp.tuple_sum(*messages)
        elif self.aggr == "cat":
            return gvp.tuple_cat(*messages)
        else:
            raise NotImplementedError(self.aggr)

    def propagate_messages(
            self, 
            node_attr_dict,
            edge_index_dict,
            edge_attr_dict,
            *args_dict,
            **kwargs_dict,
        ):
        node_out_dict = defaultdict(list)
        edge_out_dict = {}
        for edge_type in self.edge_types:
            src, rel, dst = edge_type
            edge_index = edge_index_dict[edge_type]
            edge_attr = edge_attr_dict[edge_type]

            out_node, out_edge = self.convs[edge_type](
                node_attr_dict[src], node_attr_dict[dst], edge_index, edge_attr
            )

            node_out_dict[dst].append(out_node)
            edge_out_dict[edge_type] = out_edge

        for key, values in node_out_dict.items():
            node_out_dict[key] = self.aggregate_message_types(values)

        return node_out_dict, edge_out_dict
    
    def _check_inputs(self, node_dict, edge_index_dict, edge_attr_dict, node_mask_dict):
        assert set(node_dict.keys()) == self.node_types
        assert node_mask_dict is None or set(node_mask_dict.keys()) == self.node_types
        assert set(edge_index_dict.keys()) == set(self.edge_types)
        assert set(edge_attr_dict.keys()) == set(self.edge_types)

    def forward(self, node_attr_dict, edge_index_dict, edge_attr_dict, node_mask_dict=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        self._check_inputs(node_attr_dict, edge_index_dict, edge_attr_dict, node_mask_dict)
            
        m_i_dict, e_ij_dict = self.propagate_messages(node_attr_dict, edge_index_dict, edge_attr_dict)

        for k, x in node_attr_dict.items():
            m_i = m_i_dict[k]
            node_mask = None if node_mask_dict is None else node_mask_dict[k]

            if node_mask is not None:
                x_ = x
                x, m_i = tuple_index(x, node_mask), tuple_index(m_i, node_mask)

            x = self.node_residual_update[k](x, m_i)

            if node_mask is not None:
                x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
                x = x_

            node_attr_dict[k] = x
        
        if self.update_edge_attr:
            for k, edge_attr in edge_attr_dict.items():
                # if k not in self.edge_residual_update:
                #     continue
                if edge_attr is None:
                    continue

                e_ij = e_ij_dict[k]
                edge_attr = self.edge_residual_update[k](edge_attr, e_ij)
                edge_attr_dict[k] = edge_attr

        return (node_attr_dict, edge_attr_dict) if self.update_edge_attr else node_attr_dict


class HeteroGVPGNN(nn.Module):
    """
    GVP-GNN model
    inspired by: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    and: https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/atom3d.py#L115
    """
    def __init__(
            self,
            node_in_dim_ligand, node_in_dim_pocket,
            edge_in_dim_ligand, edge_in_dim_pocket, edge_in_dim_interaction,
            node_h_dim_ligand, node_h_dim_pocket,
            edge_h_dim_ligand, edge_h_dim_pocket, edge_h_dim_interaction,
            node_out_dim_ligand=None, node_out_dim_pocket=None,
            edge_out_dim_ligand=None, edge_out_dim_pocket=None, edge_out_dim_interaction=None,
            num_layers=3, drop_rate=0.0, vector_gate=True, update_edge_attr=True, global_node_dim=None,
            aggr_nodes="multi", aggr_types="cat", residual='SimpleResidualBlock',
        ):

        super(HeteroGVPGNN, self).__init__()

        self.update_edge_attr = update_edge_attr
        self.node_in = ModuleDict({
            'ligand': GVP(node_in_dim_ligand, node_h_dim_ligand, activations=(None, None), vector_gate=vector_gate),
            'pocket': GVP(node_in_dim_pocket, node_h_dim_pocket, activations=(None, None), vector_gate=vector_gate),
        })
        self.edge_in = ModuleDict({
            ('ligand', '', 'ligand'): GVP(edge_in_dim_ligand, edge_h_dim_ligand, activations=(None, None), vector_gate=vector_gate),
            ('pocket', '', 'pocket'): GVP(edge_in_dim_pocket, edge_h_dim_pocket, activations=(None, None), vector_gate=vector_gate),
            ('ligand', '', 'pocket'): GVP(edge_in_dim_interaction, edge_h_dim_interaction, activations=(None, None), vector_gate=vector_gate),
            ('pocket', '', 'ligand'): GVP(edge_in_dim_interaction, edge_h_dim_interaction, activations=(None, None), vector_gate=vector_gate),
        })

        # tuples: in_dims_src, in_dims_dst, out_dims, edge_dims
        conv_dims = {
            ('ligand', '', 'ligand'): (node_h_dim_ligand, node_h_dim_ligand, node_h_dim_ligand, edge_h_dim_ligand),
            ('pocket', '', 'pocket'): (node_h_dim_pocket, node_h_dim_pocket, node_h_dim_pocket, edge_h_dim_pocket),
            ('ligand', '', 'pocket'): (node_h_dim_ligand, node_h_dim_pocket, node_h_dim_pocket, edge_h_dim_interaction),
            ('pocket', '', 'ligand'): (node_h_dim_pocket, node_h_dim_ligand, node_h_dim_ligand, edge_h_dim_interaction),
        }

        self.global_nodes = global_node_dim is not None
        self.supported_node_types = set(self.node_in.keys())
        if self.global_nodes:
            self.global_node_dim = global_node_dim
            conv_dims[('ligand', '', '_global')] = (node_h_dim_ligand, global_node_dim, global_node_dim, None)
            conv_dims[('_global', '', 'ligand')] = (global_node_dim, node_h_dim_ligand, node_h_dim_ligand, None)

            conv_dims[('pocket', '', '_global')] = (node_h_dim_pocket, global_node_dim, global_node_dim, None)
            conv_dims[('_global', '', 'pocket')] = (global_node_dim, node_h_dim_pocket, node_h_dim_pocket, None)

        self.layers = nn.ModuleList(
            GVPHeteroConvLayer(conv_dims,
                               n_message=3, n_feedforward=2,
                               drop_rate=drop_rate,
                               update_edge_attr=self.update_edge_attr,
                               activations=(F.silu, None),
                               vector_gate=vector_gate,
                               aggr_nodes=aggr_nodes, aggr_types=aggr_types,
                               residual=residual)
            for _ in range(num_layers)
        )

        self.node_out = ModuleDict({
            'ligand': GVP(node_h_dim_ligand, node_out_dim_ligand, activations=(None, None), vector_gate=vector_gate),
            'pocket': GVP(node_h_dim_pocket, node_out_dim_pocket, activations=(None, None), vector_gate=vector_gate) if node_out_dim_pocket is not None else None,
        })
        self.edge_out = ModuleDict({
            ('ligand', '', 'ligand'): GVP(edge_h_dim_ligand, edge_out_dim_ligand, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_ligand is not None else None,
            ('pocket', '', 'pocket'): GVP(edge_h_dim_pocket, edge_out_dim_pocket, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_pocket is not None else None,
            ('ligand', '', 'pocket'): GVP(edge_h_dim_interaction, edge_out_dim_interaction, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_interaction is not None else None,
            ('pocket', '', 'ligand'): GVP(edge_h_dim_interaction, edge_out_dim_interaction, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_interaction is not None else None,
        })

    def _add_global_nodes(self, node_attr_dict, edge_index_dict, edge_attr_dict, batch_mask_dict, batch_size, device):
        node_attr_dict['_global'] = (
            torch.zeros(batch_size, self.global_node_dim[0], device=device), 
            torch.zeros(batch_size, self.global_node_dim[1], 3, device=device), 
        )
        for _type in self.supported_node_types:
            n_nodes = node_attr_dict[_type][0].size(0)

            edge_index_dict[(_type, '', '_global')] = torch.vstack((
                torch.arange(n_nodes, device=batch_mask_dict[_type].device),
                batch_mask_dict[_type],
            ))
            edge_index_dict[('_global', '', _type)] = torch.vstack((
                batch_mask_dict[_type],
                torch.arange(n_nodes, device=batch_mask_dict[_type].device),
            ))

            edge_attr_dict[(_type, '', '_global')] = None
            edge_attr_dict[('_global', '', _type)] = None   
        
    def _remove_global_nodes(self, node_attr_dict, edge_attr_dict):
        del node_attr_dict['_global']
        for k in list(edge_attr_dict.keys()):
            if '_global' in k:
                del edge_attr_dict[k]

    def forward(self, node_attr, batch_mask, edge_index, edge_attr, batch_size=None):

        # create copies
        node_attr = {k: v for k, v in node_attr.items()}
        edge_attr = {k: v for k, v in edge_attr.items()}

        # to hidden dimension
        for k in node_attr.keys():
            node_attr[k] = self.node_in[k](node_attr[k])

        for k in edge_attr.keys():
            edge_attr[k] = self.edge_in[k](edge_attr[k])

        if self.global_nodes:
            batch_size = batch_size or len(batch_mask['ligand'].unique())
            device = batch_mask['ligand'].device
            self._add_global_nodes(node_attr, edge_index, edge_attr, batch_mask, batch_size, device)

        # convolutions
        for i, layer in enumerate(self.layers):
            out = layer(node_attr, edge_index, edge_attr)
            if self.update_edge_attr:
                node_attr, edge_attr = out
            else:
                node_attr = out
        
        if self.global_nodes:
            self._remove_global_nodes(node_attr, edge_attr)

        # to output dimension
        for k in node_attr.keys():
            node_attr[k] = self.node_out[k](node_attr[k]) \
                if self.node_out[k] is not None else None

        if self.update_edge_attr:
            for k in edge_attr.keys():
                if self.edge_out[k] is not None:
                    edge_attr[k] = self.edge_out[k](edge_attr[k])

        return node_attr, edge_attr
