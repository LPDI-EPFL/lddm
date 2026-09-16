from collections.abc import Iterable
from typing import Dict, Callable
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean
try:
    from torch_geometric.nn import radius, knn
except ImportError as e:
    print(e)  # knn edge definition won't be available

from lddm import utils
from lddm.model.gvp import _rbf, _normalize, tuple_sum, tuple_cat, tuple_index, ensure_tuple, scalar_cat, vector_cat
from lddm.model.legacy.dynamics_hetero import GVPModel
from lddm.model.hetero_gnn import HeteroGVPGNN
from lddm.model.utils import cycle_counts, eigenfeatures, map_edges, RBFEmbedding
from lddm.data.data_utils import TensorDict


class DynamicsHetero(nn.Module):
    def __init__(self, atom_nf, residue_nf, bond_dict, pocket_bond_dict, *,
                 condition_time=True,
                 num_rbf_time=None,
                 model='gvp',
                 model_params=None,
                 edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None,
                 edge_cutoff_interaction=None,
                 edge_knn_ligand=None,
                 edge_knn_pocket=None,
                 edge_knn_interaction=None,
                 add_cycle_counts=False,
                 add_spectral_feat=False,
                 reflection_equiv=False,
                 d_max=15.0,
                 num_rbf_dist=16,
                 self_conditioning=False,
                 augment_ligand_sc=False,
                 hide_uncertainty_sc=True,
                 add_all_atom_diff=False,
                 predict_confidence=False,
                 enable_masked_modeling=False,
                 add_node_features_to_edges=False,  # assign src and dst node features to each edge (usually not necessary as handled by the GNN)
                 uncertainty_act=F.softplus,
    ):

        super().__init__()

        self.model = model
        assert (edge_cutoff_ligand is None or edge_knn_ligand is None)
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_knn_l = edge_knn_ligand
        assert (edge_cutoff_pocket is None or edge_knn_pocket is None)
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_knn_p = edge_knn_pocket
        assert (edge_cutoff_interaction is None or edge_knn_interaction is None)
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_knn_i = edge_knn_interaction
        self.bond_nf = len(bond_dict)
        self.no_bond_idx = bond_dict["NOBOND"]
        self.pocket_bond_nf = len(pocket_bond_dict)
        self.add_all_atom_diff = add_all_atom_diff
        self.condition_time = condition_time
        self.predict_confidence = predict_confidence
        self.add_cycle_counts = add_cycle_counts
        self.add_spectral_feat = add_spectral_feat
        self.self_conditioning = self_conditioning
        self.augment_ligand_sc = augment_ligand_sc
        self.hide_uncertainty_sc = hide_uncertainty_sc

        # edge encoding params
        self.reflection_equiv = reflection_equiv
        self.enable_masked_modeling = enable_masked_modeling
        self.add_node_features_to_edges = add_node_features_to_edges

        # Output dimensions, always tuple (scalar, vector)
        self.atom_out_dim = (atom_nf[0], 1) if isinstance(atom_nf, Iterable) else (atom_nf, 1)
        self.residue_out_dim = (0, 0)

        if self.predict_confidence:
            self.atom_out_dim = tuple_sum(self.atom_out_dim, (1, 0))

        # Edge output dimensions, always tuple (scalar, vector)
        self.edge_out_dim = (self.bond_nf, 0)
        _edge_ligand_before_symmetrization = (model_params.edge_h_dim[0] if 'edge_h_dim' in model_params else model_params.edge_channels, 0)


        # Input dimensions, always tuple (scalar, vector)
        assert isinstance(atom_nf, int), "expected: element onehot"
        _atom_in = (atom_nf, 0)
        assert isinstance(residue_nf, Iterable), "expected: (AA-onehot, vectors to atoms)"
        _residue_in = tuple(residue_nf)
        _residue_atom_dim = residue_nf[1]

        if self.add_cycle_counts:
            _atom_in = tuple_sum(_atom_in, (3, 0))
        if self.add_spectral_feat:
            _atom_in = tuple_sum(_atom_in, (5, 0))
        if self.enable_masked_modeling:
            _atom_in = tuple_sum(_atom_in, (2, 0))

        if self.condition_time:
            self.time_dim = num_rbf_time if num_rbf_time is not None else 1
            if num_rbf_time is None:
                self.time_embedder = lambda t: t.view(t.size(0), 1)
            else:
                self.time_embedder = RBFEmbedding(d_min=0.0, d_max=1.0, d_count=self.time_dim)

            _atom_in = tuple_sum(_atom_in, (self.time_dim, 0))
            _residue_in = tuple_sum(_residue_in, (self.time_dim, 0))
        else:
            print('Warning: dynamics model is NOT conditioned on time.')            

        # Edge input dimensions
        # NOTE: there are two types of edges: (1) those already specified in the 
        #   data, and (2) additional edges for the computational graph. Not all 
        #   features are available for the purely computational edges.
        _edge_ligand_in_data = (self.bond_nf, 0)
            
        if self.enable_masked_modeling:
            _edge_ligand_in_data = tuple_sum(_edge_ligand_in_data, (1, 0))

        _edge_pocket_in_data = (self.pocket_bond_nf, 0)

        # Self-conditioning
        if self.self_conditioning:
            # Update input dimensions
            self.atom_sc_dims = (
                self.atom_out_dim[0] - self.predict_confidence * self.hide_uncertainty_sc,  
                self.atom_out_dim[1]
            )
            _atom_in = tuple_sum(_atom_in, self.atom_sc_dims)

            self.residue_sc_dims = self.residue_out_dim
            _residue_in = tuple_sum(_residue_in, self.residue_sc_dims)

            if self.augment_ligand_sc:
                _atom_in = tuple_sum(_atom_in, (0, 1))

            self.edge_sc_dims = self.edge_out_dim
            _edge_ligand_in_data = tuple_sum(_edge_ligand_in_data, self.edge_sc_dims)

            # Create memory
            self.prev_ligand = None
            self.prev_residues = None

        # Geometry-based edge features, these are computed for all edges, not only the ones specified in the data
        distance_emb_dim = 1 if num_rbf_dist is None else num_rbf_dist
        if num_rbf_dist is None:
            self.distance_embedder = lambda d: d.view(d.size(0), 1)
        else:
            self.distance_embedder = RBFEmbedding(d_min=0.0, d_max=d_max, d_count=num_rbf_dist)

        _edge_ligand_in = tuple_sum(_edge_ligand_in_data, (distance_emb_dim, 1 if self.reflection_equiv else 2))

        _n_dist_residue = _residue_atom_dim ** 2 if self.add_all_atom_diff else 1
        _edge_pocket_in = tuple_sum(_edge_pocket_in_data, (_n_dist_residue * distance_emb_dim, _n_dist_residue))

        _n_dist_interaction = _residue_atom_dim if self.add_all_atom_diff else 1
        _edge_interaction_in = (_n_dist_interaction * distance_emb_dim, _n_dist_interaction)


        if self.add_node_features_to_edges:
            _edge_ligand_in = tuple_sum(_edge_ligand_in, _atom_in)  # src node
            _edge_ligand_in = tuple_sum(_edge_ligand_in, _atom_in)  # dst node

            _edge_pocket_in = tuple_sum(_edge_pocket_in, _residue_in)  # src node
            _edge_pocket_in = tuple_sum(_edge_pocket_in, _residue_in)  # dst node

            _edge_interaction_in = tuple_sum(_edge_interaction_in, _atom_in)  # atom node
            _edge_interaction_in = tuple_sum(_edge_interaction_in, _residue_in)  # residue node


        # Embeddings for newly added edges
        self.ligand_nobond_s = nn.Parameter(torch.zeros(_edge_ligand_in_data[0]), requires_grad=True)  # trainable embedding
        self.ligand_nobond_v = nn.Parameter(torch.zeros(_edge_ligand_in_data[1], 3), requires_grad=False)  # frozen to avoid breaking equivariance
        self.pocket_nobond_s = nn.Parameter(torch.zeros(_edge_pocket_in_data[0]), requires_grad=True)
        self.pocket_nobond_v = nn.Parameter(torch.zeros(_edge_pocket_in_data[1], 3), requires_grad=False)  # frozen to avoid breaking equivariance

        # Set up the core neural network
        if model == 'gvp':

            self.net = GVPModel(
                node_in_dim_ligand=_atom_in,
                node_in_dim_pocket=_residue_in,
                edge_in_dim_ligand=_edge_ligand_in,
                edge_in_dim_pocket=_edge_pocket_in,
                edge_in_dim_interaction=_edge_interaction_in,
                node_h_dim_ligand=model_params.node_h_dim,
                node_h_dim_pocket=model_params.node_h_dim,
                edge_h_dim_ligand=model_params.edge_h_dim,
                edge_h_dim_pocket=model_params.edge_h_dim,
                edge_h_dim_interaction=model_params.edge_h_dim,
                node_out_dim_ligand=self.atom_out_dim,
                node_out_dim_pocket=self.residue_out_dim,
                edge_out_dim_ligand=_edge_ligand_before_symmetrization,
                edge_out_dim_pocket=None,
                edge_out_dim_interaction=None,
                num_layers=model_params.n_layers,
                drop_rate=model_params.dropout,
                vector_gate=model_params.vector_gate,
                update_edge_attr=True,
            )

        elif model == 'gvp_v2':

            self.net = HeteroGVPGNN(
                node_in_dim_ligand=_atom_in,
                node_in_dim_pocket=_residue_in,
                edge_in_dim_ligand=_edge_ligand_in,
                edge_in_dim_pocket=_edge_pocket_in,
                edge_in_dim_interaction=_edge_interaction_in,
                node_h_dim_ligand=model_params.node_h_dim,
                node_h_dim_pocket=model_params.node_h_dim,
                edge_h_dim_ligand=model_params.edge_h_dim,
                edge_h_dim_pocket=model_params.edge_h_dim,
                edge_h_dim_interaction=model_params.edge_h_dim,
                node_out_dim_ligand=self.atom_out_dim,
                node_out_dim_pocket=self.residue_out_dim,
                edge_out_dim_ligand=_edge_ligand_before_symmetrization,
                edge_out_dim_pocket=None,
                edge_out_dim_interaction=None,
                num_layers=model_params.n_layers,
                drop_rate=model_params.dropout,
                vector_gate=model_params.vector_gate,
                update_edge_attr=True,
                global_node_dim=model_params.global_node_dim,
                aggr_nodes=vars(model_params).get("aggr_nodes", "multi"),
                aggr_types=vars(model_params).get("aggr_types", "cat"),
                residual=vars(model_params).get("residual", "SimpleResidualBlock"),
            )

        else:
            raise NotImplementedError(f"{model} is not available")

        assert self.edge_out_dim[1] == 0
        assert _edge_ligand_before_symmetrization[1] == 0
        self.edge_decoder = nn.Sequential(
            nn.Linear(_edge_ligand_before_symmetrization[0], _edge_ligand_before_symmetrization[0]),
            torch.nn.SiLU(),
            nn.Linear(_edge_ligand_before_symmetrization[0], self.edge_out_dim[0])
        )

        self.uncertainty_act = uncertainty_act

    def make_sc_input(self, h_atoms, e_atoms, h_residues, e_residues, pred_ligand=None, pred_residues=None, sc_transform: Dict[str, Callable] = None):

        # (1) Additional inputs
        if pred_ligand is None:
            # Create zero tensors
            h_atoms_sc = (
                torch.zeros(len(h_atoms), self.atom_sc_dims[0], dtype=h_atoms.dtype, device=h_atoms.device),
                torch.zeros(len(h_atoms), self.atom_sc_dims[1], 3, dtype=h_atoms.dtype, device=h_atoms.device),
            )
            e_atoms_sc = (
                torch.zeros(len(e_atoms), self.edge_sc_dims[0], dtype=e_atoms.dtype, device=e_atoms.device),
                torch.zeros(len(e_atoms), self.edge_sc_dims[1], 3, dtype=e_atoms.dtype, device=e_atoms.device),
            )
        
        else:
            # Assemble self-conditioning tensors
            pred_h_scalar = pred_ligand['logits_h']
            if self.hide_uncertainty_sc:
                pred_h_scalar = F.one_hot(pred_h_scalar.argmax(-1), num_classes=pred_h_scalar.size(-1)).to(pred_h_scalar.dtype)
            if self.predict_confidence and not self.hide_uncertainty_sc:
                pred_h_scalar = torch.cat([pred_h_scalar, pred_ligand['uncertainty_vel'].unsqueeze(1)], dim=-1)
            h_atoms_sc = (pred_h_scalar, pred_ligand['vel'].unsqueeze(1))

            e_atoms_sc = pred_ligand['logits_e']
            if self.hide_uncertainty_sc:
                e_atoms_sc = F.one_hot(e_atoms_sc.argmax(-1), num_classes=e_atoms_sc.size(-1)).to(e_atoms_sc.dtype)

            if self.augment_ligand_sc:
                h_atoms_sc = (h_atoms_sc[0], torch.cat(
                    [h_atoms_sc[1], sc_transform['atoms'](pred_ligand['vel'].unsqueeze(1))], dim=1))

        # Residues have no self-conditioning inputs (no residue-level predictions)
        h_residues_sc = None


        # (2) Combine with the original input features
        h_atoms = (torch.cat([h_atoms, h_atoms_sc[0]], dim=-1), h_atoms_sc[1])  # always tuple

        if isinstance(e_atoms_sc, tuple):
            e_atoms = (torch.cat([e_atoms, e_atoms_sc[0]], dim=-1), e_atoms_sc[1])
        else:
            e_atoms = torch.cat([e_atoms, e_atoms_sc], dim=-1)

        if isinstance(h_residues_sc, tuple):
            h_residues = (torch.cat([h_residues, h_residues_sc[0]], dim=-1), h_residues_sc[1])
        elif h_residues_sc is None:
            h_residues = h_residues  # only the original feature
        else:
            h_residues = torch.cat([h_residues, h_residues_sc], dim=-1)
        
        e_residues = e_residues  # only the original feature

        return h_atoms, e_atoms, h_residues, e_residues

    def forward(
        self, x_atoms, h_atoms, mask_atoms, pocket, t, bonds_ligand=None, sc_transform=None,
        known_x=None, known_h=None, known_e=None,
    ):
        """
        Implements self-conditioning as in https://arxiv.org/abs/2208.04202
        """

        # default input features
        h_atoms = h_atoms
        e_atoms = bonds_ligand[1]
        h_residues = pocket['one_hot']
        e_residues = pocket['bond_one_hot']

        if self.self_conditioning:

            # Sampling: use previous prediction in all but the first time step
            if not self.training and t.min() > 0.0:
                assert t.min() == t.max(), "currently only supports sampling at same time steps"
                assert self.prev_ligand is not None

            else:
                # Training: use 50% zeros and 50% predictions with detached gradients
                if self.training and random.random() > 0.5:
                    with torch.no_grad():
                        h_atoms_tmp, e_atoms_tmp, h_residues_tmp, e_residues_tmp = self.make_sc_input(
                            h_atoms, e_atoms, h_residues, e_residues, 
                            pred_ligand=None, pred_residues=None, sc_transform=sc_transform,
                        )
                        self.prev_ligand, self.prev_residues = self._forward(
                            x_atoms, h_atoms_tmp, mask_atoms, bonds_ligand[0], e_atoms_tmp,
                            pocket['x'], h_residues_tmp, pocket['mask'], pocket['bonds'], e_residues_tmp,
                            pocket, t, known_x, known_h, known_e,
                        )

                # use zeros for first sampling step and 50% of training
                else:
                    self.prev_ligand = None
                    self.prev_residues = None

            # concatenate original inputs and self-conditioning variables
            h_atoms, e_atoms, h_residues, e_residues = self.make_sc_input(
                h_atoms, e_atoms, h_residues, e_residues, self.prev_ligand, 
                self.prev_residues, sc_transform=sc_transform,
            )

        pred_ligand, pred_residues = self._forward(
            x_atoms, h_atoms, mask_atoms, bonds_ligand[0], e_atoms,
            pocket['x'], h_residues, pocket['mask'], pocket['bonds'], e_residues,
            pocket, t, known_x, known_h, known_e,
        )

        if self.self_conditioning and not self.training:
            self.prev_ligand = TensorDict(**pred_ligand).deepcopy()
            self.prev_residues = TensorDict(**pred_residues).deepcopy()

        return pred_ligand, pred_residues

    def _compute_extra_features(self, batch_mask, edge_indices):

        feat = torch.zeros(len(batch_mask), 0, device=batch_mask.device)

        if not (self.add_cycle_counts or self.add_spectral_feat):
            return feat

        # adj = batch_mask[:, None] == batch_mask[None, :]
        adj = torch.zeros(len(batch_mask), len(batch_mask), dtype=bool, device=batch_mask.device)
        adj[edge_indices[0], edge_indices[1]] = True

        # make undirected if necessary
        adj = (adj | adj.T)

        A = adj.float()

        if self.add_cycle_counts:
            cycle_features = cycle_counts(A)
            cycle_features[cycle_features > 10] = 10  # avoid large values

            feat = torch.cat([feat, cycle_features], dim=-1)

        if self.add_spectral_feat:
            feat = torch.cat([feat, eigenfeatures(A, batch_mask)], dim=-1)

        return feat

    @staticmethod
    def _symmetrize_edge_features(edge_feat, edges, num_nodes, out_edges=None):
        if out_edges is None:
            out_edges = edges
        out = torch.zeros(
            (num_nodes, num_nodes, *edge_feat.shape[1:]),
            dtype=edge_feat.dtype, device=edge_feat.device
        )
        out[edges[0], edges[1]] = edge_feat
        out = (out + out.transpose(0, 1)) * 0.5

        # return only relevant elements
        return out[out_edges[0], out_edges[1]]

    def _forward(
        self, 
        x_atoms, h_atoms, mask_atoms, ligand_bond_indices, ligand_bond_features,
        x_residues, h_residues, mask_residues, pocket_bond_indices, pocket_bond_features,
        pocket, t,
        known_x=None, known_h=None, known_e=None,
    ):
        """
        :param x_atoms: ligand coordinates (n, 3)
        :param h_atoms: ligand atom features, either tensor or tuple (s, V)
        :param mask_atoms: ligand batch mask
        :param ligand_bond_indices: (2, n_bonds)
        :param ligand_bond_features: either tensor or tuple (s, V)
        :param x_residues: residue coordinates (nR, 3)
        :param h_residues: residue features, either tensor or tuple (s, V)
        :param mask_residues: residue batch mask
        :param pocket_bond_indices: (2, nR_bonds)
        :param pocket_bond_features: either tensor or tuple (s, V)
        :param pocket: pocket object for context
        :param t: time step
        :param known_x: mask of coordinates that are "known"
        :param known_h: mask of atom types that are "known"
        :param known_e: mask of edges that are "known"
        :return:
        """

        # Features should always be tuples (s, V) to avoid many if-else constructions
        h_atoms = ensure_tuple(h_atoms)
        h_residues = ensure_tuple(h_residues)
        ligand_bond_features = ensure_tuple(ligand_bond_features)
        pocket_bond_features = ensure_tuple(pocket_bond_features)

        if 'v' in pocket:
            h_residues = vector_cat(h_residues, pocket['v'])

        if self.enable_masked_modeling:
            known_x_feat = known_x.float().unsqueeze(-1)
            known_h_feat = known_h.float().unsqueeze(-1)
            known_e_feat = known_e.float().unsqueeze(-1)
            h_atoms = scalar_cat(h_atoms, known_x_feat, known_h_feat)
            ligand_bond_features = scalar_cat(ligand_bond_features, known_e_feat)

        if self.condition_time:
            t = self.time_embedder(t.squeeze(-1))
            if isinstance(h_atoms, tuple) :
                h_atoms = (torch.cat([h_atoms[0], t[mask_atoms]], dim=1), h_atoms[1]) 
            else: 
                h_atoms = torch.cat([h_atoms, t[mask_atoms]], dim=1)
            if isinstance(h_residues, tuple):
                h_residues = (torch.cat([h_residues[0], t[mask_residues]], dim=1), h_residues[1])
            else:
                h_residues = torch.cat([h_residues, t[mask_residues]], dim=1)     

        # Add auxiliary features that depend on the adjacency matrix
        # WARNING: assumes the bond type is always encoded first in the scalar feature
        ligand_bond_exists = ligand_bond_features[0][:, :self.bond_nf].argmax(-1) != self.no_bond_idx
        extra_features = self._compute_extra_features(mask_atoms, ligand_bond_indices[:, ligand_bond_exists])
        h_atoms = scalar_cat(h_atoms, extra_features)  

        ligand_edge_indices, ligand_edge_types = self.expand_bonds(ligand_bond_indices, ligand_bond_features)
        pocket_edge_indices, pocket_edge_types = self.expand_bonds(pocket_bond_indices, pocket_bond_features)

        # Process edges and encode in shared feature space
        edge_index_dict, edge_attr_dict = self.get_edges(
            x_atoms, h_atoms, mask_atoms, ligand_edge_indices, ligand_edge_types,
            x_residues, h_residues, mask_residues, pocket['v'], pocket_edge_indices, pocket_edge_types, 
        )

        node_attr_dict = {
            'ligand': h_atoms,
            'pocket': h_residues,
        }
        batch_mask_dict = {
            'ligand': mask_atoms,
            'pocket': mask_residues,
        }

        # Forward pass
        out_node_attr, out_edge_attr = self.net(
            node_attr_dict, batch_mask_dict, edge_index_dict, edge_attr_dict)

        h_final_atoms = out_node_attr['ligand'][0]
        vel = out_node_attr['ligand'][1].squeeze(-2)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final_atoms)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
                h_final_atoms[torch.isnan(h_final_atoms)] = 0.0
                print("[WARNING] NaN detected in network output")
            else:
                raise ValueError("NaN detected in network output")

        # predict edge type
        edge_final = out_edge_attr[('ligand', '', 'ligand')]
        edges = edge_index_dict[('ligand', '', 'ligand')]

        # Symmetrize & return upper triangular elements only (matching the input)
        edge_logits = self._symmetrize_edge_features(edge_final, edges, len(mask_atoms), out_edges=ligand_bond_indices)
        edge_final_atoms = self.edge_decoder(edge_logits)

        pred_ligand = {'vel': vel, 'logits_e': edge_final_atoms}

        if self.predict_confidence:
            pred_ligand['logits_h'] = h_final_atoms[:, :-1]
            pred_ligand['uncertainty_vel'] = self.uncertainty_act(h_final_atoms[:, -1])
        else:
            pred_ligand['logits_h'] = h_final_atoms

        pred_residues = {}

        return pred_ligand, pred_residues

    @staticmethod
    def expand_bonds(bond_indices, bond_features):
        """
        Bonds are only defined in one direction but should be undirected.
        We change this here.
        Note: we use the term 'bond' for one-directional edges and 'edge' for 
              the bi-directional version.
        """
        # make sure messages are passed both ways
        edge_indices = torch.cat([bond_indices, bond_indices.flip(dims=[0])], dim=1)
        # edge_types = torch.cat([bond_features, bond_features], dim=0)
        edge_types = tuple_cat(bond_features, bond_features, dim=0)
        return edge_indices, edge_types

    def get_edges(self, x_ligand, h_ligand, batch_mask_ligand, edges_ligand, edge_feat_ligand,
                  x_pocket, h_pocket, batch_mask_pocket, atom_vectors_pocket, edges_pocket, edge_feat_pocket,
                  self_edges_ligand=True, self_edges_pocket=False):
        
        # Adjacency matrix
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)
        elif self.edge_knn_l is not None:
            _r, _c = knn(x_ligand, x_ligand, k=self.edge_knn_l + 1, 
                         batch_x=batch_mask_ligand, batch_y=batch_mask_ligand)  # +1 accounts for self-edges
            proximity_matrix = torch.zeros_like(adj_ligand)
            proximity_matrix[_r, _c] = True
            adj_ligand = adj_ligand & proximity_matrix 

        # Add missing bonds if they got removed
        adj_ligand[edges_ligand[0], edges_ligand[1]] = True

        if not self_edges_ligand:
            adj_ligand = adj_ligand ^ torch.eye(*adj_ligand.size(), out=torch.empty_like(adj_ligand))

        if self.edge_cutoff_p is not None and len(x_pocket) > 0:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)
        elif self.edge_knn_p is not None and len(x_pocket) > 0:
            _r, _c = knn(x_pocket, x_pocket, k=self.edge_knn_p + 1, 
                         batch_x=batch_mask_pocket, batch_y=batch_mask_pocket)  # +1 accounts for self-edges
            proximity_matrix = torch.zeros_like(adj_pocket)
            proximity_matrix[_r, _c] = True
            adj_pocket = adj_pocket & proximity_matrix 

        # Add missing bonds if they got removed
        adj_pocket[edges_pocket[0], edges_pocket[1]] = True

        if not self_edges_pocket:
            adj_pocket = adj_pocket ^ torch.eye(*adj_pocket.size(), out=torch.empty_like(adj_pocket))

        if self.edge_cutoff_i is not None and len(x_pocket) > 0:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)
        elif self.edge_knn_i is not None and len(x_pocket) > 0:
            _r, _c = knn(x_pocket, x_ligand, k=self.edge_knn_i, 
                         batch_x=batch_mask_pocket, batch_y=batch_mask_ligand)
            proximity_matrix = torch.zeros_like(adj_cross)
            proximity_matrix[_r, _c] = True
            adj_cross = adj_cross & proximity_matrix

        # ligand-ligand edge features
        edges_ligand_updated = torch.stack(torch.where(adj_ligand), dim=0)
        feat_ligand_s = self.ligand_nobond_s.repeat(edges_ligand_updated.size(1), 1)  # initialise with dummy features
        feat_ligand_v = self.ligand_nobond_v.repeat(edges_ligand_updated.size(1), 1, 1)
        ligand_edge_mapping, _mask = map_edges(edges_ligand, edges_ligand_updated, shape=adj_ligand.size())
        feat_ligand_s[ligand_edge_mapping] = edge_feat_ligand[0][_mask]  # replace with available features
        feat_ligand_v[ligand_edge_mapping] = edge_feat_ligand[1][_mask]
        feat_ligand = self.ligand_edge_features(x_ligand, edges_ligand_updated, batch_mask_ligand, edge_attr=(feat_ligand_s, feat_ligand_v))

        # residue-residue edge features
        edges_pocket_updated = torch.stack(torch.where(adj_pocket), dim=0)
        feat_pocket_s = self.pocket_nobond_s.repeat(edges_pocket_updated.size(1), 1)  # initialise with dummy features
        feat_pocket_v = self.pocket_nobond_v.repeat(edges_pocket_updated.size(1), 1, 1)
        pocket_edge_mapping, _mask = map_edges(edges_pocket, edges_pocket_updated, shape=adj_pocket.size())
        feat_pocket_s[pocket_edge_mapping] = edge_feat_pocket[0][_mask]  # replace with available features
        feat_pocket_v[pocket_edge_mapping] = edge_feat_pocket[1][_mask]
        feat_pocket = self.pocket_edge_features(x_pocket, atom_vectors_pocket, edges_pocket_updated, edge_attr=(feat_pocket_s, feat_pocket_v))

        # ligand-residue edge features
        edges_cross = torch.stack(torch.where(adj_cross), dim=0)
        feat_cross = self.cross_edge_features(x_ligand, x_pocket, atom_vectors_pocket, edges_cross)

        if self.add_node_features_to_edges:
            feat_ligand = tuple_cat(feat_ligand, tuple_index(h_ligand, edges_ligand_updated[0]), tuple_index(h_ligand, edges_ligand_updated[1]))
            feat_pocket = tuple_cat(feat_pocket, tuple_index(h_pocket, edges_pocket_updated[0]), tuple_index(h_pocket, edges_pocket_updated[1]))
            feat_cross = tuple_cat(feat_cross, tuple_index(h_ligand, edges_cross[0]), tuple_index(h_pocket, edges_cross[1]))

        edge_index = {
            ('ligand', '', 'ligand'): edges_ligand_updated,
            ('pocket', '', 'pocket'): edges_pocket_updated,
            ('ligand', '', 'pocket'): edges_cross,
            ('pocket', '', 'ligand'): edges_cross.flip(dims=[0]),
        }

        edge_attr = {
            ('ligand', '', 'ligand'): feat_ligand,
            ('pocket', '', 'pocket'): feat_pocket,
            ('ligand', '', 'pocket'): feat_cross,
            ('pocket', '', 'ligand'): feat_cross,
        }

        return edge_index, edge_attr

    def ligand_edge_features(self, x, edge_index, batch_mask=None, edge_attr=None):
        """
        :param x: (n, 3)
        :param edge_index:
        :param batch_mask:
        :param edge_attr:
        :return: scalar and vector-valued edge features
        """
        row, col = edge_index
        coord_diff = x[row] - x[col]
        dist = coord_diff.norm(dim=-1)

        edge_s = self.distance_embedder(dist)
        edge_v = _normalize(coord_diff).unsqueeze(-2)

        if edge_attr is not None:
            edge_s = torch.cat([edge_s, edge_attr[0]], dim=1)
            edge_v = torch.cat([edge_v, edge_attr[1]], dim=1)

        # self.reflection_equiv: bool, use reflection-sensitive feature based on
        #                        the cross product if False
        if not self.reflection_equiv:
            mean = scatter_mean(x, batch_mask, dim=0,
                                dim_size=batch_mask.max() + 1)
            row, col = edge_index
            cross = torch.cross(x[row] - mean[batch_mask[row]],
                                x[col] - mean[batch_mask[col]], dim=1)
            cross = _normalize(cross).unsqueeze(-2)

            edge_v = torch.cat([edge_v, cross], dim=-2)

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)

    def pocket_edge_features(self, x, v, edge_index, edge_attr=None):
        """
        :param x: (nR, 3)
        :param v: (nR, nA, 3)
        :param edge_index:
        :param edge_attr:
        :return: scalar and vector-valued edge features
        """
        row, col = edge_index

        if self.add_all_atom_diff:
            all_coord = v + x.unsqueeze(1)  # (nR, nA, 3)
            coord_diff = all_coord[row, :, None, :] - all_coord[col, None, :, :]  # (nB, nA, nA, 3)
            coord_diff = coord_diff.flatten(1, 2)
            dist = coord_diff.norm(dim=-1)  # (nB, nA^2)
            dist = self.distance_embedder(dist)  # (nB, nA^2, rdb_dim)
            dist = dist.flatten(1, 2)
            coord_diff = _normalize(coord_diff)
        else:
            coord_diff = x[row] - x[col]
            dist = coord_diff.norm(dim=-1)
            dist = self.distance_embedder(dist)
            coord_diff = _normalize(coord_diff).unsqueeze(-2)

        edge_s = dist
        edge_v = coord_diff

        if edge_attr is not None:
            edge_s = torch.cat([edge_s, edge_attr[0]], dim=1)
            edge_v = torch.cat([edge_v, edge_attr[1]], dim=1)

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)

    def cross_edge_features(self, x_ligand, x_pocket, v_pocket, edge_index):
        """
        :param x_ligand: (n, 3)
        :param x_pocket: (nR, 3)
        :param v_pocket: (nR, nA, 3)
        :param edge_index: first row indexes into the ligand tensors, second row into the pocket tensors

        :return: scalar and vector-valued edge features
        """
        ligand_idx, pocket_idx = edge_index

        if self.add_all_atom_diff:
            all_coord_pocket = v_pocket + x_pocket.unsqueeze(1)  # (nR, nA, 3)
            coord_diff = x_ligand[ligand_idx, None, :] - all_coord_pocket[pocket_idx]  # (nB, nA, 3)
            dist = coord_diff.norm(dim=-1)  # (nB, nA)
            dist = self.distance_embedder(dist)  # (nB, nA, rdb_dim)
            dist = dist.flatten(1, 2)
            coord_diff = _normalize(coord_diff)
        else:
            coord_diff = x_ligand[ligand_idx] - x_pocket[pocket_idx]
            dist = coord_diff.norm(dim=-1)  # (nB, nA)
            dist = self.distance_embedder(dist)
            coord_diff = _normalize(coord_diff).unsqueeze(-2)

        edge_s = dist
        edge_v = coord_diff

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)
