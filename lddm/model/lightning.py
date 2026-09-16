import logging
import warnings
import tempfile
from typing import Optional, Union
from time import time
from pathlib import Path
from functools import partial
from itertools import accumulate
from typing import Dict, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.distributed import all_gather_object
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from rdkit import Chem
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, SubsetRandomSampler

import lddm.utils as utils
from lddm.constants import atom_encoder, atom_decoder, aa_encoder, aa_decoder, \
    bond_encoder, bond_decoder, residue_bond_encoder, \
    residue_bond_decoder, aa_atom_index, aa_atom_mask, max_num_atoms_per_residue
from lddm.config.main import LDDMConfig
from lddm.data.dataset import ProcessedDataset, ClusteredDataset, MixedDataset, DynamicBatchIterableDataset
from lddm.data import data_utils
from lddm.data.data_utils import Residues, Ligand, collate_entity, split_entity, edge_mask_by_node_mask, extract_substructure
from lddm.data.transforms import AppendVirtualNodesInCoM, AddVirtualNodesToLigand, MaskLigand, CenterData, RandomRotation
from lddm.model.flows import ICFM, CoordICFM
from lddm.model.markov_bridge import UniformPriorMarkovBridge, MarginalPriorMarkovBridge
from lddm.model.legacy.dynamics_hetero import DynamicsHetero as DynamicsHeteroV1
from lddm.model.dynamics import DynamicsHetero as DynamicsHeteroV2
from lddm.model.diffusion_utils import DistributionNodes
from lddm.model.loss_utils import TimestepWeights, UniformTimestepSampler, TrapezoidalTimestepSampler
from lddm.model import samplers
from lddm.model.utils import WarmupScheduler
from lddm.data.molecule_builder import pocket_to_rdkit, mols_to_pdbfile
from lddm.utils import CategoricalDistribution
from lddm.data.molecule_builder import build_molecule
from lddm.scatter import scatter_mean, scatter_add
from tqdm import tqdm


# derive additional constants
aa_atom_mask_tensor = torch.tensor([aa_atom_mask[aa] for aa in aa_decoder])
aa_atom_decoder = {aa: {v: k for k, v in aa_atom_index[aa].items()} for aa in aa_decoder}
aa_atom_type_tensor = torch.tensor([[atom_encoder.get(aa_atom_decoder[aa].get(i, '-')[0], -42)
                                     for i in range(14)] for aa in aa_decoder])


class LDDM(pl.LightningModule):
    def __init__(
            self,
            *,
            config: LDDMConfig | dict = None,
            **kwargs,
    ):
        super(LDDM, self).__init__()
        if isinstance(config, LDDMConfig):
            self.save_hyperparameters(config.to_dict())
        else:
            self.save_hyperparameters()

        if config is None:
            # when continued with LDDM.load_from_checkpoint, it loads the hyperparameters that were stored as a dict
            config = LDDMConfig.from_dict(kwargs)
        elif isinstance(config, dict):
            config = LDDMConfig.from_dict(config, strict=False)

        train_params = config.train_params
        dataset_params = config.dataset_params
        loss_params = config.loss_params
        eval_params = config.eval_params
        predictor_params = config.predictor_params
        simulation_params = config.simulation_params
        self.featurization_config = config.featurization_config
        self.ignore_featurization_mismatch = config.ignore_featurization_mismatch

        # Check for invalid configurations
        self.pocket_representation = self.featurization_config.pocket_representation
        assert self.pocket_representation == 'CA+'

        
        self.augment_ligand_sc = predictor_params.augment_ligand_sc

        assert not (simulation_params.predict_confidence and
                    (not predictor_params.heterogeneous_graph or simulation_params.predict_final))

        assert not (simulation_params.predict_confidence and simulation_params.predict_error)
        assert not (simulation_params.predict_error and loss_params.lambda_x_error is None) 

        assert dataset_params.val.mixed_dataset is None, "Currently MixedDatasets are only supported as training datasets" 
        assert dataset_params.test.mixed_dataset is None, "Currently MixedDatasets are only supported as training datasets" 
        # assert not (simulation_params.predict_confidence and (simulation_params.prior_h == 'gaussian' or simulation_params.prior_e == 'gaussian'))

        # Batch size must either be fixed or selected dynamincally, but not both at the same time
        assert not (train_params.batch_size is None and train_params.max_tokens_per_batch is None)
        assert train_params.batch_size is None or train_params.max_tokens_per_batch is None

        if eval_params.n_sampling_steps % eval_params.keep_frames != 0:
            print(f'WARNING: Mismatch between eval_params.n_sampling_steps={eval_params.n_sampling_steps} and eval_params.keep_frames={eval_params.keep_frames}')
            print(f'WARNING: setting eval_params.keep_frames={eval_params.n_sampling_steps}')
            eval_params.keep_frames = eval_params.n_sampling_steps

        # Set parameters
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.virtual_nodes = config.virtual_nodes
        self.debug = config.debug
        self.overfit = config.overfit
        self.predict_confidence = simulation_params.predict_confidence
        self.uncertainty_is_variance = simulation_params.uncertainty_is_variance
        self.predict_error = simulation_params.predict_error
        self.masked_modeling = simulation_params.masked_modeling

        if self.virtual_nodes:
            self.add_virtual_min = self.virtual_nodes[0]
            self.add_virtual_max = self.virtual_nodes[1]

        # Training parameters
        self.dataset_spec = dataset_params
        self.batch_size = train_params.batch_size
        self.max_tokens_per_batch = train_params.max_tokens_per_batch
        self.lr = train_params.lr
        self.betas = train_params.betas
        self.weight_decay = train_params.weight_decay
        self.lr_step_size = train_params.lr_step_size
        self.lr_gamma = train_params.lr_gamma
        self.lr_warmup_steps = train_params.lr_warmup_steps
        self.num_workers = train_params.num_workers
        self.clip_grad = train_params.clip_grad
        if self.clip_grad:
            self.gradnorm_queue = utils.Queue()
            # Add large value that will be flushed.
            self.gradnorm_queue.add(3000)

        # Evaluation parameters
        self.outdir = eval_params.outdir
        self.eval_batch_size = eval_params.eval_batch_size
        self.eval_epochs = eval_params.eval_epochs
        self.sample_epochs = eval_params.sample_epochs
        self.visualize_sample_epoch = eval_params.visualize_sample_epoch
        self.visualize_chain_epoch = eval_params.visualize_chain_epoch
        self.sample_with_ground_truth_size = eval_params.sample_with_ground_truth_size
        self.n_loss_per_sample = eval_params.n_loss_per_sample
        self.n_eval_samples = eval_params.n_eval_samples
        self.step_spacing = eval_params.step_spacing
        self.n_visualize_samples = eval_params.n_visualize_samples
        self.keep_frames = eval_params.keep_frames
        self.gnina = train_params.gnina
        self.excluded_evaluators = eval_params.exclude_evaluators
        self.reference_mols_validity3d = eval_params.reference_mols_validity3d

        # Feature encoders/decoders
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.bond_encoder = bond_encoder
        self.bond_decoder = bond_decoder
        self.aa_encoder = aa_encoder
        self.aa_decoder = aa_decoder
        self.residue_bond_encoder = residue_bond_encoder
        self.residue_bond_decoder = residue_bond_decoder

        self.atom_nf = len(self.atom_decoder)
        self.residue_nf = len(self.aa_decoder)
        self.aa_atom_index = aa_atom_index
        if self.pocket_representation == 'CA+':
            self.residue_nf = (self.residue_nf, max_num_atoms_per_residue)  # (s, V)
        self.bond_nf = len(self.bond_decoder)
        self.pocket_bond_nf = len(self.residue_bond_decoder)
        self.x_dim = 3

        # Set up the neural network
        self.dynamics = self.init_model(predictor_params)

        # Simulation params (modules will be initialised in setup)
        self.module_x = None
        self.module_h = None
        self.module_e = None
        self.sigma_x = simulation_params.sigma_x
        self.noise_scale_x = simulation_params.noise_scale_x
        self.predict_final = simulation_params.predict_final
        self.scheduler_x = simulation_params.scheduler_x
        self.scheduler_h = simulation_params.scheduler_h
        self.scheduler_e = simulation_params.scheduler_e
        self.prior_x_type  = simulation_params.prior_x
        self.prior_h_type = simulation_params.prior_h
        self.prior_e_type = simulation_params.prior_e
        self.mbm_loss_type = loss_params.discrete_loss
        self.coord_loss_type = loss_params.coord_loss_type
        self.coord_prior_scale = simulation_params.prior_scale_x
        self.train_steps = simulation_params.n_steps
        self.train_step_size = 1 / self.train_steps
        self.atom_type_histogram = None
        self.bond_type_histogram = None

        # Loss parameters
        self.loss_reduce = loss_params.reduce
        self.lambda_x = loss_params.lambda_x
        self.lambda_h = loss_params.lambda_h
        self.lambda_e = loss_params.lambda_e
        self.lambda_clash = loss_params.lambda_clash
        self.lambda_x_error = loss_params.lambda_x_error
        self.regularize_uncertainty = loss_params.regularize_uncertainty
        self.optimal_transport = loss_params.optimal_transport

        assert not (self.optimal_transport and self.prior_x_type == 'harmonic'), 'OT not supported with harmonic prior'

        if loss_params.timestep_weights is not None:
            weight_type = loss_params.timestep_weights.split('_')[0]
            kwargs = loss_params.timestep_weights.split('_')[1:]
            kwargs = {x.split('=')[0]: float(x.split('=')[1]) for x in kwargs}
            self.timestep_weights = TimestepWeights(weight_type, **kwargs)
        else:
            self.timestep_weights = None

        if loss_params.timestep_sampler == 'uniform':
            self.timestep_sampler = UniformTimestepSampler()
        elif loss_params.timestep_sampler.startswith('trapezoid'):
            height_left = float(loss_params.timestep_sampler.split('_')[1])
            self.timestep_sampler = TrapezoidalTimestepSampler(height_left=height_left)
        else:
            raise NotImplementedError()

        # Sampling
        self.sampler = getattr(samplers, simulation_params.sampler)(self)
        self.T_sampling = eval_params.n_sampling_steps
        self.size_distribution = None  # initialized only if needed
        self.size_histogram_file = simulation_params.size_histogram_file
        self.size_histogram = None

        # Metrics, initialized only if needed
        self.train_smiles = None
        self.evaluator = None
        self.ligand_atom_type_distribution = None
        self.ligand_bond_type_distribution = None

        # containers for metric aggregation
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.masking = train_params.masking
        self.eval_masking = eval_params.masking
        self.apply_random_rotations = train_params.apply_random_rotations

    @property
    def default_size_histogram_path(self):
        return Path(self.dataset_spec.meta_info_root, 'size_distribution.npy')

    @property
    def default_atom_type_histogram_path(self):
        return Path(self.dataset_spec.meta_info_root, 'ligand_type_histogram.npy')
    
    @property
    def default_bond_type_histogram_path(self):
        return Path(self.dataset_spec.meta_info_root, 'ligand_bond_type_histogram.npy')
    
    @property
    def default_training_smiles_path(self):
        return Path(self.dataset_spec.meta_info_root, 'train_smiles.npy')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['size_histogram'] = self.size_histogram  # for sampling
        checkpoint['atom_type_histogram'] = self.atom_type_histogram  # for Markov Bridge with marginal prior
        checkpoint['bond_type_histogram'] = self.bond_type_histogram  # for Markov Bridge with marginal prior

    def on_load_checkpoint(self, checkpoint):

        # for backwards compatibility
        if "prior_h" in checkpoint["state_dict"]:
            checkpoint['atom_type_histogram'] = checkpoint["state_dict"]["prior_h"]
            del checkpoint["state_dict"]["prior_h"]
        if "prior_e" in checkpoint["state_dict"]:
            checkpoint['bond_type_histogram'] = checkpoint["state_dict"]["prior_e"]
            del checkpoint["state_dict"]["prior_e"]

        def safe_load(key):
            if getattr(self, key) is not None:
                logging.debug(f"Overriding attribute '{key}' with value from checkpoint.")
            setattr(self, key, checkpoint.get(key))

        safe_load('size_histogram')
        safe_load('atom_type_histogram')
        safe_load('bond_type_histogram')

        # repeat basic setup because Trainer.fit(..., ckpt_path=...) calls setup() before loading the checkpoint
        self.setup("basics")

    def init_model(self, predictor_params):
        model_type = predictor_params.backbone
        dynamics_cls = {
            "hetero_v1": DynamicsHeteroV1,
            "hetero_v2": partial(DynamicsHeteroV2, add_node_features_to_edges=predictor_params.add_node_features_to_edges, hide_uncertainty_sc=predictor_params.hide_uncertainty_sc),
        }[vars(predictor_params).get("dynamics_version", "hetero_v1")]
        activation_functions = {
            'softplus': F.softplus,
            'exp': torch.exp,
        }
        return dynamics_cls(
            atom_nf=self.atom_nf,
            residue_nf=self.residue_nf,
            bond_dict=self.bond_encoder,
            pocket_bond_dict=self.residue_bond_encoder,
            model=model_type,
            num_rbf_time=predictor_params.__dict__.get('num_rbf_time'),
            model_params=predictor_params.backbone_params,
            edge_cutoff_ligand=predictor_params.edge_cutoff_ligand,
            edge_cutoff_pocket=predictor_params.edge_cutoff_pocket,
            edge_cutoff_interaction=predictor_params.edge_cutoff_interaction,
            edge_knn_ligand=predictor_params.edge_knn_ligand,
            edge_knn_pocket=predictor_params.edge_knn_pocket,
            edge_knn_interaction=predictor_params.edge_knn_interaction,
            add_cycle_counts=predictor_params.cycle_counts,
            add_spectral_feat=predictor_params.spectral_feat,
            reflection_equiv=predictor_params.reflection_equivariant,
            d_max=predictor_params.d_max,
            num_rbf_dist=predictor_params.num_rbf,
            self_conditioning=predictor_params.self_conditioning,
            augment_ligand_sc=self.augment_ligand_sc,
            add_all_atom_diff=predictor_params.add_all_atom_diff,
            predict_confidence=self.predict_confidence or self.predict_error,
            enable_masked_modeling=predictor_params.enable_masked_modeling,
            uncertainty_act=activation_functions[predictor_params.uncertainty_act],
        )

    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                amsgrad=True,
                betas=tuple(self.betas),
                weight_decay=self.weight_decay #1e-12
            ),
        ]

        if self.lr_step_size is None or self.lr_gamma is None:
            lr_schedulers = []
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=self.lr_step_size, gamma=self.lr_gamma)
            if self.lr_warmup_steps is not None:
                lr_scheduler = WarmupScheduler(optimizers[0], lr_scheduler, warmup_steps=self.lr_warmup_steps)
            lr_schedulers = [lr_scheduler]
        return optimizers, lr_schedulers

    def setup(self, stage: Optional[str] = None):
        self.setup_simulation_modules()
        self.setup_sampling()

        if stage == 'fit':
            self.train_dataset = self.get_dataset(stage='train')
            self.val_dataset = self.get_dataset(stage='val')
            if not self.ignore_featurization_mismatch:
                assert self.train_dataset.config.featurization == self.featurization_config, \
                    self.featurization_config.get_diff(self.train_dataset.config.featurization)
                assert self.val_dataset.config.featurization == self.featurization_config, \
                    self.featurization_config.get_diff(self.val_dataset.config.featurization)
            assert not (self.global_rank > 0 and isinstance(self.val_dataset, IterableDataset)), 'Validation IterableDataset is not compatible with DDP'
            self.setup_metrics()
        elif stage == 'val' or stage == 'validate':
            self.val_dataset = self.get_dataset(stage='val')
            if not self.ignore_featurization_mismatch:
                assert self.val_dataset.config.featurization == self.featurization_config, \
                    self.featurization_config.get_diff(self.val_dataset.config.featurization)
            assert not (self.global_rank > 0 and isinstance(self.val_dataset, IterableDataset)), 'Validation IterableDataset is not compatible with DDP'
            self.setup_metrics()
        elif stage == 'test':
            self.test_dataset = self.get_dataset(stage='test')
            if not self.ignore_featurization_mismatch:
                assert self.test_dataset.config.featurization == self.featurization_config, \
                    self.featurization_config.get_diff(self.test_dataset.config.featurization)
            self.setup_metrics()
        elif stage == 'generation':
            pass
        elif stage == 'basics':
            pass
        else:
            raise NotImplementedError

    def setup_simulation_modules(self):
        # Initialize objects for each variable type
        
        # Continuous variables
        logging.debug("Initializing CoordICFM for ligand coordinates")
        self.module_x = CoordICFM(
            sigma=self.sigma_x,
            loss_type=self.coord_loss_type,
            prior_scale=self.coord_prior_scale,
            scheduler_args=self.scheduler_x,
            predict_final=self.predict_final,
            noise_scale=self.noise_scale_x,
        )
        
        # Discrete variables
        if self.atom_type_histogram is None:  # load from file
            self.atom_type_histogram = torch.tensor(CategoricalDistribution.from_file(self.default_atom_type_histogram_path, self.atom_encoder).p)
            logging.debug(f"Atom type histogram loaded from {self.default_atom_type_histogram_path}.")
        hist_for_printing = {k: self.atom_type_histogram[i].item() for k, i in self.atom_encoder.items()}
        logging.debug(f"Atom type histogram: {hist_for_printing}.")
        if self.prior_h_type == 'uniform':
            self.module_h = UniformPriorMarkovBridge(
                dim=self.atom_nf, 
                loss_type=self.mbm_loss_type, 
                num_steps=self.train_steps, 
                scheduler_args=self.scheduler_h,
            )
        elif self.prior_h_type == 'marginal':
            self.module_h = MarginalPriorMarkovBridge(
                dim=self.atom_nf,
                prior_p=self.atom_type_histogram,
                loss_type=self.mbm_loss_type,
                num_steps=self.train_steps,
                scheduler_args=self.scheduler_h,
            )

        if self.bond_type_histogram is None:  # load from file
            self.bond_type_histogram = torch.tensor(CategoricalDistribution.from_file(self.default_bond_type_histogram_path, self.bond_encoder).p)
            logging.debug(f"Bond type histogram loaded from {self.default_bond_type_histogram_path}.")
        hist_for_printing = {k: self.bond_type_histogram[i].item() for k, i in self.bond_encoder.items()}
        logging.debug(f"Bond type histogram: {hist_for_printing}.")
        if self.prior_e_type == 'uniform':
            self.module_e = UniformPriorMarkovBridge(
                dim=self.bond_nf, 
                loss_type=self.mbm_loss_type, 
                num_steps=self.train_steps,
                scheduler_args=self.scheduler_e,
            )
        elif self.prior_e_type == 'marginal':
            self.module_e = MarginalPriorMarkovBridge(
                dim=self.bond_nf,
                prior_p=self.bond_type_histogram,
                loss_type=self.mbm_loss_type,
                num_steps=self.train_steps,
                scheduler_args=self.scheduler_e,
            )

    def setup_sampling(self):

        size_histogram = None
        
        # 1) If user provides the histogram explicitly, use it
        if self.size_histogram_file is not None:
            size_histogram = np.load(self.size_histogram_file).tolist()
            logging.debug(f"Size distribution loaded from {self.size_histogram_file} (user input).")

        # 2) If not explicitly provided, check if it was loaded from a checkpoint
        elif self.size_histogram is not None:
            size_histogram = self.size_histogram
            logging.debug(f"Size distribution already available (usually because it was loaded from a checkpoint).")

        # 3) If the first two options fail, attempt to load the file from the default location
        elif Path(self.default_size_histogram_path).exists():
            size_histogram = np.load(self.default_size_histogram_path).tolist()
            logging.debug(f"Size distribution loaded from {self.default_size_histogram_path} (default location).")
        
        if size_histogram is not None:
            self.size_histogram = size_histogram
            self.size_distribution = DistributionNodes(size_histogram)

    def setup_metrics(self):
        from lddm.sbdd_metrics.metrics import FullEvaluator
        
        # For metrics
        smiles_file = self.default_training_smiles_path
        self.train_smiles = None if not smiles_file.exists() else np.load(smiles_file)

        logging.debug(f"Setting up evaluators")
        self.evaluator = FullEvaluator(
            gnina=self.gnina, 
            reference_mols=self.train_smiles, 
            exclude_evaluators=self.excluded_evaluators,
            reference_mols_validity3d=self.reference_mols_validity3d,
        )

        self.ligand_atom_type_distribution = CategoricalDistribution.from_file(self.default_atom_type_histogram_path, self.atom_encoder)
        self.ligand_bond_type_distribution = CategoricalDistribution.from_file(self.default_bond_type_histogram_path, self.bond_encoder)

    def get_transforms(self, stage):

        transforms = [
            MaskLigand(self.masking if stage == 'train' else self.eval_masking)
        ]

        if stage == 'train':
            transforms.append(CenterData())  # NOTE: for non-translation equivariant architectures, this should also be applied for validation/testing
            if self.apply_random_rotations:
                transforms.append(RandomRotation())

        # when sampling we don't append virtual nodes as we might need access to the ground truth size
        if self.virtual_nodes and stage == "train":
            transforms.append(AddVirtualNodesToLigand(
                atom_encoder=atom_encoder,
                bond_encoder=bond_encoder,
                add_min=self.add_virtual_min,
                add_max=self.add_virtual_max
            ))

        return transforms

    def get_dataset(self, stage, dataset_params=None):
        logging.debug(f"Setting up {stage} dataset")

        dataset_params = dataset_params or getattr(self.dataset_spec, stage)

        # handle MixedDataset as special case
        if "mixed_dataset" in dataset_params and dataset_params.mixed_dataset is not None:
            return MixedDataset(
                datasets=[self.get_dataset(stage, utils.dict_to_namespace(params)) 
                          for params in dataset_params.mixed_dataset.datasets],
                weights=dataset_params.mixed_dataset.weights, 
                samples_per_epoch=dataset_params.mixed_dataset.num_samples_per_epoch_per_gpu,
            )

        # the standard cases are handled below

        # we want to know if something goes wrong on the validation or test set
        catch_errors = stage == "train"
        transforms = self.get_transforms(stage)

        # if self.debug:
        #     stage = 'val'

        if dataset_params.clustered_dataset:
            # val/test sets should be deterministic
            return ClusteredDataset(
                data_path=dataset_params.datadir,
                stage=stage,
                transforms=transforms, 
                catch_errors=catch_errors, 
                deterministic=stage in {'val', 'test'},
            )

        return ProcessedDataset(
            data_path=dataset_params.datadir,
            stage=stage, 
            transforms=transforms, 
            catch_errors=catch_errors
        )

    def train_dataloader(self):
        if self.max_tokens_per_batch is not None:
            assert isinstance(self.train_dataset, IterableDataset), "Dynamic batching is only available for iterable datasets."
            return DataLoader(
                dataset=DynamicBatchIterableDataset(
                    self.train_dataset, 
                    max_tokens_per_batch=self.max_tokens_per_batch,
                ),
                batch_size=None,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=ProcessedDataset.collate_fn,
                pin_memory=True,
            )
        else:
            shuffle = None if self.overfit else False if isinstance(self.train_dataset, IterableDataset) else True
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=shuffle,
                sampler=SubsetRandomSampler([0]) if self.overfit else None,
                num_workers=self.num_workers,
                collate_fn=ProcessedDataset.collate_fn,
                pin_memory=True,
            )

    def val_dataloader(self):
        if self.overfit:
            return self.train_dataloader()
        
        sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset) if self.trainer.num_devices > 1 else None
        return DataLoader(self.val_dataset, self.eval_batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=ProcessedDataset.collate_fn,
                          sampler=sampler,
                          pin_memory=True)

    def test_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset) if self.trainer.num_devices > 1 else None
        return DataLoader(self.test_dataset, self.eval_batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=ProcessedDataset.collate_fn,
                          sampler=sampler,
                          pin_memory=True)

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def aggregate_metrics(self, step_outputs, prefix):
        if len(step_outputs) == 0:
            return

        if 'timestep' in step_outputs[0]:
            timesteps = torch.cat([x['timestep'] for x in step_outputs]).squeeze()

        if 'loss_per_sample' in step_outputs[0]:
            losses = torch.cat([x['loss_per_sample'] for x in step_outputs])
            pearson_corr = torch.corrcoef(torch.stack([timesteps, losses], dim=0))[0, 1]
            self.log(f'corr_loss_timestep/{prefix}', pearson_corr, prog_bar=False)

        if 'eps_hat_norm' in step_outputs[0]:
            eps_norm = torch.cat([x['eps_hat_norm'] for x in step_outputs])
            pearson_corr = torch.corrcoef(torch.stack([timesteps, eps_norm], dim=0))[0, 1]
            self.log(f'corr_eps_timestep/{prefix}', pearson_corr, prog_bar=False)

    def on_train_epoch_end(self):
        self.aggregate_metrics(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()

    def project_final_ligand(self, zt_ligand, pred_ligand, t):
        pred_z1_ligand = zt_ligand.deepcopy()
        pred_z1_ligand['x'] = self.module_x.get_z1_given_zt_and_pred(zt_ligand['x'], pred_ligand['vel'], None, t, zt_ligand['mask'])
        pred_z1_ligand['h'] = pred_ligand['logits_h'].softmax(dim=-1)
        pred_z1_ligand['h'] = self.module_h.sample_categorical(pred_z1_ligand['h'])
        pred_z1_ligand['e'] = pred_ligand['logits_e'].softmax(dim=-1)
        pred_z1_ligand['e'] = self.module_e.sample_categorical(pred_z1_ligand['e'])

        # Masking
        pred_z1_ligand = pred_z1_ligand.insert_known_variables()
        return pred_z1_ligand

    def get_sc_transform_fn(self, zt_x, t, ligand_mask):
        sc_transform = {}

        if self.augment_ligand_sc:
            # sc_transform['atoms'] = partial(self.module_x.get_z1_given_zt_and_pred, zt=zs_x, z0=None, t=t, batch_mask=lig_mask)
            sc_transform['atoms'] = lambda pred: (self.module_x.get_z1_given_zt_and_pred(
                zt_x, pred.squeeze(1), None, t, ligand_mask) - zt_x).unsqueeze(1)

        return sc_transform

    def compute_loss(self, ligand, pocket, return_info=False):
        """
        Samples time steps and computes network predictions
        """
        ligand = Ligand(**ligand)
        pocket = Residues(**pocket)

        batch_size = len(ligand['size'])

        # For sampling the coordinate prior
        pocket_com = pocket.get_com()

        # # Normalize pocket coordinates
        # pocket['x'] = self.module_x.normalize(pocket['x'])

        # Sample a timestep t for each example in batch
        t = self.timestep_sampler(batch_size, device=ligand['x'].device).unsqueeze(-1)

        # Noise
        if self.prior_x_type == 'harmonic':
            ptr = utils.batch_to_ptr(ligand['mask'])
            edge_idx = torch.where(ligand['bond_one_hot'][:, bond_encoder['NOBOND']] == 0.)[0]
            edges = ligand['bonds'][:, edge_idx].T
            if 'virtual_mask' in ligand:
                virtual_nodes = torch.where(ligand['virtual_mask'])[0]
                virtual_edges = []
                for vn in virtual_nodes:
                    batch_idx = ligand['mask'][vn] # batch index of virtual node
                    real_nodes = torch.where((ligand['mask'] == batch_idx) & (~ligand['virtual_mask']))[0] # all real nodes of the current molecule
                    selected_node = real_nodes[torch.randperm(real_nodes.size(0))[0]] # select a random node to make an edge with virtual node
                    virtual_edges.append([selected_node, vn])
                
                virtual_edges = torch.tensor(virtual_edges, device=edges.device)
                edges = torch.cat([edges, virtual_edges], dim=0)
            z0_x = self.module_x.sample_z0(pocket_com, ligand['mask'], edges=edges, ptr=ptr)
        else:
            z0_x = self.module_x.sample_z0(pocket_com, ligand['mask'])
        
        if self.optimal_transport:
            z0_x = utils.optimal_remapping(z0_x, ligand['x'], ligand['mask'])

        z0_h = self.module_h.sample_z0(ligand['mask'])
        z0_e = self.module_e.sample_z0(ligand['bond_mask'])
        zt_x = self.module_x.sample_zt(z0_x, ligand['x'], t, ligand['mask'])
        zt_h = self.module_h.sample_zt(z0_h, ligand['one_hot'], t, ligand['mask'])
        zt_e = self.module_e.sample_zt(z0_e, ligand['bond_one_hot'], t, ligand['bond_mask'])

        # Apply fragment mask
        zt_x = Ligand.replace_values(zt_x, ligand['x'], where=ligand['known_x'])
        zt_h = Ligand.replace_values(zt_h, ligand['one_hot'], where=ligand['known_h'])
        zt_e = Ligand.replace_values(zt_e, ligand['bond_one_hot'], where=ligand['known_e'])

        # Predict denoising
        sc_transform = self.get_sc_transform_fn(zt_x, t, ligand['mask'])
        # sc_transform = None
        pred_ligand, _ = self.dynamics(
            zt_x, zt_h, ligand['mask'], pocket, t,
            bonds_ligand=(ligand['bonds'], zt_e), sc_transform=sc_transform,
            known_x=ligand['known_x'], known_h=ligand['known_h'], known_e=ligand['known_e'],
        )

        # Compute L2 loss
        if self.predict_confidence:
            loss_x = self.module_x.compute_loss(pred_ligand['vel'], z0_x, ligand['x'], zt_x, t, ligand['mask'], reduce='none', batch_size=batch_size)

            # compute confidence regularization
            k = self.module_x.dim  # pred.size(-1)
            sigma2 = pred_ligand['uncertainty_vel'] if self.uncertainty_is_variance else pred_ligand['uncertainty_vel'] ** 2
            loss_x = loss_x / (2 * sigma2) + k / 2 * torch.log(sigma2)

            if self.regularize_uncertainty is not None:
                loss_x = loss_x + self.regularize_uncertainty * (pred_ligand['uncertainty_vel'] - 1) ** 2

            loss_x = self.module_x.reduce_loss(loss_x[~ligand['known_x']], ligand['mask'][~ligand['known_x']], reduce=self.loss_reduce, batch_size=batch_size)
        elif self.predict_error:
            loss_x = self.module_x.compute_loss(pred_ligand['vel'], z0_x, ligand['x'], zt_x, t, ligand['mask'], reduce='none', batch_size=batch_size)
            predicted_error = pred_ligand['uncertainty_vel']
            true_error = loss_x.detach() if self.module_x.loss_type == "L1" else torch.sqrt(loss_x.detach())
            loss_x_error = F.mse_loss(predicted_error, true_error, reduction='none')

            loss_x = self.module_x.reduce_loss(loss_x[~ligand['known_x']], ligand['mask'][~ligand['known_x']], reduce=self.loss_reduce, batch_size=batch_size)
            loss_x_error = scatter_mean(loss_x_error[~ligand['known_x']], ligand['mask'][~ligand['known_x']], dim=0, dim_size=batch_size)

        else:
            loss_x = self.module_x.compute_loss(pred_ligand['vel'], z0_x, ligand['x'], zt_x, t, ligand['mask'], reduce=self.loss_reduce, known_mask=ligand['known_x'], batch_size=batch_size)

        # Loss for categorical variables
        if isinstance(self.module_h, UniformPriorMarkovBridge) or isinstance(self.module_h, MarginalPriorMarkovBridge):
            t_next = torch.clamp(t + self.train_step_size, max=1.0)
            loss_h = self.module_h.compute_loss(pred_ligand['logits_h'], zt_h, ligand['one_hot'], ligand['mask'], t, t_next, reduce=self.loss_reduce, known_mask=ligand['known_h'], batch_size=batch_size)
        elif isinstance(self.module_h, ICFM):
            loss_h = self.module_h.compute_loss(pred_ligand['logits_h'], z0_h, ligand['one_hot'], zt_h, t, ligand['mask'], reduce=self.loss_reduce, known_mask=ligand['known_h'], batch_size=batch_size)

        if isinstance(self.module_e, UniformPriorMarkovBridge) or isinstance(self.module_e, MarginalPriorMarkovBridge):
            t_next = torch.clamp(t + self.train_step_size, max=1.0)
            loss_e = self.module_e.compute_loss(pred_ligand['logits_e'], zt_e, ligand['bond_one_hot'], ligand['bond_mask'], t, t_next, reduce=self.loss_reduce, known_mask=ligand['known_e'], batch_size=batch_size)
        elif isinstance(self.module_e, ICFM):
            loss_e = self.module_e.compute_loss(pred_ligand['logits_e'], z0_e, ligand['bond_one_hot'], zt_e, t, ligand['bond_mask'], reduce=self.loss_reduce, known_mask=ligand['known_e'], batch_size=batch_size)
        
        loss = self.lambda_x * loss_x

        if self.predict_error:
            loss += self.lambda_x_error * loss_x_error

        if loss_h.numel() > 0:
            loss += self.lambda_h * loss_h

        if loss_e.numel() > 0:
            loss += self.lambda_e * loss_e

        if self.timestep_weights is not None:
            w_t = self.timestep_weights(t).squeeze()
            loss = w_t * loss

        loss = loss.mean(0)
        info = {
            'loss_x': loss_x.mean().item(),
            'loss_h': loss_h.mean().item(),
            'loss_e': loss_e.mean().item(),
        }
        if self.lambda_clash is not None:
            info['loss_clash'] = loss_clash.mean().item()
        if self.predict_error:
            info['loss_x_error'] = loss_x_error.mean().item()
        if self.predict_confidence or self.predict_error:
            relevant_sigma_x = pred_ligand['uncertainty_vel'][~ligand['known_x']]
            relevant_sigma_x_mask = ligand['mask'][~ligand['known_x']]
            sigma_x_mol = scatter_mean(relevant_sigma_x, relevant_sigma_x_mask, dim=0, dim_size=batch_size)
            info['pearson_sigma_x'] = torch.corrcoef(torch.stack([sigma_x_mol.detach(), t.squeeze(-1)]))[0, 1].item()
            info['pearson_sigma_x'] = torch.corrcoef(torch.stack([sigma_x_mol.detach(), t.squeeze(-1)]))[0, 1].item()
            info['mean_sigma_x'] = sigma_x_mol.mean().item()
            
            entropy_h = Categorical(logits=pred_ligand['logits_h']).entropy()
            relevant_entropy_h = entropy_h[~ligand['known_h']]
            relevant_entropy_h_mask = ligand['mask'][~ligand['known_h']]
            entropy_h_mol = scatter_mean(relevant_entropy_h, relevant_entropy_h_mask, dim=0, dim_size=batch_size)
            info['pearson_entropy_h'] = torch.corrcoef(torch.stack([entropy_h_mol.detach(), t.squeeze(-1)]))[0, 1].item()
            info['pearson_entropy_h'] = torch.corrcoef(torch.stack([entropy_h_mol.detach(), t.squeeze(-1)]))[0, 1].item()
            info['mean_entropy_h'] = entropy_h_mol.mean().item()

            entropy_e = Categorical(logits=pred_ligand['logits_e']).entropy()
            relevant_entropy_e = entropy_e[~ligand['known_e']]
            relevant_entropy_e_mask = ligand['bond_mask'][~ligand['known_e']]
            entropy_e_mol = scatter_mean(relevant_entropy_e, relevant_entropy_e_mask, dim=0, dim_size=batch_size)
            info['pearson_entropy_e'] = torch.corrcoef(torch.stack([entropy_e_mol.detach(), t.squeeze(-1)]))[0, 1].item()
            info['pearson_entropy_e'] = torch.corrcoef(torch.stack([entropy_e_mol.detach(), t.squeeze(-1)]))[0, 1].item()
            info['mean_entropy_e'] = entropy_e_mol.mean().item()

        return (loss, info) if return_info else loss

    def training_step(self, data, *args):
        ligand, pocket = data['ligand'], data['pocket']
        try:
            loss, info = self.compute_loss(ligand, pocket, return_info=True)
            info["batch_size"] = len(ligand["size"])
        except RuntimeError as e:
            # this is not supported for multi-GPU
            if self.trainer.num_devices < 2 and 'out of memory' in str(e):
                print('WARNING: ran out of memory, skipping to the next batch')
                return None
            else:
                raise e

        log_dict = {k: v for k, v in info.items() if isinstance(v, (float, int))
                    or torch.numel(v) <= 1}

        self.log_metrics({'loss': loss, **log_dict}, 'train', batch_size=len(ligand['size']), sync_dist=True)

        out = {'loss': loss, **info}
        self.training_step_outputs.append(out)
        return out

    def validation_step(self, data, *args):
        ligand = Ligand(**data['ligand']) 
        pocket = Residues(**data['pocket'])

        # Compute the loss N times and average to get a better estimate
        loss_list, info_list = [], []
        self.dynamics.train()  # currently necessary to make self-conditioning work
        for _ in range(self.n_loss_per_sample):
            loss, info = self.compute_loss(ligand.deepcopy(),
                                           pocket.deepcopy(),
                                           return_info=True)
            loss_list.append(loss.item())
            info_list.append(info)
        self.dynamics.eval()
        if len(loss_list) >= 1:
            loss = np.mean(loss_list)
            info = {k: np.mean([x[k] for x in info_list]) for k in info_list[0]}
            self.log_metrics({'loss': loss, **info}, 'val', batch_size=len(data['ligand']['size']), sync_dist=True)

        if (self.current_epoch + 1) % self.sample_epochs == 0:
            # Sample
            rdmols, rdpockets, _ = self.sample(
                data=data,
                n_samples=self.n_eval_samples,
                step_spacing=self.step_spacing,
                num_nodes="ground_truth" if self.sample_with_ground_truth_size else None,
            )

            out = {
                'ligands': rdmols,
                'pockets': rdpockets,
                'receptor_files': [Path(self.dataset_spec.val.datadir, 'val', x) for x in data['pocket']['name']]
            }
            self.validation_step_outputs.append(out)
        
            return out
        
        return  # None

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}_step_{self.global_step}')
        outdir.mkdir(exist_ok=True, parents=True)

        rdmols = [m for x in self.validation_step_outputs for m in x['ligands']]
        rdpockets = [p for x in self.validation_step_outputs for p in x['pockets']]
        self.validation_step_outputs.clear()

        self.save_sampled_molecules(rdmols=rdmols, rdpockets=rdpockets, outdir=outdir)
        self.compute_and_log_kl_divergence(rdmols)
        self.compute_and_log_evaluation_metrics(rdmols=rdmols, receptors=(rdpockets if len(rdpockets) != 0 else None))
        self.save_sampling_trajectory(outdir=outdir)

    # NOTE: temporary fix of this Lightning bug:
    # https://github.com/Lightning-AI/pytorch-lightning/discussions/18110
    # Without it resume training has a strange behavior and fails
    @property
    def total_batch_idx(self) -> int:
        """Returns the current batch index (across epochs)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return max(0, self.batch_progress.total.ready - 1)

    @property
    def batch_idx(self) -> int:
        """Returns the current batch index (within this epoch)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return max(0, self.batch_progress.current.ready - 1)
    
    def save_sampled_molecules(self, rdmols, rdpockets, outdir):
        if (self.current_epoch + 1) % self.visualize_sample_epoch == 0 and self.trainer.is_global_zero:
            tic = time()
            
            # center for better visualization
            rdmols_to_visualize = rdmols[:self.n_visualize_samples]
            rdpockets_to_visualize = rdpockets[:self.n_visualize_samples]
            for m, p in zip(rdmols_to_visualize, rdpockets_to_visualize):
                center = m.GetConformer().GetPositions().mean(axis=0)
                for i in range(m.GetNumAtoms()):
                    x, y, z = m.GetConformer().GetPositions()[i] - center
                    m.GetConformer().SetAtomPosition(i, (x, y, z))
                for i in range(p.GetNumAtoms()):
                    x, y, z = p.GetConformer().GetPositions()[i] - center
                    p.GetConformer().SetAtomPosition(i, (x, y, z))
            
            # save molecules and pockets
            utils.write_sdf_file(Path(outdir, 'molecules.sdf'), rdmols_to_visualize)
            utils.write_sdf_file(Path(outdir, 'pockets.sdf'), rdpockets_to_visualize)
            print(f'Sample visualization took {time() - tic:.2f} seconds')
    
    def compute_and_log_kl_divergence(self, rdmols):
        ligand_atom_types = [atom_encoder[a.GetSymbol()] for m in rdmols for a in m.GetAtoms()]
        ligand_bond_types = []
        for m in rdmols:
            bonds = m.GetBonds()
            no_bonds = m.GetNumAtoms() * (m.GetNumAtoms() - 1) // 2 - m.GetNumBonds()
            ligand_bond_types += [bond_encoder['NOBOND']] * no_bonds
            for b in bonds:
                ligand_bond_types.append(bond_encoder[b.GetBondType().name])
        
        if self.trainer.num_devices > 1:
            self.trainer.strategy.barrier()
            gathered_ligand_atom_types = [None for _ in range(self.trainer.num_devices)]
            gathered_ligand_bond_types = [None for _ in range(self.trainer.num_devices)]
            all_gather_object(gathered_ligand_atom_types, ligand_atom_types)
            all_gather_object(gathered_ligand_bond_types, ligand_bond_types)
            ligand_atom_types = [x for subset in gathered_ligand_atom_types for x in subset]
            ligand_bond_types = [x for subset in gathered_ligand_bond_types for x in subset]

        if self.trainer.is_global_zero:
            # Distributions of node and edge types
            kl_div_atom = (
                self.ligand_atom_type_distribution.kl_divergence(ligand_atom_types)
                if self.ligand_atom_type_distribution is not None else -1
            )
            kl_div_bond = (
                self.ligand_bond_type_distribution.kl_divergence(ligand_bond_types)
                if self.ligand_bond_type_distribution is not None else -1
            )
            results = {
                'kl_div_atom_types': kl_div_atom,
                'kl_div_bond_types': kl_div_bond,
            }
            self.log_metrics(results, 'val')

    def compute_and_log_evaluation_metrics(self, rdmols, receptors=None):
        from lddm.sbdd_metrics.evaluation import VALIDITY_METRIC_NAME, aggregated_metrics, collection_metrics

        results = []
        rdmols_iterator = tqdm(rdmols, desc='FullEvaluator') if self.trainer.is_global_zero else rdmols
        if receptors is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                for mol, receptor in zip(rdmols_iterator, receptors):
                    receptor_path = Path(tmpdir, 'receptor.pdb')
                    Chem.MolToPDBFile(receptor, str(receptor_path))
                    results.append(self.evaluator(mol, receptor_path))
        else:
            self.evaluator = FullEvaluator(pb_conf='mol')
            for mol in rdmols_iterator:
                results.append(self.evaluator(mol))

        if self.trainer.num_devices > 1:
            self.trainer.strategy.barrier()
            gathered_results = [None for _ in range(self.trainer.num_devices)]
            all_gather_object(gathered_results, results)
            results = [x for subset in gathered_results for x in subset]

        if self.trainer.is_global_zero:
            table = pd.DataFrame(results)
            agg_results = aggregated_metrics(table, self.evaluator.dtypes, VALIDITY_METRIC_NAME).fillna(0)
            agg_results['metric'] = agg_results['metric'].str.replace('.', '/')
            col_results = collection_metrics(
                table=table, 
                reference_smiles=self.train_smiles, 
                validity_metric_name=VALIDITY_METRIC_NAME, 
                exclude_evaluators=['fcd', 'ring_system_distribution']
            )
            col_results['metric'] = 'collection/' + col_results['metric']
            all_results = pd.concat([agg_results, col_results])
            all_results = dict(all_results[['metric', 'value']].values)
            all_results['total_number_of_validation_samples'] = len(table)
            self.log_metrics(all_results, 'val')

    def save_sampling_trajectory(self, outdir):
        # Visualize a sampling trajectory
        if (self.current_epoch + 1) % self.visualize_chain_epoch == 0 and self.trainer.is_global_zero:
            tic = time()
            outdir.mkdir(exist_ok=True, parents=True)

            batch = self.val_dataset.collate_fn([self.val_dataset[torch.randint(len(self.val_dataset), size=(1,))]])
            
            batch['ligand'] = Ligand(**batch['ligand']).to(self.device)
            batch['pocket'] = Residues(**batch['pocket']).to(self.device)

            num_nodes = "ground_truth" if self.sample_with_ground_truth_size else None

            if self.masking['design'] > 0:
                ligand_chain, pocket_chain, info = self.sample_chain(batch, self.keep_frames, num_nodes=num_nodes)

                # save molecules and pocket
                utils.write_sdf_file(Path(outdir, 'chain_ligand_design.sdf'), ligand_chain)
                mols_to_pdbfile(pocket_chain, Path(outdir, 'chain_pocket_design.pdb'))

            if self.masking['docking'] > 0:
                assert len(batch['pocket']['x']) > 0
                ligand_chain, pocket_chain, info = self.sample_chain(batch, self.keep_frames, num_nodes=num_nodes, docking=True)
                utils.write_sdf_file(Path(outdir, 'chain_ligand_docking.sdf'), ligand_chain)
                mols_to_pdbfile(pocket_chain, Path(outdir, 'chain_pocket_docking.pdb'))

            self.log_metrics(info, 'val')
            print(f'Chain visualization took {time() - tic:.2f} seconds')

    @staticmethod
    def change_ligand_size_and_keep_masked_values(batch: Ligand, num_nodes: torch.Tensor) -> Ligand:
        device = batch['x'].device
        new_data_list = []
        transform = AppendVirtualNodesInCoM(atom_encoder, bond_encoder)
        for data, total_size in zip(split_entity(batch), num_nodes):
            if data['known_h'].sum() > 0:
                data = extract_substructure(data, atoms_to_keep=data['known_h'])

            num_to_add = total_size - data['size']
            if num_to_add > 0:
                transform.add_min = transform.add_max = num_to_add
                data = transform(data)
                del data['virtual_mask']
            elif num_to_add < 0:
                data = extract_substructure(data, atoms_to_keep=torch.arange(data["size"], device=data["x"].device) < total_size)

            new_data_list.append(data)

        out = collate_entity(new_data_list)
        return Ligand(**out).to(device)

    def init_ligand(self, num_nodes_lig, pocket):
        device = pocket['x'].device

        n_samples = len(pocket['size'])
        lig_mask = utils.num_nodes_to_batch_mask(n_samples, num_nodes_lig, device)

        # only consider upper triangular matrix for symmetry
        lig_bonds = torch.stack(torch.where(torch.triu(lig_mask[:, None] == lig_mask[None, :], diagonal=1)), dim=0)
        lig_edge_mask = lig_mask[lig_bonds[0]]

        # Sample from Normal distribution in the pocket center
        pocket_com = pocket.get_com()
        z0_x = self.module_x.sample_z0(pocket_com, lig_mask)
        z0_h = self.module_h.sample_z0(lig_mask)
        z0_e = self.module_e.sample_z0(lig_edge_mask)

        return Ligand(**{
            'x': z0_x, 'h': z0_h, 'e': z0_e, 'mask': lig_mask,
            'bonds': lig_bonds, 'bond_mask': lig_edge_mask
        })

    def init_pocket(self, pocket):
        # The pocket is kept fixed during generation.
        return pocket

    def parse_num_nodes_spec(
            self, batch: Dict[str, Ligand],
            spec: Union[int, str, torch.Tensor] = None,
            min_size: torch.Tensor = None,
        ) -> torch.Tensor:

        if spec == "histogram" or spec is None:  # default option
            if not 'pocket' in batch or (batch['pocket']['size'] == 0).any():
                num_nodes, _ = self.size_distribution.sample(n_samples=len(batch['ligand']['size']))
                num_nodes = num_nodes.to(batch['ligand']['x'].device)
            else:
                # condition on the pocket size
                num_nodes = self.size_distribution.sample_conditional(n1=None, n2=batch['pocket']['size'])

            # make sure there is at least one potential bond
            num_nodes[num_nodes < 2] = 2

        elif isinstance(spec, int):
            num_nodes = torch.ones_like(batch['ligand']['size']) * spec

        elif isinstance(spec, torch.Tensor):
            num_nodes = spec.clone()

        elif spec == "ground_truth":
            assert "ligand" in batch
            num_nodes = batch['ligand']['size'].clone()

        elif isinstance(spec, str) and spec.startswith("uniform"):
            # expected format: uniform_low_high
            assert "pocket" in batch
            left, right = map(int, spec.split("_")[1:])
            shape = batch['pocket']['size'].shape
            num_nodes = torch.randint(left, right + 1, shape, dtype=torch.long, device=batch['ligand']['x'].device)

        else:
            raise NotImplementedError(f"Invalid size specification {spec}")

        if self.virtual_nodes:
            num_nodes += self.add_virtual_max

        if min_size is not None:
            num_nodes = torch.max(num_nodes, min_size)

        return num_nodes

    @torch.no_grad()
    def sample(
        self, data, n_samples, num_nodes=None, timesteps=None, step_spacing='linear', 
        return_pocket_names=False,
        **kwargs
    ):

        data['pocket'] = Residues(**data['pocket'])
        data['ligand'] = Ligand(**data['ligand'])

        timesteps = self.T_sampling if timesteps is None else timesteps

        input_pocket = data_utils.repeat_items(data['pocket'], n_samples)
        input_ligand = data_utils.repeat_items(data['ligand'], n_samples)

        batch = {"ligand": input_ligand, "pocket": input_pocket}
        fragment_sizes = scatter_add(input_ligand['known_h'].float(), index=input_ligand['mask'], dim_size=len(input_pocket['size'])).long()
        num_nodes = self.parse_num_nodes_spec(batch, spec=num_nodes, min_size=fragment_sizes)
        input_ligand = self.change_ligand_size_and_keep_masked_values(input_ligand, num_nodes)

        # Sample from prior
        ligand = self.init_ligand(num_nodes, input_pocket)
        pocket = self.init_pocket(input_pocket)

        # Masking
        ligand.register_known_variables(
            true_x=input_ligand['x'], known_x=input_ligand['known_x'],
            true_h=input_ligand['one_hot'], known_h=input_ligand['known_h'],
            true_e=input_ligand['bond_one_hot'], known_e=input_ligand['known_e'],
        )
        ligand.insert_known_variables()

        # return prior samples
        if timesteps == 0:
            # Convert into rdmols
            rdmols = [build_molecule(coords=m['x'], 
                atom_types=m['h'].argmax(1), 
                bonds=m['bonds'], 
                bond_types=m['e'].argmax(1), 
                atom_decoder=self.atom_decoder, bond_decoder=self.bond_decoder) 
                for m in data_utils.split_entity(ligand.detach().cpu())]

            rdpockets = pocket_to_rdkit(pocket, self.pocket_representation,
                                        self.atom_encoder, self.atom_decoder,
                                        self.aa_decoder, self.aa_atom_index)

            return rdmols, rdpockets, input_ligand['name']

        out_tensors_ligand, out_tensors_pocket = self.sampler(
            ligand, pocket, timesteps, 0.0, 1.0,
            known_x=input_ligand['known_x'],
            known_h=input_ligand['known_h'],
            known_e=input_ligand['known_e'],
            step_spacing=step_spacing,
        )

        # Build mol objects
        x = out_tensors_ligand['x'].detach().cpu()
        ligand_type = out_tensors_ligand['h'].argmax(1).detach().cpu()
        edge_type = out_tensors_ligand['e'].argmax(1).detach().cpu()
        lig_mask = ligand['mask'].detach().cpu()
        lig_bonds = ligand['bonds'].detach().cpu()
        lig_edge_mask = ligand['bond_mask'].detach().cpu()
        sizes = torch.unique(ligand['mask'], return_counts=True)[1].tolist()
        offsets = list(accumulate(sizes[:-1], initial=0))
        mol_kwargs = {
            'coords': utils.batch_to_list(x, lig_mask),
            'atom_types': utils.batch_to_list(ligand_type, lig_mask),
            'bonds': utils.batch_to_list_for_indices(lig_bonds, lig_edge_mask, offsets),
            'bond_types': utils.batch_to_list(edge_type, lig_edge_mask)
        }
        if self.predict_confidence:
            sigma_x = out_tensors_ligand['sigma_x'].detach().cpu()
            entropy_h = out_tensors_ligand['entropy_h'].detach().cpu()
            mol_kwargs['atom_props'] = [
                {'sigma_x': x[0], 'entropy_h': x[1]}
                for x in zip(utils.batch_to_list(sigma_x, lig_mask),
                             utils.batch_to_list(entropy_h, lig_mask))
            ]
        mol_kwargs = [{k: v[i] for k, v in mol_kwargs.items()}
                      for i in range(len(mol_kwargs['coords']))]

        # Convert into rdmols
        rdmols = [build_molecule(
            **m, atom_decoder=self.atom_decoder, bond_decoder=self.bond_decoder)
            for m in mol_kwargs
        ]

        out_pocket = pocket.deepcopy()
        out_pocket['x'] = out_tensors_pocket['x']
        out_pocket['v'] = out_tensors_pocket['v']
        rdpockets = pocket_to_rdkit(out_pocket, self.pocket_representation,
                                    self.atom_encoder, self.atom_decoder,
                                    self.aa_decoder, self.aa_atom_index)

        if return_pocket_names:
            return rdmols, rdpockets, input_ligand['name'], input_pocket['name']
        else:
            return rdmols, rdpockets, input_ligand['name']

    @torch.no_grad()
    def sample_chain(self, data, keep_frames, num_nodes=None, timesteps=None, guide_log_prob=None, docking=False, project_final=False, **kwargs):
        input_pocket = Residues(**data['pocket'])
        input_ligand = Ligand(**data['ligand'])
        info = {}

        timesteps = self.T_sampling if timesteps is None else timesteps
        keep_frames = min(keep_frames, timesteps)

        assert len(input_pocket['mask'].unique()) <= 1, "sample_chain only supports a single sample"

        if not docking:
            fragment_sizes = scatter_add(input_ligand['known_h'].float(), index=input_ligand['mask']).long()
            num_nodes = self.parse_num_nodes_spec(batch={"pocket": input_pocket, "ligand": input_ligand}, spec=num_nodes, min_size=fragment_sizes)
            input_ligand = self.change_ligand_size_and_keep_masked_values(input_ligand, num_nodes)
            input_ligand['known_h'] = input_ligand['known_x']
            input_ligand['known_e'] = edge_mask_by_node_mask(input_ligand['known_h'], input_ligand['bonds'])
            # Sample from prior
            ligand = self.init_ligand(num_nodes, input_pocket)
            pocket = self.init_pocket(input_pocket)
         
        else:
            input_ligand['known_h'] = torch.ones_like(input_ligand['known_h'])
            input_ligand['known_e'] = edge_mask_by_node_mask(input_ligand['known_h'], input_ligand['bonds'])
            ligand = self.init_ligand(input_ligand['size'], input_pocket)
            pocket = self.init_pocket(input_pocket)

        # Masking
        ligand.register_known_variables(
            true_x=input_ligand['x'], known_x=input_ligand['known_x'],
            true_h=input_ligand['one_hot'], known_h=input_ligand['known_h'],
            true_e=input_ligand['bond_one_hot'], known_e=input_ligand['known_e'],
        )
        ligand.insert_known_variables()

        out_tensors_ligand, out_tensors_pocket = self.sampler(
            ligand, pocket, timesteps, 0.0, 1.0, return_frames=keep_frames,
            project_final=project_final, known_x=input_ligand['known_x'],
            known_h=input_ligand['known_h'], known_e=input_ligand['known_e'],
        )

        info['traj_displacement_lig'] = torch.norm(out_tensors_ligand['x'][-1] - out_tensors_ligand['x'][0], dim=-1).mean()
        info['traj_rms_lig'] = out_tensors_ligand['x'].std(dim=0).mean()

        # Flatten
        assert keep_frames == out_tensors_ligand['x'].size(0) == out_tensors_pocket['x'].size(0)
        n_atoms = out_tensors_ligand['x'].size(1)
        n_bonds = out_tensors_ligand['e'].size(1)
        n_residues = out_tensors_pocket['x'].size(1)
        device = out_tensors_ligand['x'].device

        def flatten_tensor(chain):
            if len(chain.size()) == 3:  # l=0 values
                return chain.view(-1, chain.size(-1))
            elif len(chain.size()) == 4:  # vectors
                return chain.view(-1, chain.size(-2), chain.size(-1))
            else:
                warnings.warn(f"Could not flatten frame dimension of tensor with shape {list(chain.size())}")
                return chain

        out_tensors_ligand_flat = {k: flatten_tensor(chain) for k, chain in out_tensors_ligand.items()}
        ligand_mask_flat = torch.arange(keep_frames).repeat_interleave(n_atoms).to(device)

        bond_mask_flat = torch.arange(keep_frames).repeat_interleave(n_bonds).to(device)
        edges_flat = ligand['bonds'].repeat(1, keep_frames)

        # Build ligands
        x = out_tensors_ligand_flat['x'].detach().cpu()
        ligand_type = out_tensors_ligand_flat['h'].argmax(1).detach().cpu()
        ligand_mask_flat = ligand_mask_flat.detach().cpu()
        bond_mask_flat = bond_mask_flat.detach().cpu()
        edges_flat = edges_flat.detach().cpu()
        edge_type = out_tensors_ligand_flat['e'].argmax(1).detach().cpu()
        offsets = torch.zeros(keep_frames, dtype=int)  # edges_flat is already zero-based
        molecules = list(
            zip(utils.batch_to_list(x, ligand_mask_flat),
                utils.batch_to_list(ligand_type, ligand_mask_flat),
                utils.batch_to_list_for_indices(edges_flat, bond_mask_flat, offsets),
                utils.batch_to_list(edge_type, bond_mask_flat)
                )
        )

        # Convert into rdmols
        ligand_chain = [build_molecule(
            *graph, atom_decoder=self.atom_decoder,
            bond_decoder=self.bond_decoder) for graph in molecules
        ]

        # Build pockets
        # the pocket does not change during sampling, so we write it once
        out_pocket = pocket
        pocket_chain = pocket_to_rdkit(out_pocket, self.pocket_representation,
                                       self.atom_encoder, self.atom_decoder,
                                       self.aa_decoder, self.aa_atom_index)

        return ligand_chain, pocket_chain, info

    # def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
    # def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
    def configure_gradient_clipping(self, optimizer, *args, **kwargs):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # hard upper limit
        max_grad_norm = min(max_grad_norm, 10.0)

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')
            grad_norm = max_grad_norm

        self.gradnorm_queue.add(float(grad_norm))
    
