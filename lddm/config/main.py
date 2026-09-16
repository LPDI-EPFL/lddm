from dataclasses import dataclass, field, asdict, make_dataclass
from typing import Literal, Any
from pathlib import Path

from lddm.utils import namespace_to_dict, Namespace
from lddm.config.base import BaseConfig
from lddm.config.data import FeaturizationConfig


@dataclass
class TrainParams(BaseConfig):
    logdir: str
    enable_progress_bar: bool = False
    num_sanity_val_steps: int = 0
    batch_size: int | None = None
    max_tokens_per_batch: int | None = None
    accumulate_grad_batches: int = 1
    lr: float = 5.0e-4
    lr_step_size: float = None
    lr_gamma: float = None
    lr_warmup_steps: int | None = None
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-12
    n_epochs: int = 1000
    num_workers: int = 0
    num_nodes: int = 1
    gpus: int | Literal["auto"] = "auto"
    clip_grad: bool = True
    precision: int | str = 32
    gnina: str = None
    masking: dict[str, int] = field(default_factory=lambda: {'design': 1, 'docking': 0, 'context': 0})
    apply_random_rotations: bool = False

    def __post_init__(self):
        if isinstance(self.gpus, str):
           assert self.gpus == "auto"
           import torch
           self.gpus = torch.cuda.device_count()


@dataclass
class SingleDatasetParams(BaseConfig):
    datadir: str | Path = None
    clustered_dataset: bool = False
    sample_from_clusters: bool = False
    

@dataclass
class MixedDatasetParams(BaseConfig):
    num_samples_per_epoch_per_gpu: int
    datasets: list[SingleDatasetParams]
    weights: list[float] = None

    @classmethod
    def from_dict(cls, config_as_dict, **kwargs):
        config_as_dict['datasets'] = [
            SingleDatasetParams.from_dict(_params, **kwargs)
            for _params in config_as_dict['datasets']
        ]
        return super().from_dict(config_as_dict, **kwargs)


@dataclass
class DatasetSplitParams(SingleDatasetParams):
    mixed_dataset: MixedDatasetParams = None


@dataclass
class DatasetParams(BaseConfig):
    meta_info_root: str | Path
    train: DatasetSplitParams
    val: DatasetSplitParams
    test: DatasetSplitParams


@dataclass
class WandbParams(BaseConfig):
    entity: str = None
    group: str = None
    mode: Literal['disabled', 'offline', 'online'] = 'disabled'


@dataclass
class LossParams(BaseConfig):
    reduce: str = 'mean'
    discrete_loss: str = 'VLB'
    coord_loss_type: str = None  # None -> standard L2 loss
    lambda_x: float = 1.0
    lambda_x_error: float = None
    lambda_h: float = 1.0
    lambda_e: float = 1.0
    lambda_clash: float = None
    timestep_weights: str = None  # e.g. sigmoid_a=1_b=10
    optimal_transport: bool = False
    regularize_uncertainty: float = None
    timestep_sampler: str = 'uniform'


@dataclass
class SimulationParams(BaseConfig):
    n_steps: int = 5000
    prior_x: Literal['gaussian', 'harmonic'] = 'gaussian'
    prior_h: Literal['uniform', 'marginal'] = 'uniform'
    prior_e: Literal['uniform', 'marginal'] = 'uniform'
    sigma_x: float = None
    predict_final: bool = False
    predict_confidence: bool = False
    uncertainty_is_variance: bool = False
    predict_error: bool = False
    masked_modeling: bool = False
    sampler: str = 'ForwardEuler'
    noise_scale_x: float | None = None
    prior_scale_x: float = 1.0
    scheduler_x: dict = None
    scheduler_h: dict = None
    scheduler_e: dict = None
    size_histogram_file: str = None


@dataclass
class EvalParams(BaseConfig):
    eval_epochs: int = 1
    n_loss_per_sample: int = 1
    sample_epochs: int = 1
    checkpoint_every_n_train_steps: int = None
    n_eval_samples: int = 4
    n_sampling_steps: int = 5
    step_spacing: Literal['log', 'linear'] = 'linear'
    eval_batch_size: int = 8
    visualize_sample_epoch: int = 1
    n_visualize_samples: int = 5
    visualize_chain_epoch: int = 1
    keep_frames: int = None
    sample_with_ground_truth_size: bool = True
    exclude_evaluators: list[str] = field(default_factory=lambda: ['geometry', 'energy', 'ring_count', 'gnina', 'interactions', 'fingerprint_novelty', 'ff_relaxation'])
    reference_mols_validity3d: str = None
    masking: dict[str, int] = None
    val_check_interval: int | float = None
    outdir: str | Path = None

    def __post_init__(self):
        if self.keep_frames is None:
           self.keep_frames = self.n_sampling_steps


@dataclass
class PredictorParams(BaseConfig):
    heterogeneous_graph: bool = True
    dynamics_version: str = 'hetero_v1'
    backbone: str = 'gvp'
    backbone_params: dict[str, Any] = field(default_factory=lambda: {})
    num_rbf_time: int = 16
    edge_cutoff_ligand: float = None
    edge_cutoff_pocket: float = 10.0
    edge_cutoff_interaction: float = 10.0
    edge_knn_ligand: int = None
    edge_knn_pocket: int = None
    edge_knn_interaction: int = None
    cycle_counts: bool = True
    spectral_feat: bool = False
    reflection_equivariant: bool = False
    num_rbf: int = 16
    d_max: float = 15.0
    self_conditioning: bool = True
    transform_sc_pred: bool = False
    augment_ligand_sc: bool = False
    add_all_atom_diff: bool = False
    enable_masked_modeling: bool = False
    uncertainty_act: Literal['softplus', 'exp'] = 'softplus'
    add_node_features_to_edges: bool = False
    hide_uncertainty_sc: bool = True

    def __post_init__(self):
        self.backbone_params = make_dataclass(
            "BackboneParams", 
            ((k, type(v)) for k, v in self.backbone_params.items()), 
            bases=(BaseConfig,),
        )(**self.backbone_params)


@dataclass
class LDDMConfig(BaseConfig):
    train_params: TrainParams
    dataset_params: DatasetParams
    loss_params: LossParams 
    simulation_params: SimulationParams
    eval_params: EvalParams 
    predictor_params: PredictorParams
    featurization_config: FeaturizationConfig
    wandb_params: WandbParams = None
    run_name: str = None
    virtual_nodes: tuple[int, int] = (0, 10)
    debug: bool = False
    overfit: bool = False
    ignore_featurization_mismatch: bool = False  # mainly for backward compatibility when configs aren't present

    def __post_init__(self):
        if self.eval_params.masking is None:
           self.eval_params.masking = self.train_params.masking

        if self.eval_params.n_sampling_steps is None:
           self.eval_params.n_sampling_steps = self.simulation_params.n_steps
