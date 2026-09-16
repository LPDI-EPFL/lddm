from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar, Type
import yaml

from lddm.config.base import BaseConfig
from lddm.constants import atom_encoder, bond_encoder, aa_encoder, residue_bond_encoder, aa_atom_index


T = TypeVar("T")


@dataclass
class FeaturizationConfig(BaseConfig):
    # Ligand features
    compute_fragment_mask: bool = True
    kekulize: bool = False
    atom_encoder: dict[str, int] = field(default_factory=lambda: atom_encoder)
    bond_encoder: dict[str, int] = field(default_factory=lambda: bond_encoder)

    # Pocket features
    pocket_representation: Literal["CA+"] = "CA+"
    dist_cutoff: float | None = 8.0
    amino_acid_encoder: dict[str, int] = field(default_factory=lambda: aa_encoder)
    residue_bond_encoder: dict[str, int] = field(default_factory=lambda: residue_bond_encoder)
    aa_atom_index: dict[str, dict[str, int]] = field(default_factory=lambda: aa_atom_index)

    @property
    def max_num_atoms_per_residue(self):
        return max([x for aa in self.aa_atom_index.values() for x in aa.values()]) + 1

    
@dataclass
class DatasetConfig(BaseConfig):
    dataset: str
    name: str
    split: Literal["train", "val", "test"] | None
    limit: int | None = None

    featurization: FeaturizationConfig = field(default_factory=lambda: FeaturizationConfig())
    filters: dict[str, str | bool | float] = field(default_factory=lambda: {})
    
    @classmethod
    def from_yaml(cls: Type[T], file: Path) -> T:
        with open(file, "r") as f:
            config = DatasetConfig.from_dict(yaml.safe_load(f))
        return config

    def __post_init__(self):
        self.filters = self.filters or {}
