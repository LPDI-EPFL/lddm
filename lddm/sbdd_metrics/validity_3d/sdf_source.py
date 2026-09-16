from rdkit import Chem

from .data_source import DataSource


class SDFSource(DataSource):
    
    def __init__(self,
                 ligands_path: str,
                 name: str,
                 limit: int = None,
                 ) -> None:
        super().__init__(name)
        self.ligands_path = ligands_path
        self.limit = limit
    
    def __iter__(self):
        for i, mol in enumerate(Chem.SDMolSupplier(self.ligands_path)):
            if self.limit is not None and i >= self.limit:
                break
            yield mol