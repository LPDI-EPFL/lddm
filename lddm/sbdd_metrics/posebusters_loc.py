import pandas as pd
from rdkit.Chem import Mol
from typing import Any, Generator, Iterable
from pathlib import Path

from posebusters import PoseBusters
from posebusters.tools.loading import safe_load_mol, safe_supply_mols

# adapted from https://github.com/maabuu/posebusters/tree/main/posebusters

LOCAL_MODULES = ['Ring flatness', 'Double bond flatness', 'Geometry', 'Distance to protein']

class PoseBustersLocal(PoseBusters):
    def bust(
        self,
        mol_pred: Iterable[Mol | Path | str] | Mol | Path | str,
        mol_true: Mol | Path | str | None = None,
        mol_cond: Mol | Path | str | None = None,
        full_report: bool = False,
    ) -> pd.DataFrame:
        """Run tests on one or more molecules.

        Args:
            mol_pred: Generated molecule(s), e.g. de-novo generated molecule or docked ligand, with one or more poses.
            mol_true: True molecule, e.g. crystal ligand, with one or more poses.
            mol_cond: Conditioning molecule, e.g. protein.
            full_report: Whether to include all columns in the output or only the boolean ones specified in the config.

        Notes:
            - Molecules can be provided as rdkit molecule objects or file paths.

        Returns:
            Pandas dataframe with results.
        """
        mol_pred_list: Iterable[Mol | Path | str] = [mol_pred] if isinstance(mol_pred, (Mol, Path, str)) else mol_pred

        columns = ["mol_pred", "mol_true", "mol_cond"]
        self.file_paths = pd.DataFrame([[mol_pred, mol_true, mol_cond] for mol_pred in mol_pred_list], columns=columns)

        results_gen = self._run()

        df = pd.concat([_dataframe_from_output(d, self.config, full_report=full_report) for d in results_gen])
        df.index.names = ["file", "molecule"]
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        return df

    def _run(self) -> Generator[dict, None, None]:
        """Run all tests on molecules provided in file paths.

        Yields:
            Generator of result dictionaries.
        """
        self._initialize_modules()

        for _, paths in self.file_paths.iterrows():
            mol_args = {}
            if "mol_cond" in paths and paths["mol_cond"] is not None:
                mol_cond_load_params = self.config.get("loading", {}).get("mol_cond", {})
                mol_args["mol_cond"] = safe_load_mol(path=paths["mol_cond"], **mol_cond_load_params)
            if "mol_true" in paths and paths["mol_true"] is not None:
                mol_true_load_params = self.config.get("loading", {}).get("mol_true", {})
                mol_args["mol_true"] = safe_load_mol(path=paths["mol_true"], **mol_true_load_params)

            mol_pred_load_params = self.config.get("loading", {}).get("mol_pred", {})
            for i, mol_pred in enumerate(safe_supply_mols(paths["mol_pred"], **mol_pred_load_params)):
                if self.config["top_n"] is not None and i >= self.config["top_n"]:
                    break

                mol_args["mol_pred"] = mol_pred

                results_key = (str(paths["mol_pred"]), self._get_name(mol_pred, i))

                for name, fname, func, args in zip(self.module_name, self.fname, self.module_func, self.module_args):
                    if name not in LOCAL_MODULES:
                        continue
                    
                    # pick needed arguments for module
                    args_needed = {k: v for k, v in mol_args.items() if k in args}
                    # loading takes all inputs
                    if fname == "loading":
                        args_needed = {k: args_needed.get(k, None) for k in args_needed}
                    # run module when all needed input molecules are valid Mol objects
                    if fname != "loading" and not all(args_needed.get(m, None) for m in args_needed):
                        module_output: dict[str, Any] = {"results": {}}
                    else:
                        module_output = func(**args_needed)

                    # save to object
                    self.results[results_key].extend([(name, k, v) for k, v in module_output["results"].items()])
                    if "details" in module_output:
                        module_config = next(m for m in self.config['modules'] if m['name'] == name)
                        thresholds = {k: v for k, v in module_config.get('parameters', {}).items()
                                      if k in ('threshold_bad_bond_length', 'threshold_bad_angle', 'threshold_clash')}
                        results_per_atom = _process_details(module_output["details"], name, **thresholds)
                        self.results[results_key].extend([(name, k, v) for k, v in results_per_atom.items()])

                # return results for this entry
                yield {results_key: self.results[results_key]}

def _dataframe_from_output(results_dict, config, full_report: bool = False) -> pd.DataFrame:
    d = {id: {(module, output): value for module, output, value in results} for id, results in results_dict.items()}
    df = pd.DataFrame.from_dict(d, orient="index")

    test_columns = [(c["name"], n) for c in config["modules"] for n in c.get("chosen_binary_test_output", [])]
    names_lookup = {(c["name"], k): v for c in config["modules"] for k, v in c.get("rename_outputs", {}).items()}
    suffix_lookup = {c["name"]: c["rename_suffix"] for c in config["modules"] if "rename_suffix" in c}

    available_columns = df.columns.tolist()
    missing_columns = [c for c in test_columns if c not in available_columns]
    extra_columns = [c for c in available_columns if c not in test_columns]
    columns = test_columns + extra_columns if full_report else test_columns

    df[missing_columns] = pd.NA
    df = df[columns]
    df.columns = [names_lookup.get(c, c[-1] + suffix_lookup.get(c[0], "")) for c in df.columns]

    return df

def _process_details(details: dict, module: str,
                     threshold_bad_bond_length: float = 0.2,
                     threshold_bad_angle: float = 0.2,
                     threshold_clash: float = 0.2,
                     ) -> dict:
    results_per_atom = {}
    if module == 'Ring flatness':
        results_per_atom['atm_passed_planar_rings'] = []
        for i, ring in enumerate(details['planar_group']):
            if details['flatness_passes'][i] == False:
                results_per_atom['atm_passed_planar_rings'].extend(ring)

    elif module == 'Double bond flatness':
        results_per_atom['atm_passed_planar_bonds'] = []
        for i, ring in enumerate(details['planar_group']):
            if details['flatness_passes'][i] == False:
                results_per_atom['atm_passed_planar_bonds'].extend(ring)

    elif module == 'Geometry':
        df_bonds = details['bonds']
        df_angles = details['angles']
        df_clash_internal = details['clash']

        wrong_bonds = df_bonds.loc[(df_bonds["percent_error"] < -threshold_bad_bond_length) | \
                                   (df_bonds["percent_error"] > threshold_bad_bond_length)]['atom_pair']
        wrong_angles = df_angles.loc[df_angles["bound_absolute_percent_error"] > threshold_bad_angle]['atom_pair']
        wrong_clashes = df_clash_internal.loc[df_clash_internal["bound_percent_error"] < -threshold_clash]['atom_pair']

        flagged = set()
        all_pairs = list(wrong_angles) + list(wrong_bonds) + list(wrong_clashes)
        for atm_pair in all_pairs:
            for atm in atm_pair:
                flagged.add(atm)
        results_per_atom['atm_passed_geometry'] = list(flagged)

    elif module=='Distance to protein':
        df_clashes = details.loc[details['clash'] == True]['ligand_atom_id']
        results_per_atom['atm_passed_clashes'] = list(df_clashes)
    
    else:
        print(f"Warning: Module {module} not supported for per atom analysis")

    return results_per_atom
