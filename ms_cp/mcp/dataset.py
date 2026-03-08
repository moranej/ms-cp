from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Set
import numpy as np
from torch.utils.data import DataLoader, Subset
from data import RetrievalDataset_PrecompFPandInchi
#from massspecgym.data.data_module import MassSpecDataModule
from ms_cp.retrieval.data_module import MassSpecDataModule
from massspecgym.data.transforms import MolFingerprinter, SpecBinner


@dataclass
class LoadedData:
    dataset: RetrievalDataset_PrecompFPandInchi
    calib_loader: DataLoader
    test_loader: DataLoader
    calib_idx: List[int]
    test_idx: List[int]
    train_inchikeys: Set[str]


def load_retrieval_data(
    tsv_path: str,
    helper_dir: str,
    batch_size: int,
    num_workers: int,
    max_mz: float = 1005.0,
    bin_width: float = 0.1,
    fp_size: int = 4096,
) -> LoadedData:
    dataset = RetrievalDataset_PrecompFPandInchi(
        spec_transform=SpecBinner(max_mz=max_mz, bin_width=bin_width, to_rel_intensities=True),
        mol_transform=MolFingerprinter(fp_size=fp_size),
        pth=tsv_path,
        fp_pth=os.path.join(helper_dir, f"fp_{fp_size}.npy"),
        inchi_pth=os.path.join(helper_dir, "inchis.npy"),
        candidates_pth=os.path.join(helper_dir, "MassSpecGym_retrieval_candidates_formula.json"),
        candidates_fp_pth=os.path.join(helper_dir, "MassSpecGym_retrieval_candidates_formula_fps.npz"),
        candidates_inchi_pth=os.path.join(helper_dir, "MassSpecGym_retrieval_candidates_formula_inchi.npz"),
    )

    dm = MassSpecDataModule(dataset, batch_size=batch_size, num_workers=num_workers)
    dm.setup(stage="test")
    split = dm.split

    calib_ids = split[split == "calib"].index.tolist()
    test_ids = split[split == "test"].index.tolist()
    train_ids = split[split == "train"].index.tolist()

    md = dataset.metadata
    id_to_idx = {rid: i for i, rid in md["identifier"].items()}
    calib_idx = [id_to_idx[i] for i in calib_ids if i in id_to_idx]
    test_idx = [id_to_idx[i] for i in test_ids if i in id_to_idx]
    train_idx = [id_to_idx[i] for i in train_ids if i in id_to_idx]

    def _valid(i: int) -> bool:
        smiles = md.iloc[i]["smiles"]
        if smiles not in dataset.candidate_fps:
            return False
        cand_fp = dataset.candidate_fps[smiles]
        return not np.allclose(cand_fp, 0)

    calib_idx = [i for i in calib_idx if _valid(i)]
    test_idx = [i for i in test_idx if _valid(i)]

    calib_loader = DataLoader(
        Subset(dataset, calib_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    train_inchikeys = set(md.iloc[train_idx]["inchikey"].astype(str).tolist())
    return LoadedData(
        dataset=dataset,
        calib_loader=calib_loader,
        test_loader=test_loader,
        calib_idx=calib_idx,
        test_idx=test_idx,
        train_inchikeys=train_inchikeys,
    )
