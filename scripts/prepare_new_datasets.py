#!/usr/bin/env python3
"""
Convert Mulan ARFF datasets (bibtex, Corel5k) to the paper_protocol npz format,
then download and convert Eurlex-sm from MEKA.

Output: data/paper_protocol/<dataset>/{train.npz, test.npz}

The npz format stores sparse CSR components:
  X_data, X_indices, X_indptr, X_shape
  Y_data, Y_indices, Y_indptr, Y_shape
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy import sparse
from scipy.io import arff as scipy_arff

MULAN_ROOT = Path("data/sources/mulan/extracted")
OUT_ROOT = Path("data/paper_protocol")


def parse_label_names_xml(xml_path: Path) -> list[str]:
    """Parse Mulan XML to get label attribute names."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"m": "http://mulan.sourceforge.net/labels"}
    labels = root.findall(".//m:label", ns)
    return [l.attrib["name"] for l in labels]


def load_arff_mulan(arff_path: Path, label_names: list[str]) -> tuple:
    """Load a Mulan ARFF, split into X (features) and Y (labels).
    Returns sparse CSR matrices.
    """
    import arff as liac_arff
    
    with open(arff_path) as f:
        dataset = liac_arff.load(f)
    
    attr_names = [a[0] for a in dataset["attributes"]]
    data = dataset["data"]
    
    n = len(data)
    d_total = len(attr_names)
    
    label_set = set(label_names)
    label_indices = [i for i, name in enumerate(attr_names) if name in label_set]
    feature_indices = [i for i, name in enumerate(attr_names) if name not in label_set]
    
    print(f"    {arff_path.name}: {n} instances, {len(feature_indices)} features, {len(label_indices)} labels")
    
    # Build dense arrays (ARFF data is usually small enough)
    X_dense = np.zeros((n, len(feature_indices)), dtype=np.float64)
    Y_dense = np.zeros((n, len(label_indices)), dtype=np.int8)
    
    for i, row in enumerate(data):
        for j, fi in enumerate(feature_indices):
            val = row[fi]
            if val is not None:
                X_dense[i, j] = float(val)
        for j, li in enumerate(label_indices):
            val = row[li]
            if val is not None:
                Y_dense[i, j] = int(float(val))
    
    return sparse.csr_matrix(X_dense), sparse.csr_matrix(Y_dense)


def save_npz(path: Path, X: sparse.csr_matrix, Y: sparse.csr_matrix):
    """Save sparse X, Y in the project's npz format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X_data=X.data,
        X_indices=X.indices,
        X_indptr=X.indptr,
        X_shape=np.array(X.shape),
        Y_data=Y.data,
        Y_indices=Y.indices,
        Y_indptr=Y.indptr,
        Y_shape=np.array(Y.shape),
    )
    print(f"    Saved {path} (X={X.shape}, Y={Y.shape})")


def process_mulan_dataset(name: str, arff_prefix: str, xml_name: str):
    """Process a Mulan dataset from ARFF to npz."""
    print(f"\n=== Processing {name} ===")
    
    ds_dir = MULAN_ROOT / name.lower()
    if not ds_dir.exists():
        # Try original case
        ds_dir = MULAN_ROOT / name
    if not ds_dir.exists():
        print(f"  ERROR: {ds_dir} not found!")
        return
    
    xml_path = ds_dir / f"{xml_name}.xml"
    label_names = parse_label_names_xml(xml_path)
    print(f"  Labels: {len(label_names)}")
    
    train_arff = ds_dir / f"{arff_prefix}-train.arff"
    test_arff = ds_dir / f"{arff_prefix}-test.arff"
    
    if not train_arff.exists():
        print(f"  ERROR: {train_arff} not found!")
        return
    
    X_train, Y_train = load_arff_mulan(train_arff, label_names)
    X_test, Y_test = load_arff_mulan(test_arff, label_names)
    
    out_dir = OUT_ROOT / name
    save_npz(out_dir / "train.npz", X_train, Y_train)
    save_npz(out_dir / "test.npz", X_test, Y_test)
    
    print(f"  ✓ {name}: train={X_train.shape[0]}, test={X_test.shape[0]}, "
          f"features={X_train.shape[1]}, labels={Y_train.shape[1]}")


def download_eurlex():
    """Download Eurlex-sm from MEKA repository."""
    print("\n=== Downloading Eurlex-sm ===")
    
    # Eurlex-sm is not available in scikit-multilearn
    # Skip this for now — we have bibtex, corel5k, and mediamill
    print("  Eurlex-sm not available in scikit-multilearn, skipping")


def download_via_skmultilearn(name: str, out_name: str | None = None):
    """Download a dataset via scikit-multilearn and save as npz."""
    out_name = out_name or name
    print(f"\n=== Downloading {name} via scikit-multilearn ===")
    
    out_dir = OUT_ROOT / out_name
    if (out_dir / "train.npz").exists() and (out_dir / "test.npz").exists():
        print("  Already exists, skipping")
        return
    
    from skmultilearn.dataset import load_dataset
    
    X_train, Y_train, _, _ = load_dataset(name, "train")
    X_test, Y_test, _, _ = load_dataset(name, "test")
    
    X_train = sparse.csr_matrix(X_train, dtype=np.float64)
    Y_train = sparse.csr_matrix(Y_train, dtype=np.int8)
    X_test = sparse.csr_matrix(X_test, dtype=np.float64)
    Y_test = sparse.csr_matrix(Y_test, dtype=np.int8)
    
    save_npz(out_dir / "train.npz", X_train, Y_train)
    save_npz(out_dir / "test.npz", X_test, Y_test)
    
    print(f"  ✓ {out_name}: train={X_train.shape[0]}, test={X_test.shape[0]}, "
          f"features={X_train.shape[1]}, labels={Y_train.shape[1]}")


def main():
    # 1. Bibtex (already extracted)
    process_mulan_dataset("bibtex", "bibtex", "bibtex")
    
    # 2. Corel5k (already extracted)
    process_mulan_dataset("corel5k", "Corel5k", "Corel5k")
    
    # 3. Eurlex-sm (not available)
    download_eurlex()
    
    # 4. tmc2007_500 (via scikit-multilearn)
    download_via_skmultilearn("tmc2007_500", "tmc2007")
    
    # 5. mediamill (via scikit-multilearn)
    download_via_skmultilearn("mediamill", "mediamill")
    
    print("\n" + "=" * 60)
    print("Done! Datasets are in data/paper_protocol/")
    print("Next step: run export_cv_splits_to_mat.py to create .mat folds")
    print("  python scripts/export_cv_splits_to_mat.py \\")
    print("    --data-root data/paper_protocol \\")
    print("    --datasets bibtex corel5k tmc2007 mediamill \\")
    print("    --output-dir data/paper_matlab_minmax \\")
    print("    --folds 5 --seed 42 --scaler minmax \\")
    print("    --split-mode repeated_holdout")


if __name__ == "__main__":
    main()
