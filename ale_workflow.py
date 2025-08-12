"""Workflow for running coordinates-based meta-analysis."""
import argparse
import os
import os.path as op
import shutil
from glob import glob

import nibabel as nib
import numpy as np
from nilearn.image import math_img
from nimare.dataset import Dataset
from nimare.diagnostics import FocusCounter, Jackknife
from nimare.io import convert_sleuth_to_dataset
from nimare.meta.cbma import ALESubtraction
from nimare.reports.base import run_reports
from nimare.results import MetaResult
from nimare.transforms import p_to_z
from nimare.workflows.cbma import CBMAWorkflow, PairwiseCBMAWorkflow


def _get_parser():
    parser = argparse.ArgumentParser(description="Run coordinates-based meta-analysis workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=4,
        required=False,
        help="CPUs",
    )
    return parser


ALE_PVAL = 0.01  # FWE-corrected threshold for diagnostics, one-tailed
SUB_PVAL = 0.05  # FDR-corrected threshold for diagnostics, two-tailed


def main(project_dir, n_cores):
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    conjunction_fn = op.join(results_dir, "conjunction_map.nii.gz")
    if not op.isfile(conjunction_fn):
        mult_formula = ""
        min_formula = ""
        imgs_dict = {}

    # Standard analysis
    annots = ["SDHC", "HCSD"]
    for annot_i, annot in enumerate(annots):
        output_dir = op.join(results_dir, annot, "ale")
        os.makedirs(output_dir, exist_ok=True)

        # Create Dataset
        dset_fn = op.join(data_dir, f"{annot}_dataset.pkl.gz")
        print(f"Processing annnotation '{annot}'...", flush=True)
        if not op.isfile(dset_fn):
            print("\tLoading sleuth files and creating Dataset object...", flush=True)
            sleuth_files = glob(op.join(data_dir, f"SStruct_{annot}_*.txt"))
            dset = convert_sleuth_to_dataset(sleuth_files)
            dset.save(dset_fn)
        else:
            print("\tLoading Dataset object...", flush=True)
            dset = Dataset.load(dset_fn)

        # Run Workflow
        results_fn = op.join(results_dir, annot, f"{annot}-ale_result.pkl.gz")
        if not op.isfile(results_fn):
            os.makedirs(op.dirname(results_fn), exist_ok=True)

            print("\tRunning CBMA workflow...", flush=True)
            voxel_thresh = round(p_to_z(ALE_PVAL, tail="one"), 2)
            ale_workflow = CBMAWorkflow(
                estimator="ale",
                corrector="montecarlo",
                diagnostics=["jackknife", "focuscounter"],
                voxel_thresh=voxel_thresh,
                output_dir=output_dir,
                n_cores=n_cores,
            )
            results = ale_workflow.fit(dset)
            results.save(results_fn)

            # Organize maps and tables in folders
            maps_dir = op.join(output_dir, "maps")
            tables_dir = op.join(output_dir, "tables")
            os.makedirs(maps_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            maps_files = glob(op.join(output_dir, "*.nii.gz"))
            tables_files = glob(op.join(output_dir, "*.tsv"))
            [shutil.move(file_, maps_dir) for file_ in maps_files]
            [shutil.move(file_, tables_dir) for file_ in tables_files]
        else:
            print("\tLoading results...", flush=True)
            results = MetaResult.load(results_fn)

        # Generate Report
        if not op.isfile(op.join(output_dir, "report.html")):
            print("\tGenerating report for ALE analysis...", flush=True)
            run_reports(results, output_dir)

        if not op.isfile(conjunction_fn):
            # Generate variable for conjunction analysis
            img = results.get_map("z_desc-mass_level-cluster_corr-FWE_method-montecarlo")
            img_lb = f"img{annot_i}"
            imgs_dict[img_lb] = img
            mult_formula += img_lb
            min_formula += img_lb

            if annot != annots[-1]:
                mult_formula += " * "
                min_formula += ", "

    # Conjunction analysis
    if not op.isfile(conjunction_fn):
        print("Running conjunction analysis...", flush=True)
        formula = f"np.where({mult_formula} > 0, np.minimum.reduce([{min_formula}]), 0)"
        print(f"\tFormula: {formula}", flush=True)

        img_conj = math_img(formula, **imgs_dict)
        nib.save(img_conj, conjunction_fn)

    # Subtraction analysis
    print("Contrast analysis...", flush=True)
    for annot in annots:
        output_dir = op.join(results_dir, annot, "subtraction")
        os.makedirs(output_dir, exist_ok=True)

        sub_result_fn = op.join(results_dir, annot, f"{annot}-subtraction_result.pkl.gz")
        if not op.isfile(sub_result_fn):
            print(f"\tRunning contrast analysis on {annot} vs others...", flush=True)
            dset_fn = op.join(data_dir, f"{annot}_dataset.pkl.gz")
            dset = Dataset.load(dset_fn)

            # Create Dataset 1 and 2
            remaining_dset = None
            for rem_annot in annots:
                if rem_annot == annot:
                    continue

                temp_dset_fn = op.join(data_dir, f"{rem_annot}_dataset.pkl.gz")
                temp_dset = Dataset.load(temp_dset_fn)
                if remaining_dset is None:
                    remaining_dset = temp_dset.copy()
                else:
                    shared_ids = np.intersect1d(remaining_dset.ids, temp_dset.ids)
                    if shared_ids.size:
                        # Remove duplicates from remaining_dset
                        print("\t\tDuplicate IDs detected in both datasets.", flush=True)
                        unique_ids = np.setdiff1d(temp_dset.ids, remaining_dset.ids)
                        temp_dset = temp_dset.slice(unique_ids)

                    remaining_dset.merge(temp_dset)

            # Run Workflow
            voxel_thresh = round(p_to_z(SUB_PVAL, tail="two"), 2)
            sub_workflow = PairwiseCBMAWorkflow(
                ALESubtraction(n_iters=10000, n_cores=n_cores),
                corrector="fdr",
                diagnostics=[
                    FocusCounter(
                        voxel_thresh=voxel_thresh,
                        display_second_group=True,
                        n_cores=n_cores,
                    ),
                    Jackknife(
                        voxel_thresh=voxel_thresh,
                        display_second_group=True,
                        n_cores=n_cores,
                    ),
                ],
                output_dir=output_dir,
                n_cores=n_cores,
            )
            sub_result = sub_workflow.fit(dset, remaining_dset)
            sub_result.save(sub_result_fn)

            # Organize maps and tables in folders
            maps_dir = op.join(output_dir, "maps")
            tables_dir = op.join(output_dir, "tables")
            os.makedirs(maps_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            maps_files = glob(op.join(output_dir, "*.nii.gz"))
            tables_files = glob(op.join(output_dir, "*.tsv"))
            [shutil.move(file_, maps_dir) for file_ in maps_files]
            [shutil.move(file_, tables_dir) for file_ in tables_files]
        else:
            print("\tLoading results...", flush=True)
            sub_result = MetaResult.load(sub_result_fn)

        # Generate Report
        if not op.isfile(op.join(output_dir, "report.html")):
            print("\tGenerating report for substraction analysis...", flush=True)
            run_reports(sub_result, output_dir)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
