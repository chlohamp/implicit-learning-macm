"""Workflow for running ROI-based meta-analytic connectivity modeling (MACM)."""

import argparse
import os
import os.path as op
import shutil
from glob import glob

import nibabel as nib
import numpy as np
from nilearn.image import threshold_img
from nimare.dataset import Dataset
from nimare.reports.base import run_reports
from nimare.results import MetaResult
from nimare.transforms import p_to_z
from nimare.workflows.cbma import CBMAWorkflow
from nimare.meta import ALE


def _get_parser():
    parser = argparse.ArgumentParser(description="Run ROI-based meta-analysis workflow")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--roi_dir",
        dest="roi_dir",
        required=True,
        help="Path to directory containing ROI NIfTI files (.nii or .nii.gz)",
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


def _is_binary(img: nib.Nifti1Image) -> bool:
    data = np.asarray(img.dataobj)
    uniq = np.unique(data[np.isfinite(data)])
    return np.all(np.isin(uniq, [0, 1])) and uniq.size <= 2


def _binarize(img: nib.Nifti1Image, z_thresh=2.33, clust_thresh=10) -> nib.Nifti1Image:
    """
    Make a binary mask. If already binary (0/1), return as-is.
    Otherwise, apply |z| > z_thresh with cluster extent threshold, then binarize >0.
    """
    if _is_binary(img):
        return img

    thr = threshold_img(img, threshold=z_thresh, cluster_threshold=clust_thresh)
    data = thr.get_fdata()
    data = (data > 0).astype(np.int16)
    return nib.Nifti1Image(data, thr.affine, thr.header)


def main(project_dir, roi_dir, n_cores):
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)
    roi_dir = op.abspath(roi_dir)

    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Load (or fetch+convert) Neurosynth dataset
    neurosynth_dset_path = op.join(data_dir, "neurosynth_terms_dataset.pkl.gz")
    if not op.isfile(neurosynth_dset_path):
        from nimare.extract import fetch_neurosynth
        from nimare.io import convert_neurosynth_to_dataset

        files = fetch_neurosynth(
            data_dir=data_dir,
            version="7",
            overwrite=False,
            source="abstract",
            vocab="terms",
        )
        neurosynth_db = files[0]
        dset = convert_neurosynth_to_dataset(
            coordinates_file=neurosynth_db["coordinates"],
            metadata_file=neurosynth_db["metadata"],
            annotations_files=neurosynth_db["features"],
        )
        dset.save(neurosynth_dset_path)
    else:
        dset = Dataset.load(neurosynth_dset_path)

    # Collect ROI files
    roi_files = sorted(
        glob(op.join(roi_dir, "*.nii")) + glob(op.join(roi_dir, "*.nii.gz"))
    )
    if not roi_files:
        raise FileNotFoundError(f"No NIfTI ROIs found in {roi_dir}")

    print(f"Found {len(roi_files)} ROI file(s) in {roi_dir}", flush=True)

    voxel_thresh = round(p_to_z(ALE_PVAL, tail="one"), 2)

    for roi_path in roi_files:
        roi_name = op.basename(roi_path)
        roi_stem = roi_name.replace(".nii.gz", "").replace(".nii", "")
        print(f"\n=== Processing ROI: {roi_stem} ===", flush=True)

        # Load and (if needed) binarize ROI
        img = nib.load(roi_path)
        bin_img = _binarize(img)

        # Optional: save the binary version alongside results for traceability
        bin_out_dir = op.join(results_dir, "macm", roi_stem)
        os.makedirs(bin_out_dir, exist_ok=True)
        bin_mask_path = op.join(bin_out_dir, f"{roi_stem}_binary_mask.nii.gz")
        nib.save(bin_img, bin_mask_path)
        print(f"Saved binary mask: {bin_mask_path}", flush=True)

        # Select studies for meta-analysis
        sel_ids = dset.get_studies_by_mask(bin_img)
        sel_dset = dset.slice(sel_ids)
        n_foci_db = dset.coordinates.shape[0]
        n_foci_sel = sel_dset.coordinates.shape[0]
        n_exps_db = len(dset.ids)
        n_exps_sel = len(sel_dset.ids)
        print(
            f"Database: {n_exps_db} exps / {n_foci_db} foci | "
            f"Selected: {n_exps_sel} exps / {n_foci_sel} foci",
            flush=True,
        )

        # Output directory for this ROI
        output_dir = bin_out_dir
        results_file = op.join(output_dir, f"{roi_stem}_macm_result.pkl.gz")

        if not op.isfile(results_file):
            print(f"\tRunning MACM for {roi_stem}...", flush=True)
            ale_workflow = CBMAWorkflow(
                estimator=ALE(kernel__sample_size=20),
                corrector="montecarlo",
                diagnostics=["jackknife", "focuscounter"],
                voxel_thresh=voxel_thresh,
                output_dir=output_dir,
                n_cores=n_cores,
            )
            results = ale_workflow.fit(sel_dset)
            results.save(results_file)
            print(f"\tCompleted MACM for {roi_stem}.", flush=True)

            # Organize maps and tables in subfolders
            maps_dir = op.join(output_dir, "maps")
            tables_dir = op.join(output_dir, "tables")
            os.makedirs(maps_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            for f in glob(op.join(output_dir, "*.nii.gz")):
                if not f.endswith("_macm_result.pkl.gz"):
                    shutil.move(f, maps_dir)
            for f in glob(op.join(output_dir, "*.tsv")):
                shutil.move(f, tables_dir)
        else:
            print("\tLoading existing results...", flush=True)
            results = MetaResult.load(results_file)

        # Report
        report_path = op.join(output_dir, "report.html")
        if not op.isfile(report_path):
            print("\tGenerating report...", flush=True)
            run_reports(results, output_dir)
            print(f"\tReport saved: {report_path}", flush=True)
        else:
            print("\tReport already exists; skipping.", flush=True)


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
