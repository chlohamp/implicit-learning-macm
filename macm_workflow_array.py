"""Workflow for running ROI-based meta-analytic connectivity modeling (MACM).

- Loads a prebuilt NiMARE Dataset from your derivatives folder (no downloads).
- Processes either a single ROI (--roi_file) or all NIfTI files in --roi_dir.
- Writes per-ROI outputs to derivatives/macm/<ROI_STEM>/ with maps, tables, and report.html.
- Uses FWECorrector with explicit n_iters to avoid NoneType errors in Monte Carlo correction.
"""

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

# Try to import FWECorrector (preferred). If unavailable, we’ll fall back to the string method.
try:
    from nimare.correct import FWECorrector
    _HAVE_FWECORRECTOR = True
except Exception:
    _HAVE_FWECORRECTOR = False


def _get_parser():
    parser = argparse.ArgumentParser(description="Run ROI-based MACM workflow")
    parser.add_argument("--project_dir", required=True, help="Path to project directory")
    parser.add_argument("--roi_dir", required=True, help="Path to ROI directory containing .nii/.nii.gz")
    parser.add_argument("--roi_file", required=False, help="Path to a single ROI NIfTI to process")
    parser.add_argument("--n_cores", default=4, required=False, help="Number of CPUs NiMARE can use")
    return parser


# ---------- Tunables ----------
ALE_PVAL = 0.01                 # voxel threshold, one-tailed (for diagnostics/reporting)
MIN_EXPERIMENTS = 5             # skip ROIs selecting fewer than this many experiments
MC_N_ITERS = 5000               # explicit Monte-Carlo iterations for FWE correction
KERNEL_SAMPLE_SIZE = 20         # used by ALE kernel if sample sizes missing
# ------------------------------


def _is_binary(img: nib.Nifti1Image) -> bool:
    data = np.asarray(img.dataobj)
    uniq = np.unique(data[np.isfinite(data)])
    return np.all(np.isin(uniq, [0, 1])) and uniq.size <= 2


def _binarize(img: nib.Nifti1Image, z_thresh=2.33, clust_thresh=10) -> nib.Nifti1Image:
    """Return a 0/1 mask. If not already binary, threshold and binarize."""
    if _is_binary(img):
        return img
    thr = threshold_img(img, threshold=z_thresh, cluster_threshold=clust_thresh)
    data = (thr.get_fdata() > 0).astype(np.int16)
    return nib.Nifti1Image(data, thr.affine, thr.header)


def main(project_dir, roi_dir, n_cores, roi_file=None):
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)
    roi_dir = op.abspath(roi_dir)

    # ===== Derivatives layout =====
    derivatives_dir = op.join(project_dir, "derivatives")
    macm_root = op.join(derivatives_dir, "macm")
    os.makedirs(macm_root, exist_ok=True)

    # ===== Load prebuilt NiMARE dataset from derivatives (no downloads) =====
    candidates = [
        op.join(derivatives_dir, "neurosynth", "neurosynth_terms_dataset.pkl.gz"),
        op.join(derivatives_dir, "neurosynth_terms_dataset.pkl.gz"),
        op.join(derivatives_dir, "neurosynth", "neurosynth_terms_set.pkl.gz"),
        op.join(derivatives_dir, "neurosynth_terms_set.pkl.gz"),
    ]
    hits = [p for p in candidates if op.isfile(p)]
    if not hits:
        hits = glob(op.join(derivatives_dir, "**", "neurosynth_terms*.pkl.gz"), recursive=True)
    if not hits:
        raise FileNotFoundError(
            "Couldn't find a prebuilt NiMARE dataset under derivatives/.\n"
            "Expected something like 'neurosynth_terms_dataset.pkl.gz'."
        )
    dset_path = hits[0]
    print(f"Loading NiMARE dataset: {dset_path}", flush=True)
    dset = Dataset.load(dset_path)

    # ===== Collect ROI files (single or all in dir) =====
    if roi_file:
        roi_files = [op.abspath(roi_file)]
    else:
        roi_files = sorted(glob(op.join(roi_dir, "*.nii")) + glob(op.join(roi_dir, "*.nii.gz")))
        if not roi_files:
            raise FileNotFoundError(f"No NIfTI ROIs found in {roi_dir}")

    print(f"Processing {len(roi_files)} ROI file(s).", flush=True)

    voxel_thresh = round(p_to_z(ALE_PVAL, tail="one"), 2)

    for roi_path in roi_files:
        roi_name = op.basename(roi_path)
        roi_stem = roi_name.replace(".nii.gz", "").replace(".nii", "")
        print(f"\n=== ROI: {roi_stem} ===", flush=True)

        img = nib.load(roi_path)
        bin_img = _binarize(img)

        # Per-ROI directory under derivatives/macm/<roi_stem>
        out_dir = op.join(macm_root, roi_stem)
        os.makedirs(out_dir, exist_ok=True)

        # Save binary mask for provenance
        bin_mask_path = op.join(out_dir, f"{roi_stem}_binary_mask.nii.gz")
        nib.save(bin_img, bin_mask_path)
        print(f"Saved binary mask: {bin_mask_path}", flush=True)

        # Select studies for meta-analysis
        sel_ids = dset.get_studies_by_mask(bin_img)
        sel_dset = dset.slice(sel_ids)
        n_exps_sel = len(sel_dset.ids)
        n_foci_sel = sel_dset.coordinates.shape[0] if hasattr(sel_dset, "coordinates") else 0
        print(f"Selected: {n_exps_sel} experiments / {n_foci_sel} foci", flush=True)

        # Guard: skip empty/tiny selections to avoid downstream errors
        if n_exps_sel < MIN_EXPERIMENTS or n_foci_sel == 0:
            reason = (
                f"SKIPPED: insufficient data for ALE (experiments={n_exps_sel}, "
                f"foci={n_foci_sel}, min_experiments={MIN_EXPERIMENTS})."
            )
            print(reason, flush=True)
            with open(op.join(out_dir, "SKIPPED.txt"), "w") as fp:
                fp.write(reason + "\n")
            continue

        results_file = op.join(out_dir, f"{roi_stem}_macm_result.pkl.gz")

        if not op.isfile(results_file):
            print("\tRunning MACM...", flush=True)

            # Estimator
            ale_est = ALE(kernel__sample_size=KERNEL_SAMPLE_SIZE)

            # Prefer a Corrector object so we can pass n_iters explicitly
            if _HAVE_FWECORRECTOR:
                corrector = FWECorrector(
                    method="montecarlo",
                    n_iters=MC_N_ITERS,
                    voxel_thresh=voxel_thresh,
                    n_cores=n_cores,
                )
                ale_workflow = CBMAWorkflow(
                    estimator=ale_est,
                    corrector=corrector,
                    diagnostics=["jackknife", "focuscounter"],
                    voxel_thresh=voxel_thresh,
                    output_dir=out_dir,
                    n_cores=n_cores,
                )
            else:
                # Fallback: old API (cannot pass kwargs). If this fails, we catch below.
                ale_workflow = CBMAWorkflow(
                    estimator=ale_est,
                    corrector="montecarlo",
                    diagnostics=["jackknife", "focuscounter"],
                    voxel_thresh=voxel_thresh,
                    output_dir=out_dir,
                    n_cores=n_cores,
                )

            try:
                results = ale_workflow.fit(sel_dset)
            except Exception as e:
                msg = f"FAILED during ALE on {roi_stem}: {type(e).__name__}: {e}"
                print("\t" + msg, flush=True)
                with open(op.join(out_dir, "ERROR.txt"), "w") as fp:
                    fp.write(msg + "\n")
                continue

            results.save(results_file)
            print("\tMACM complete.", flush=True)

            # Organize maps and tables
            maps_dir = op.join(out_dir, "maps")
            tables_dir = op.join(out_dir, "tables")
            os.makedirs(maps_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            for f in glob(op.join(out_dir, "*.nii.gz")):
                if not f.endswith("_macm_result.pkl.gz"):
                    shutil.move(f, maps_dir)
            for f in glob(op.join(out_dir, "*.tsv")):
                shutil.move(f, tables_dir)
        else:
            print("\tLoading existing results...", flush=True)
            results = MetaResult.load(results_file)

        # Report (saved within the ROI's folder)
        report_path = op.join(out_dir, "report.html")
        if not op.isfile(report_path):
            print("\tGenerating report...", flush=True)
            run_reports(results, out_dir)
            print(f"\tReport saved: {report_path}", flush=True)
        else:
            print("\tReport already exists; skipping.", flush=True)


def _main(argv=None):
    args = _get_parser().parse_args(argv)
    main(**vars(args))


if __name__ == "__main__":
    _main()
