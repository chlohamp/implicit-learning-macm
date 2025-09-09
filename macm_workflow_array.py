"""Workflow for running ROI-based meta-analytic connectivity modeling (MACM).

- Loads a prebuilt NiMARE Dataset from your derivatives folder (no downloads).
- Processes either a single ROI (--roi_file) or all NIfTI files in --roi_dir.
- Writes per-ROI outputs to derivatives/macm/<ROI_STEM>/ with maps, tables, and report.html.
- Runs TWO corrected outputs per ROI:
    1) More-lenient cluster-FWE (voxel-forming z≈1.65) via Monte Carlo.
    2) FDR q=0.05 (voxelwise).
- Keeps diagnostics (jackknife, focuscounter) for each run.
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

# Correctors
_HAVE_FWECORRECTOR = True
_HAVE_FDRCORRECTOR = True
try:
    from nimare.correct import FWECorrector
except Exception:
    _HAVE_FWECORRECTOR = False
try:
    from nimare.correct import FDRCorrector
except Exception:
    _HAVE_FDRCORRECTOR = False


def _get_parser():
    parser = argparse.ArgumentParser(description="Run ROI-based MACM workflow")
    parser.add_argument("--project_dir", required=True, help="Path to project directory")
    parser.add_argument("--roi_dir", required=True, help="Path to ROI directory containing .nii/.nii.gz")
    parser.add_argument("--roi_file", required=False, help="Path to a single ROI NIfTI to process")
    parser.add_argument("--n_cores", default=4, required=False, help="Number of CPUs NiMARE can use")
    return parser


# ---------- Tunables ----------
# For diagnostics / mask binarization when needed:
ALE_PVAL = 0.01                  # voxel threshold (one-tailed) used only for binarizing non-binary ROIs
MIN_EXPERIMENTS = 5              # skip ROIs selecting fewer than this many experiments
KERNEL_SAMPLE_SIZE = 20          # used by ALE kernel if sample sizes missing

# FWE Monte Carlo:
MC_N_ITERS = 20000               # iterations for stable null (more lenient voxel-forming z compensates)
FWE_CLUSTER_SIZE = 10            # cluster size (voxels)
FWE_VOX_Z = 1.65                 # voxel-forming z (~p=.05, one-tailed)

# FDR:
FDR_Q = 0.05                     # voxelwise FDR level
FDR_METHOD = "indep"             # "indep" or "negcorr"
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


def _move_outputs_into(out_dir: str, subdir: str):
    """Move .nii.gz and .tsv outputs produced in out_dir into subfolders for each correction pass."""
    maps_dir = op.join(subdir, "maps")
    tables_dir = op.join(subdir, "tables")
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    for f in glob(op.join(out_dir, "*.nii.gz")):
        # keep pkl.gz where it is (saved separately)
        if not f.endswith(".pkl.gz"):
            shutil.move(f, maps_dir)
    for f in glob(op.join(out_dir, "*.tsv")):
        shutil.move(f, tables_dir)


def _run_workflow(sel_dset, out_dir, n_cores, estimator, corrector, voxel_thresh, tag):
    """Run CBMAWorkflow with a given corrector into a tagged subfolder."""
    subdir = op.join(out_dir, tag)
    os.makedirs(subdir, exist_ok=True)
    results_file = op.join(subdir, f"{op.basename(out_dir)}_{tag}_macm_result.pkl.gz")

    if op.isfile(results_file):
        print(f"\t[{tag}] Loading existing results...", flush=True)
        return MetaResult.load(results_file), subdir

    print(f"\t[{tag}] Running MACM...", flush=True)
    wf = CBMAWorkflow(
        estimator=estimator,
        corrector=corrector,
        diagnostics=["jackknife", "focuscounter"],
        voxel_thresh=voxel_thresh,   # cluster-forming z used by some correctors (FWE)
        output_dir=subdir,           # write temp outputs into the subdir
        n_cores=n_cores,
    )
    results = wf.fit(sel_dset)
    results.save(results_file)
    _move_outputs_into(subdir, subdir)
    print(f"\t[{tag}] MACM complete.", flush=True)
    return results, subdir


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

    # Threshold used only if we need to binarize a non-binary ROI
    diag_vox_z = round(p_to_z(ALE_PVAL, tail="one"), 2)

    for roi_path in roi_files:
        roi_name = op.basename(roi_path)
        roi_stem = roi_name.replace(".nii.gz", "").replace(".nii", "")
        print(f"\n=== ROI: {roi_stem} ===", flush=True)

        img = nib.load(roi_path)
        bin_img = _binarize(img, z_thresh=diag_vox_z, clust_thresh=10)

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

        # Guard: skip empty/tiny selections
        if n_exps_sel < MIN_EXPERIMENTS or n_foci_sel == 0:
            reason = (
                f"SKIPPED: insufficient data for ALE (experiments={n_exps_sel}, "
                f"foci={n_foci_sel}, min_experiments={MIN_EXPERIMENTS})."
            )
            print(reason, flush=True)
            with open(op.join(out_dir, "SKIPPED.txt"), "w") as fp:
                fp.write(reason + "\n")
            continue

        # Estimator
        ale_est = ALE(kernel__sample_size=KERNEL_SAMPLE_SIZE)

        # -------- Pass 1: More-lenient cluster-FWE (z≈1.65) --------
        if _HAVE_FWECORRECTOR:
            try:
                fwe_corrector = FWECorrector(
                    method="montecarlo",
                    n_iters=MC_N_ITERS,
                    voxel_thresh=FWE_VOX_Z,          # voxel-forming z (~p=.05 one-tailed)
                    cluster_threshold=FWE_CLUSTER_SIZE,
                    two_sided=True,                  # set False if you have a one-sided hypothesis
                    n_cores=n_cores,
                )
                fwe_results, fwe_dir = _run_workflow(
                    sel_dset=sel_dset,
                    out_dir=out_dir,
                    n_cores=n_cores,
                    estimator=ale_est,
                    corrector=fwe_corrector,
                    voxel_thresh=FWE_VOX_Z,
                    tag="fwe_p05",
                )
            except TypeError:
                # Older NiMARE may not accept some kwarg names; retry with minimal args.
                fwe_corrector = FWECorrector(method="montecarlo", n_iters=MC_N_ITERS)
                fwe_results, fwe_dir = _run_workflow(
                    sel_dset, out_dir, n_cores, ale_est, fwe_corrector, FWE_VOX_Z, "fwe_p05"
                )
        else:
            print("\t[FWE] FWECorrector not available; skipping cluster-FWE.", flush=True)

        # -------- Pass 2: FDR q=0.05 --------
        if _HAVE_FDRCORRECTOR:
            try:
                fdr_corrector = FDRCorrector(method=FDR_METHOD, q=FDR_Q)
                # FDR doesn't use cluster-forming z, but we must pass something; it is ignored.
                fdr_results, fdr_dir = _run_workflow(
                    sel_dset=sel_dset,
                    out_dir=out_dir,
                    n_cores=n_cores,
                    estimator=ale_est,
                    corrector=fdr_corrector,
                    voxel_thresh=FWE_VOX_Z,
                    tag=f"fdr_q{str(FDR_Q).replace('.', '')}",
                )
            except Exception as e:
                print(f"\t[FDR] Failed to run FDR correction: {e}", flush=True)
        else:
            print("\t[FDR] FDRCorrector not available; skipping FDR.", flush=True)

        # -------- Per-ROI report (top-level for convenience) --------
        # Prefer FDR if it exists; else FWE; else skip
        report_src = None
        for candidate in [
            op.join(out_dir, f"fdr_q{str(FDR_Q).replace('.', '')}", f"{roi_stem}_fdr_q{str(FDR_Q).replace('.', '')}_macm_result.pkl.gz"),
            op.join(out_dir, "fwe_p05", f"{roi_stem}_fwe_p05_macm_result.pkl.gz"),
        ]:
            if op.isfile(candidate):
                report_src = candidate
                break

        if report_src:
            results = MetaResult.load(report_src)
            report_path = op.join(out_dir, "report.html")
            if not op.isfile(report_path):
                print("\tGenerating report...", flush=True)
                run_reports(results, out_dir)
                print(f"\tReport saved: {report_path}", flush=True)
            else:
                print("\tReport already exists; skipping.", flush=True)
        else:
            print("\tNo result found for report generation.", flush=True)


def _main(argv=None):
    args = _get_parser().parse_args(argv)
    main(**vars(args))


if __name__ == "__main__":
    _main()
