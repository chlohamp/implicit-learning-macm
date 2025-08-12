"""Workflow for running roi-based meta-analysic connectivity modeling."""

### needs to be optimized to run 3 rois all at once ###

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
from nimare.meta import ALE
from nimare.meta.cbma import ALESubtraction
from nimare.reports.base import run_reports
from nimare.results import MetaResult
from nimare.transforms import p_to_z
from nimare.workflows.cbma import CBMAWorkflow, PairwiseCBMAWorkflow
from nilearn.image import threshold_img, index_img, resample_to_img



def _get_parser():
    parser = argparse.ArgumentParser(description="Run ROI-based meta-analysis workflow")
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

def main(project_dir, n_cores):
    n_cores = int(n_cores)
    project_dir = op.abspath(project_dir)
    data_dir = op.join(project_dir, "data")
    results_dir = op.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    #this doesnt work on hpc
    #have to download neurosynth on local and tranfer

    # Load Neurosynth database
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

    rois = ['OMNI']
    z_thresh = 2.33
    clust_thresh = 10
    
    for roi in rois:
        nii_zmap = nib.load(op.join(data_dir, f'z_desc-mass_level-cluster_corr-FWE_method-montecarlo_{roi}.nii.gz'))
        nii_zmap_thr = threshold_img(nii_zmap, z_thresh, cluster_threshold=clust_thresh)

        thresh_mask_path = op.join(data_dir, f'z_desc-mass_level-cluster_corr-FWE_method-montecarlo_thresh_{roi}.nii.gz')
        nib.save(nii_zmap_thr, thresh_mask_path)
        print(f"Thresholded ROI for {roi} saved to {thresh_mask_path}", flush=True)

        # Force a binary mask by applying a threshold of 0 and then casting to integer (0 or 1)
        nii_zmap_bin = threshold_img(nii_zmap_thr, threshold=0)
        
        # Ensure the result is binary by converting it to 0 or 1 explicitly
        binary_data = nii_zmap_bin.get_fdata()
        binary_data[binary_data > 0] = 1  # Force all non-zero values to 1
        nii_zmap_bin = nib.Nifti1Image(binary_data, nii_zmap_bin.affine)
        
        # Get the image data as a numpy array
        bin_image_data = nii_zmap_bin.get_fdata()
        
        # Print basic information about the data
        print(f"ROI: {roi}", flush=True)
        print("Shape:", bin_image_data.shape, flush=True)
        print("Data type:", bin_image_data.dtype, flush=True)
        print("Min value:", bin_image_data.min(), flush=True)
        print("Max value:", bin_image_data.max(), flush=True)
        print("Mean value:", bin_image_data.mean(), flush=True)

        # Save the binary mask to the data directory
        binary_mask_path = op.join(data_dir, f'binary_mask_{roi}.nii.gz')
        nib.save(nii_zmap_bin, binary_mask_path)
        print(f"Binary mask for {roi} saved to {binary_mask_path}", flush=True)

        # Select studies for meta-analysis
        sel_ids = dset.get_studies_by_mask(nii_zmap_bin)
        sel_dset = dset.slice(sel_ids)
        n_foci_db = dset.coordinates.shape[0]
        n_foci_sel = sel_dset.coordinates.shape[0]
        n_exps_db = len(dset.ids)
        n_exps_sel = len(sel_dset.ids)
        
        # Make output directory for MACM results

        output_dir = op.join(results_dir, "macm", f"{roi}-2")
        os.makedirs(output_dir, exist_ok=True)

        results_file = op.join(output_dir, f"{roi}-macm_result.pkl.gz")

        if not op.isfile(results_file):
            print(f"\tRunning MACM for {roi}...", flush=True)
            #run the MACM
            ALE_PVAL = 0.01
            voxel_thresh = round(p_to_z(ALE_PVAL, tail="one"), 2)
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
            print(f"\tCompleted MACM for {roi}...", flush=True)
            
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

def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
