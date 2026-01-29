# Implicit Learning Meta-analysis

This repository conducts a meta-analytic connectivity modeling analysis for regions that were posed by [Kovacs et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/33630631/) to be involved in implicit learning and subsequent automatization of learned rules.


As we were not interested in visual stimuli, we are instead including regions of interest in clique 1 that were found in [Ramage et al. (2024)](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00355/124919/Elucidating-a-statistical-learning-brain-network/) to be associated with implicit artificial grammar learning, as well as the globus pallidus interna from [Manes et al. (2013)](https://pubmed.ncbi.nlm.nih.gov/25050431/) and medial dorsal nucleus from Iwabuchi et al. 2024.



Centroid coordinates for the 8 volumes included in the analysis are:

| x   | y   | z   | Label     |
|-----|-----|-----|-----------|
| 50  | 26  | 4   | rIFGtri1  |
| -40 | 10  | 18  | lIFGop7   |
| 10  | 10  | 6   | rCaud31   |
| -8  | 4   | 6   | lCaud33   |
| 8   | -15 | 6   | rMDN      |
| -6  | -15 | 7   | lMDN      |
| 11  | 2   | 2   | rGPi      |
| -12 | 2   | 2   | lGPi      |


<!--
| x   | y   | z   | Label     |
|-----|-----|-----|-----------|
| 50  | 26  | 4   | rIFGtri1  |
| 48  | 24  | 22  | rMFG2     |
| 58  | 16  | 20  | rIFGop3   |
| 50  | 40  | 0   | rFpole4   |
| 50  | 8   | 42  | rMFG5     |
| 50  | 10  | 38  | rMFG6     |
| -40 | 10  | 18  | lIFGop7   |
| -38 | 22  | -2  | lIns8     |
| -50 | 36  | 8   | lIFGtri9  |
| -44 | 28  | 20  | lIFGtri10 |
| 10  | 10  | 6   | rCaud31   |
| -8  | 4   | 6   | lCaud33   |
| 8   | -15 | 6   | rMDN      |
| -6  | -15 | 7   | lMDN      |
| 11  | 2   | 2   | rGPi      |
| -12 | 2   | 2   | lGPi      |
-->

### Analysis

- **macm_workflow_array.py**: The main Python workflow for running ROI-based meta-analytic connectivity modeling (MACM) analyses. It loads a prebuilt NiMARE Dataset, processes either a single ROI or all NIfTI files in a directory, and outputs results (maps, tables, reports) to derivatives/macm/<ROI>/. Both FDR and cluster-FWE corrections are run per ROI. Use this script for batch or automated MACM analyses.

- **run_macm_array.sh**: A SLURM batch script for running MACM analyses on a computing cluster. It sets up the environment and paths, iterates over all ROIs in the dataset, and submits array jobs for parallel MACM processing. Use this script to efficiently run large-scale MACM analyses on HPC systems.

- **extract-macm-studies.ipynb**: An interactive Jupyter notebook for exploring and analyzing the MACM study selection process. It allows you to load all ROI-specific study CSVs, compare intersections of studies across any subset or all ROIs, and output new CSVs for custom ROI combinations. This notebook is ideal for flexible, exploratory analysis and for generating summary tables of study overlap.


### Visualizations
#### Must use:

python 3.8, 3.9, 3.10, and 3.11
nilearn 0.10.2
netneurotools 0.2.4

(I personally use Python 3.9.6)

#### To install the necessary packages: 

Install in this order:

1. First install gradec:
   ```
   pip install git+https://github.com/JulioAPeraza/gradec.git
   ```

2. Then install seaborn:
   ```
   pip install seaborn
   ```

3. Install remaining packages:
   ```
   pip install numpy pandas matplotlib neuromaps nibabel nilearn==0.10.2 surfplot scipy
   ```

4. **Important**: Ensure netneurotools is version 0.2.4:
   ```
   pip install netneurotools==0.2.4
   ```
