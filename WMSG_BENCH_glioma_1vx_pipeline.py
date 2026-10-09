# Importing tools
import numpy as np
import subprocess
from pathlib import Path

from file_tree import FileTree
from fsl_pipe import Pipeline, In, Out

from fsl import wrappers
from fsl.wrappers import flirt, bet, fslmaths, applyxfm, fnirt, epi_reg, convertwarp, applywarp
from fsl.wrappers import wrapperutils as wutils
from fsl.utils import assertions as asrt

KURTFIT = ("/Users/ecd25/miniconda3/envs/numba-py311/bin/kurtfit")

# Trying to write a wrapper to call fslmeants
@wutils.fileOrImage('input', 'output')
@wutils.fslwrapper
def fslmeants(input, output, *args):
    """Wrapper for the ``fslmeants`` tool."""
    asrt.assertIsNifti(input)
    cmd = ['fslmeants', input, output] + [str(a) for a in args]
    return cmd

# Trying to write a wrapper to call select_dwi_vols
@wutils.fileOrImage('input', 'output')
@wutils.fslwrapper
def select_dwi_vols(input, bvals, output, approx_bval, **kwargs):
    """Wrapper for the ``select_dwi_vols`` tool."""
    asrt.assertIsNifti(input)
    cmd = ['select_dwi_vols', input, bvals, output, str(approx_bval)]
    return cmd + wutils.applyArgStyle('-', **kwargs)

# Load the file-tree describing the data directory
BASE = "/Users/ecd25/Desktop/BENCH/BENCH_Tutorial/one_vs_many/preop"
tree = FileTree.read(f"{BASE}/data.tree", top_level=BASE)
tree = tree.update_glob("T1w")
print("subject:", tree.placeholders.get("subject"))

# Create the pipeline
pipe = Pipeline()

@pipe
def preproc1(T1w: In, T1w_brain: Out, T1w_brain_mask: Out):
    """BET T1"""
    bet(T1w, T1w_brain, mask=T1w_brain_mask)

@pipe
def avb0(DWI: In, bvals: In, rB0: Out, B0: Out):
    """Averaging B0"""
    select_dwi_vols(DWI, bvals, rB0, approx_bval=0)
    fslmaths(rB0).Tmean().run(B0)

@pipe
def betb0(B0: In, B0_brain: Out, B0_brain_mask: Out):
    """Brain extract B0 for running KURTFIT"""
    bet(B0, B0_brain, mask=B0_brain_mask)

@pipe
def run_initial_affine(T1w_brain: In, MNI: In, T1_to_MNI_mat: Out, Aff: Out):
    """FLIRT 12-dof affine, skull-stripped T1 -> MNI brain."""
    flirt(T1w_brain, ref=MNI, omat=T1_to_MNI_mat, out=Aff, dof=12, cost="corratio", searchrx=(-90, 90), searchry=(-90, 90), searchrz=(-90, 90))

@pipe
def run_nonlinear_registration(T1w:In, MNI_HEAD: In, T1_to_MNI_mat: In, MNI_cnf: In, WARP: Out, NONLIN: Out):
    """FNIRT, T1 2 MNI, initialised with the affine."""
    fnirt(T1w, ref=MNI_HEAD, aff=T1_to_MNI_mat, config=MNI_cnf, cout=WARP, iout=NONLIN)

@pipe
def run_diff2standard(B0: In, T1w: In, T1w_brain: In, diff2str: Out, diff2str_mat: Out):
    """Diffusion 2 MNI warp."""
    diff2str_prefix = str(diff2str).removesuffix(".nii.gz")
    epi_reg(epi=B0, t1=T1w, t1brain=T1w_brain, out=diff2str_prefix)

@pipe
def conwarp(MNI: In, diff2str_mat: In, WARP: In, D_WARP: Out):
    """Converting to make diff2standard_warp"""
    convertwarp(ref=MNI, premat=diff2str_mat, warp1=WARP, out=D_WARP)

@pipe
def b0_to_mni(B0: In, MNI: In, D_WARP: In, B0_in_MNI: Out):
    """Mean b0 into MNI space"""
    applywarp(B0, ref=MNI, out=B0_in_MNI, warp=D_WARP, interp="spline")

@pipe 
def fwdti(DWI: In, bvals: In, bvecs: In, B0_brain_mask: In, KURT: Out):
    """KURT is the expected RMSE output file."""
    out_prefix = str(KURT).removesuffix("_rmse.nii.gz")
    cmd = [
        KURTFIT,
        "-o", out_prefix,
        "-k", str(DWI),
        "-m", str(B0_brain_mask),
        "-b", str(bvals),
        "-r", str(bvecs),
        "--model", "dti-kurt",
        "-f",
    ]
    subprocess.run(cmd, check=True)

@pipe
def select_ss(DWI: In, bvals: In, bvecs: In, b0b1200: Out):
    """Selecting b0 and b=1200 volumes for single-shell analysis."""
    select_dwi_vols(DWI,bvals, b0b1200, 0, b=1200, obv=bvecs)

@pipe
def diff_summary(b0b1200: In, b0b1200_bvec: In, b0b1200_bval: In, WM_common: In, D_WARP: In, SUM: Out):
    """Running BENCH summary metric extraction."""
    cmd = ["bench", "diff-summary", "--data", str(b0b1200), "--bvecs", str(b0b1200_bvec), "--bvals", str(b0b1200_bval), "--mask", str(WM_common), "--xfm", str(D_WARP), "--summarytype", "dt", "--output", str(SUM)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

@pipe
def run_glm(SUM_all: In, WM_common: In, D_mat: In, D_con: In, subjlist: In, GLM: Out):
    """Running BENCH GLM."""
    cmd = ["bench", "glm", "--summarydir", str(SUM_all), "--mask", str(WM_common), "--designmat", str(D_mat), "--designcon", str(D_con), "--subjectlist", str(subjlist), "--output", str(GLM)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

@pipe
def run_inference(SS_model: In, GLM: In, WM_common: In, Results: Out):
    """Running BENCH inference."""
    cmd = [
        "bench", "inference",
        "--changemodel", str(SS_model),
        "--glmdir", str(GLM),
        "--mask", str(WM_common),
        "--output", str(Results),
    ]
    print("Running:", " ".join(cmd))

    for name, path in [
        ("Change model", SS_model),
        ("GLM directory", GLM),
        ("WM mask", WM_common),
    ]:
        if not Path(str(path)).exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    pipe.cli(tree)