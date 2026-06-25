#Import path + fsl

import subprocess
from pathlib import Path

#List subjs (test) and root path

IDs = "sub-02", "sub-10", "sub-11"
root = Path("/Users/ecd25/Desktop/PhD_Diss/Glioma/OA/Glioma")

#Fit DTI and FW to data

for ID in IDs:
	subj = root / ID
	data = subj / "data_b0_b500_b1000_b1500.nii.gz"
	bvecs = subj / "bvecs_b0_b500_b1000_b1500"
	bvals = subj / "bvals_b0_b500_b1000_b1500"
	mask = subj / "nodif_brain_mask.nii.gz"
	out = subj / "kurtfit_mss" 

	subprocess.run(["kurtfit", f"-o={out}", f"-k={data}", f"-m={mask}", f"-r={bvecs}", f"-b={bvals}", f"--model=dti", "-f"], check=True)

	print("kurtfit has been run for ms for ", ID)
