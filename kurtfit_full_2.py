#Import path + fsl

import subprocess
from pathlib import Path

#List subjs (test) and root path

IDs = "sub-13", "sub-14", "sub-16", "sub-18", "sub-19"
root = Path("/Users/ecd25/Desktop/PhD_Diss/Glioma/OA/Glioma")

#Fit DTI and FW to data

for ID in IDs:
	subj = root / ID
	data = subj / "data.nii.gz"
	bvecs = subj / "bvecs"
	bvals = subj / "bvals"
	mask = subj / "nodif_brain_mask.nii.gz"
	out = subj / "kurtfit" 

	subprocess.run(["kurtfit", f"-o={out}", f"-k={data}", f"-m={mask}", f"-r={bvecs}", f"-b={bvals}", f"--model=dti", "-f"], check=True)

	print("kurtfit has been run for", ID)
