#Import path + fsl

import subprocess
from pathlib import Path

#List subjs (test) and root path

IDs = "sub-13", "sub-14", "sub-16", "sub-18", "sub-19"
root = Path("/Users/ecd25/Desktop/PhD_Diss/Glioma/OA/Glioma")

#Fit DTI and FW to data

for ID in IDs:
        subj = root / ID
        mask = subj / "nodif.nii.gz"
        out_mask = subj / "nodif_brain.nii.gz"

        subprocess.run(["bet", f"{mask}", f"{out_mask}", f"-f", "0.25", "-m"])
        print("bet has been run for", ID)


