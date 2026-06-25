import numpy as np
import nibabel as nib

subj_list = ["sub-13", "sub-14", "sub-16", "sub-18", "sub-19"]

for ID in subj_list:
	bvals = np.loadtxt(f"/Users/ecd25/Desktop/PhD_Diss/Glioma/OA/Glioma/{ID}/bvals")
	bvecs = np.loadtxt(f"/Users/ecd25/Desktop/PhD_Diss/Glioma/OA/Glioma/{ID}/bvecs")
	data = nib.load(f"/Users/ecd25/Desktop/PhD_Diss/Glioma/OA/Glioma/{ID}/data.nii.gz")

	bvals_b0_b1500 = f"/Users/ecd25/Desktop/PhD_Diss/Glioma/OA/Glioma/{ID}/bvals_b0_b1500"
	bvecs_b0_b1500 = f"/Users/ecd25/Desktop/PhD_Diss/Glioma/OA/Glioma/{ID}/bvecs_b0_b1500"
	data_b0_b1500 = f"/Users/ecd25/Desktop/PhD_Diss/Glioma/OA/Glioma/{ID}/data_b0_b1500.nii.gz"

	# Only b 1500
	mask = ((bvals >= 0) & (bvals <= 10)) | ((bvals >= 1490) & (bvals <= 1610))

	# Alter bvals and bvecs file
	bvals_filtered = bvals[mask]
	bvecs_filtered = bvecs[:, mask]
	data_filtered = data.get_fdata()[..., mask]

	# Save
	np.savetxt(bvals_b0_b1500, bvals_filtered[np.newaxis, :], fmt='%g')
	np.savetxt(bvecs_b0_b1500, bvecs_filtered, fmt='%.6f')
	nib.save(nib.Nifti1Image(data_filtered, data.affine, data.header), data_b0_b1500)

	print("data updated for ", {ID})
