import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.io import loadmat
import copy
import mri_utils
from fft_utils import numpy_2_complex
import numpy as np
DEFAULT_OPTS = {'root_dir':'data/knee',
				'name':'coronal_pd', 
				'patients':[1,2,3,4,5,6,7,8,9,10],
				'start_slice':11,'end_slice':30,
				'eval_patients':[11,12,13,14,15,16,17,18,19,20],
				'eval_slices':[x for x in range(10,30)],'mode':'train',
				'load_target':True,'sampling_pattern':'cartesian_with_os',
				'normalization':'max'} 

class KneeDataset(Dataset):
	""" MRI knee dataset with k-space raw data, coil sensitivities and sampling mask
	Adapted from Hammernik et al """
	def __init__(self, **kwargs):
		"""
		Parameters:
		root_dir: str
			root directory of data
		dataset_name: list of str
			list of directory to load data from
		transform: 
		"""
		options = DEFAULT_OPTS

		for key in kwargs.keys():
			options[key] = kwargs[key]

		self.options = options
		self.root_dir = Path(self.options['root_dir'])
		# Processing directory
		if not options['name'] in ['coronal_pd','axial_t2','coronal_pd_fs','sagittal_pd','sagittal_t2']:
			raise ValueError('Dataset {} not supported!'.format(options['name']))

		self.filename = []
		self.coil_sens_list = []
		data_dir = self.root_dir / options['name']

		# Load raw data and coil sensitivities name
		if options['mode'] == 'train':
			patient_key = 'patients'
			slice_no = [x for x in range(options['start_slice'],options['end_slice']+1)]
		elif options['mode'] == 'eval':
			patient_key = 'eval_patients'
			slice_no = options['eval_slices']

		for patient in options[patient_key]:
			patient_dir = data_dir / str(patient)
			for i in slice_no:
				slice_dir = patient_dir / 'rawdata{}.mat'.format(i)
				self.filename.append(str(slice_dir))
				coil_sens_dir = patient_dir / 'espirit{}.mat'.format(i)
				self.coil_sens_list.append(str(coil_sens_dir))

		
		self.mask_dir = data_dir/ 'masks'
		self.mask_dir = list(self.mask_dir.glob('*at4*'))
		self.mask = loadmat(str(self.mask_dir[0]))
		self.mask = self.mask['mask'].astype(np.float32)

	def __len__(self):
		return len(self.filename)

	def __getitem__(self,idx):
		mask = copy.deepcopy(self.mask)
		filename = self.filename[idx]
		coil_sens = self.coil_sens_list[idx]

		raw_data = loadmat(filename)
		f = np.ascontiguousarray(np.transpose(raw_data['rawdata'],(2,0,1))).astype(np.complex64)
		
		coil_sens_data = loadmat(coil_sens)
		c = np.ascontiguousarray(np.transpose(coil_sens_data['sensitivities'],(2,0,1))).astype(np.complex64)

		if self.options['load_target']:
			ref = coil_sens_data['reference'].astype(np.complex64)
		else:
			ref = np.zeros_like(mask,dtype=np.complex64)

		if 'padlength_left' in raw_data and 'padlength_right' in raw_data:
			padlength_left = int(raw_data['padlength_left'])
			padlength_right = int(raw_data['padlength_right'])
		else:
			padlength_left = 0
			padlength_right = 0

		if padlength_left > 0:
			mask[:,:padlength_left] = 1
		if padlength_right  > 0:
			mask[:,-padlength_right:] = 1

		# mask rawdata
		f *= mask

		# compute initial image input0
		input0 = mri_utils.mriAdjointOp(f,c,mask).astype(np.complex64)

		# remove frequency encoding oversampling
		if self.options['sampling_pattern'] == 'cartesian_with_os':
			if self.options['load_target']:
				ref = mri_utils.removeFEOversampling(ref) # remove RO Oversampling
			input0 = mri_utils.removeFEOversampling(input0) # remove RO Oversampling

		elif self.options['sampling_pattern'] == 'cartesian':
			pass
		else:
			raise ValueError('sampling_pattern has to be in [cartesian_with_os, cartesian]')

		# normalize the data
		if self.options['normalization'] == 'max':
			norm = np.max(np.abs(input0))
		elif self.options['normalization'] == 'no':
			norm = 1.0
		else:
			raise ValueError("Normalization has to be in ['max','no']")

		f /= norm
		input0 /= norm

		if self.options['load_target']:
			ref /= norm
		else:
			ref = np.zeros_like(input_0)

		input0 = numpy_2_complex(input0)
		f = numpy_2_complex(f)
		c = numpy_2_complex(c)
		mask = torch.from_numpy(mask)
		ref = numpy_2_complex(ref)

		data = {'u_t':input0,'f':f,'coil_sens':c,'sampling_mask':mask,'reference':ref}
		return data

import os
import glob
import h5py

class FastMRISpecificDataset(Dataset):
    def __init__(self, data_dir, smaps_dir, target_shape=(16, 20, 640, 320), **kwargs):
        self.data_dir = data_dir
        self.smaps_dir = smaps_dir
        self.files = glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True)
        self.slice_indices = []
        
        # 遍历 Volume 下的所有切片
        for f_path in self.files:
            try:
                with h5py.File(f_path, 'r') as f:
                    if f['kspace'].shape == target_shape:
                        num_slices = f['kspace'].shape[0]
                        for s_idx in range(num_slices):
                            self.slice_indices.append((f_path, s_idx))
            except:
                pass

        mask_tmp = torch.zeros((1, 640, 320), dtype=torch.complex64)
        num_low = int(320 * 0.08)
        pad = (320 - num_low + 1) // 2
        mask_tmp[:, :, pad : pad + num_low] = 1

        torch.manual_seed(42) 
        high_mask = torch.rand(1, 1, 320) < (0.25 - (num_low/320))
        self.fixed_mask = torch.logical_or(mask_tmp.bool(), high_mask).to(torch.complex64)

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        f_path, s_idx = self.slice_indices[idx]
        file_name = os.path.basename(f_path)
        smap_path = os.path.join(self.smaps_dir, file_name.replace('.h5', '_smaps.h5'))

        with h5py.File(f_path, 'r') as f_orig, h5py.File(smap_path, 'r') as f_smap:
            k_volume = torch.tensor(f_orig['kspace'][:], dtype=torch.complex64)
            k_slice = k_volume[s_idx]
            target = torch.tensor(f_orig['reconstruction_rss'][s_idx], dtype=torch.float32) 
            smaps = torch.tensor(f_smap['smaps'][s_idx], dtype=torch.complex64)

            n_sl = k_volume.shape[0]
            f_volume_norm = torch.linalg.norm(k_volume)
            paper_norm = (torch.sqrt(torch.tensor(n_sl * 1.0)* 10000.0) ) / (f_volume_norm + 1e-8)

        mask = self.fixed_mask.clone()
        
        f = k_slice * mask
        
        # 使用简单的 Adjoint 操作把 kspace 转到图像域
        Finv = torch.fft.ifftshift(f * mask, dim=(-2, -1))
        Finv = torch.fft.ifft2(Finv, norm="ortho")
        Finv = torch.fft.fftshift(Finv, dim=(-2, -1))
        input0 = torch.sum(Finv * torch.conj(smaps), dim=-3)

        f_norm = f * paper_norm
        input0_norm = input0 * paper_norm
        target_norm = target * paper_norm

        data = {
            'u_t': torch.view_as_real(input0_norm.to(torch.complex64)),    
            'f': torch.view_as_real(f_norm.to(torch.complex64)),            
            'coil_sens': torch.view_as_real(smaps.to(torch.complex64)),    
            'sampling_mask': mask.real.to(torch.float32).squeeze(0),   # <--- 修改这里
            'reference': torch.view_as_real(target_norm.to(torch.complex64)), 
        }
        return data
