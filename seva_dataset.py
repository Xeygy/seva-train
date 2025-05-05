import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
import json
from glob import glob
def dict_arrays_to_dict_tensors(dict_of_arrays):
    """
    Converts a dictionary of arrays (NumPy arrays) to a dictionary of tensors (PyTorch tensors).

    Args:
        dict_of_arrays (dict): A dictionary where keys are strings and values are NumPy arrays.

    Returns:
        dict: A dictionary with the same keys as the input, but values are PyTorch tensors.
    """
    dict_of_tensors = {}
    for key, array in dict_of_arrays.items():
        dict_of_tensors[key] = torch.from_numpy(array)
    return dict_of_tensors


class SevaDataset(Dataset):
    def __init__(self, dataPath, T=21):
        super(SevaDataset, self).__init__()
        self.dataPath = dataPath
        self.scene_dirs = sorted(glob(f"{self.dataPath}/*/*"))
        assert T <= 21
        self.T  = T
    
    def __len__(self):
        return len(self.scene_dirs)
    
    def load_train_test_split(self,scene_dir):
        json_file = glob(f"{scene_dir}/train_test_split*")[0]
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data["train_ids"], data["test_ids"]
    
    def rescale_batch(self, train_views, test_views):
        num_views = len(train_views) + len(test_views)
        # Compute target counts based on ratio
        test_ratio = len(test_views) / num_views

        num_tests = min(max(round(self.T * test_ratio), 1), len(test_views))
        num_trains = min(max(self.T  - num_tests, 1), len(train_views))

        # In case clamping broke the sum, fix by adjusting
        while num_trains + num_tests > self.T:
            if num_tests > 1:
                num_tests -= 1
            else:
                num_trains -= 1

        train_indices = np.arange(0, len(train_views))
        test_indices = np.arange(len(train_views), len(train_views) + len(test_views))
        train_indices_rescale = np.sort(np.random.choice(train_indices,num_trains, replace=False ))
        test_indices_rescale = np.sort(np.random.choice(test_indices,num_tests, replace=False ))

        return train_indices_rescale, test_indices_rescale

    def __getitem__(self, index):
        scene_dir = self.scene_dirs[index]
        x_in = np.load(f"{scene_dir}/x_in.npy")
        x_out = np.load(f"{scene_dir}/x_out.npy")
        cond = dict(np.load(f"{scene_dir}/cond.npz"))
        train_views, test_views = self.load_train_test_split(scene_dir) 
        num_views = len(train_views) + len(test_views)
        if num_views <= self.T:
            train_indices = np.arange(0, len(train_views))
            test_indices = np.arange(len(train_views), len(train_views) + len(test_views))
            x_in = x_in[:self.T]
            x_out = x_out[:self.T]
            cond = {k:v[:self.T] for k,v in cond.items()}
        else:
            ## Rescale to a batch of T
            train_indices, test_indices = self.rescale_batch(train_views, test_views)
            all_indices = np.concatenate((train_indices, test_indices))
            
            x_in = x_in[all_indices]
            x_out = x_out[all_indices]
            cond = {k:v[all_indices] for k,v in cond.items()}
            train_indices = np.arange(0, len(train_indices))
            test_indices = np.arange(len(train_indices), len(train_indices) + len(test_indices))


        x_in = torch.from_numpy(x_in)
        x_out = torch.from_numpy(x_out)
        train_indices = torch.from_numpy(train_indices)
        test_indices = torch.from_numpy(test_indices)
        cond = dict_arrays_to_dict_tensors(cond)

        return {
            "x_in": x_in,
            "x_gt": x_out,
            "cond": cond,
            "train_indices": train_indices,
            "test_indices": test_indices
        }
    