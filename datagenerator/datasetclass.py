import torch as T
import torch.nn as nn
from .DG import generate_data

class SynthSignalsDataset(T.utils.data.Dataset):
  """
  A PyTorch Dataset class for synthetic signal data.

  This dataset is designed to handle synthetic signals generated using the 
  `generate_data` function. It supports loading signal data, ground truth 
  labels, and signal class information, and is compatible with PyTorch's 
  DataLoader for efficient batching and shuffling.

  Attributes:
    x_data (torch.Tensor): A tensor containing the input signal data.
    y_data (torch.Tensor): A tensor containing the ground truth labels.
    signalclass (torch.Tensor): A tensor containing the signal class information.

  Args:
    num_samples (int, optional): The number of samples to generate. Defaults to None.
    device (torch.device, optional): The device to store the tensors on (e.g., 'cpu' or 'cuda'). Defaults to None.
    noise_level (float, optional): The level of noise to add to the generated data. Defaults to 0.0.

  Methods:
    __len__(): Returns the number of samples in the dataset.
    __getitem__(idx): Retrieves a sample from the dataset at the specified index.

  Example:
    >>> dataset = SynthSignalsDataset(num_samples=1000, device='cuda', noise_level=0.1)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for batch in dataloader:
    ...     print(batch['signals'], batch['gt'], batch['sc'])
  """
  def __init__(self, params, num_samples=None, device=None):
    x_tmp, y_tmp, signalclass = generate_data(params, num_samples)
    self.x_data = T.tensor(x_tmp,
      dtype=T.float32).to(device)
    self.y_data = T.tensor(y_tmp,
      dtype=T.long).to(device)
    self.signalclass = T.tensor(signalclass,
      dtype=T.long).to(device)

  def __len__(self):
    return len(self.x_data)  # required

  def __getitem__(self, idx):
    if T.is_tensor(idx):
      idx = idx.tolist()
    
    signals = self.x_data[idx, :]
    gt = self.y_data[idx,:,:]
    signalclass = self.signalclass[idx]
    
    sample = { 'signals' : signals, 'gt' : gt, 'sc': signalclass }
    
    return sample
