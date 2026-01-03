import time
from dataset import OxfordFlowersDataSet

class MonitoredDataset(OxfordFlowersDataSet):
    def __init__(self, data_dir: str, transform=None):
      super().__init__(data_dir, transform)
      self.access_counts = {}
      self.load_times = []

    def __len__(self):
      return super().__len__()

    def __getitem__(self, idx):
      start_time = time.time()
      self.access_counts[idx] = self.access_counts.get(idx, 0) + 1
      
      item = super().__getitem__(idx)

      load_time = time.time() - start_time
      self.load_times.append(load_time)

      if load_time > 1.0:  # threshold in seconds
        print(f"Warning: Loading image at index {idx} took {load_time:.2f} seconds.")

      return item
