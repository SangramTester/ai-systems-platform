import os
from PIL import Image
import scipy.io
from torch.utils.data import Dataset
import time

class OxfordFlowersDataSet(Dataset):
  def __init__(self, data_dir: str, transform=None):
    self.data_dir = data_dir
    self.img_dir = os.path.join(data_dir, "jpg")
    self.transform = transform
    self.error_log = []
    
    # load labels
    labels_mat = scipy.io.loadmat(os.path.join(data_dir, "imagelabels.mat"))
    labels_array = labels_mat['labels']
    assert labels_array is not None, "'labels' key not found in .mat file"
    self.labels = labels_array[0] - 1
 
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    start_time = time.time()
    for _ in range(len(self)):
      try:
        img_name = os.path.join(self.img_dir, f"image_{idx+1:05d}.jpg")
        image = Image.open(img_name).convert("RGB")
        label = self.labels[idx]
        image.verify()
        image = Image.open(img_name).convert("RGB")

        if image.size[0] < 32 or image.size[1] < 32:
          raise ValueError(f"Image at index {idx} is smaller than 32x32 pixels. {image.size}")

        if self.transform:
          image = self.transform(image)
        return image, label
      except Exception as e:
        self.error_log.append((idx, str(e)))
        print(f"Error loading image at index {idx}: {e}")
        idx = (idx + 1) % len(self)
    raise RuntimeError("All images failed to load.")
  

  def get_error_summary(self):
    for error in self.error_log[:5]:
      print(f"Index: {error[0]}, Error: {error[1]}")
    if len(self.error_log) > 5:
      print(f"... and {len(self.error_log) - 5} more errors.")