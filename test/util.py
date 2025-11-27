import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import string
import pickle

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd

import glob
from itertools import zip_longest

ALPHABET = string.ascii_lowercase + string.digits + "."
char2idx = {c: i + 1 for i, c in enumerate(ALPHABET)}  # padding=0
idx2char = {i: c for c, i in char2idx.items()}  # Reverse mapping index -> character
vocab_size = len(char2idx) + 1

MAX_LEN = 50

def domain_to_tensor(domain):
    arr = [char2idx.get(c, 0) for c in domain.lower()][:MAX_LEN]
    arr += [0] * (MAX_LEN - len(arr))
    return torch.tensor(arr, dtype=torch.long)

class DomainDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dom, lbl = self.samples[idx]
        x = domain_to_tensor(dom)
        return x, lbl

# def transfer_pkl_to_csv():
#     with open('./domain2/dga_2_test.pkl', 'rb') as f:
#         data = pickle.load(f)

#     print(f"Data type: {type(data)}")
#     print(f"Data length: {len(data)}")

#     samples = data.samples 

#     df = pd.DataFrame(samples, columns=['domain', 'label'])
#     df.to_csv('./domain2/dga_2_test.csv', index=False)

#     print("Conversion complete!")
#     print(f"Total samples: {len(df)}")
#     print(df.head()) 

train_files = glob.glob('./domain2/*train*.csv')
print(f"Found {len(train_files)} train files: {train_files}")

# Đọc và gộp tất cả file train
train_dfs = []
for file in train_files:
    df = pd.read_csv(file)
    train_dfs.append(df)
    print(f"Loaded {file}: {len(df)} samples")

# Gộp tất cả
combined_train = pd.concat(train_dfs, ignore_index=True)
print(f"Total train samples: {len(combined_train)}")

# Tách theo label
train_label_0 = combined_train[combined_train['label'] == 0].values.tolist()
train_label_1 = combined_train[combined_train['label'] == 1].values.tolist()

print(f"Train - Label 0: {len(train_label_0)}, Label 1: {len(train_label_1)}")

# Xen kẽ label 0 và 1
train_interleaved = []
for item_0, item_1 in zip_longest(train_label_0, train_label_1):
    if item_0 is not None:
        train_interleaved.append(item_0)
    if item_1 is not None:
        train_interleaved.append(item_1)

# Lưu file train
train_final = pd.DataFrame(train_interleaved, columns=['domain', 'label'])
train_final.to_csv('./domain2/final_train_interleaved.csv', index=False)
print(f"Saved final_train_interleaved.csv: {len(train_final)} samples\n")


# === TỔNG hợp các file TEST ===
print("Processing TEST files...")

# Lấy tất cả file test CSV
test_files = glob.glob('./domain2/*test*.csv')
print(f"Found {len(test_files)} test files: {test_files}")

# Đọc và gộp tất cả file test
test_dfs = []
for file in test_files:
    df = pd.read_csv(file)
    test_dfs.append(df)
    print(f"Loaded {file}: {len(df)} samples")

# Gộp tất cả
combined_test = pd.concat(test_dfs, ignore_index=True)
print(f"Total test samples: {len(combined_test)}")

# Tách theo label
test_label_0 = combined_test[combined_test['label'] == 0].values.tolist()
test_label_1 = combined_test[combined_test['label'] == 1].values.tolist()

print(f"Test - Label 0: {len(test_label_0)}, Label 1: {len(test_label_1)}")

# Xen kẽ label 0 và 1
test_interleaved = []
for item_0, item_1 in zip_longest(test_label_0, test_label_1):
    if item_0 is not None:
        test_interleaved.append(item_0)
    if item_1 is not None:
        test_interleaved.append(item_1)

# Lưu file test
test_final = pd.DataFrame(test_interleaved, columns=['domain', 'label'])
test_final.to_csv('./domain2/final_test_interleaved.csv', index=False)
print(f"Saved final_test_interleaved.csv: {len(test_final)} samples")

print("\n=== SUMMARY ===")
print(f"Final train: {len(train_final)} samples")
print(f"Final test: {len(test_final)} samples")