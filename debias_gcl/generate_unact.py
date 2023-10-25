import numpy as np
import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from parser import args
from tqdm import tqdm
import time
from setproctitle import setproctitle
import os
os.environ["TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT"] = "0"

setproctitle('EXP@Xuheng')



# hyperparameters


# load data
path = 'data/' + args.data + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
#train_np = train.toarray()

# 计算每个用户的点击数
user_click_counts = np.array(train.sum(axis=1)).squeeze()

# 计算阈值
unactive_threshold = np.percentile(user_click_counts, 80)

# 根据点击数为每个用户分配一个类别
def assign_category(click_count):
    if click_count <= unactive_threshold:
        return 'unactive'
    else:
        return 'middle'

active_list = [assign_category(click_count) for click_count in user_click_counts]

# 打印前10个用户的类别，作为验证
print(active_list[:10])
train_csr = (train!=0).astype(np.float32)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
#test_np = test.toarray()
# Step 1: Identify indices of unactive users
unactive_indices = [index for index, category in enumerate(active_list) if category == 'unactive']

# Step 2: Extract rows corresponding to these indices
# unactive_test = test[unactive_indices, :]
unactive_test = test.tocsr()[unactive_indices, :]
# Step 3: Save these subsets to a .pkl file
converted_back_to_coo = unactive_test.tocoo()

with open(path+'unactive_test.pkl', 'wb') as f:
    pickle.dump(converted_back_to_coo, f)

print(f"Saved unactive data to {path}unactive_train.pkl and {path}unactive_test.pkl")



