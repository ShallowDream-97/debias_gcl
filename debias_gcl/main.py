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

device = 'cuda:' + args.cuda

# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
svd_q = args.q
random_seed = args.random_seed
np.random.seed(random_seed)
usr_eps_flag = args.usr_eps_flag
item_eps_flag = args.item_eps_flag
usr_loss_flag = args.usr_loss_flag
item_loss_flag = args.item_loss_flag
print(usr_eps_flag,item_eps_flag,usr_loss_flag,item_loss_flag)
# load data
path = 'data/' + args.data + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
#train_np = train.toarray()

# 计算每个用户的点击数
user_click_counts = np.array(train.sum(axis=1)).squeeze()

# 计算阈值
active_threshold = np.percentile(user_click_counts, args.active_threshold)
unactive_threshold = np.percentile(user_click_counts,  args.unactive_threshold)

# 根据点击数为每个用户分配一个类别
def assign_category(click_count):
    if click_count >= active_threshold:
        return 'active'
    elif click_count <= unactive_threshold:
        return 'unactive'
    else:
        return 'middle'

active_list = [assign_category(click_count) for click_count in user_click_counts]

# 打印前10个用户的类别，作为验证
print(active_list[:10])
train_csr = (train!=0).astype(np.float32)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
unactive_f = open(path+'unactive_test.pkl','rb')
unactive_test = pickle.load(unactive_f)
#test_np = test.toarray()



item_popularity = np.array(train.sum(axis=0)).squeeze()  # 计算每个item的出现次数

# 对items按照出现的频率进行排序，得到的索引列表
sorted_items = np.argsort(item_popularity)[::-1]

# 根据item的数量，计算三类的阈值
num_items = len(sorted_items)
pop_threshold = int(num_items * 0.2)  # 前20%为popular
middle_threshold = int(num_items * 0.4)  # 接下来的20%为middle
# 剩下的为unpopular
unpopular_items = sorted_items[int(num_items * 0.2):]
# 创建pop_list，其中包含每个item的类别
pop_list = np.full_like(item_popularity, 'unpopular', dtype=object)  # 假设 your_array 是您用来存储 item 数据的数组
pop_list[sorted_items[:pop_threshold]] = 'popular'
pop_list[sorted_items[pop_threshold:2*pop_threshold]] = 'middle'




print('Data loaded.')

print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'temp:',temp,'q:',svd_q)

epoch_user = min(train.shape[0], 30000)

adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))

print('Performing SVD...')
svd_u,s,svd_v = torch.svd_lowrank(adj,q=svd_q)
u_mul_s = svd_u @ torch.diag(s)
v_mul_s = svd_v @ torch.diag(s)
del adj
del s
print('SVD done.')

rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
print('Adj matrix normalized.')

test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')
print(len(test_labels))


unactive_test_labels = [[] for i in range(unactive_test.shape[0])]
for i in range(len(unactive_test.data)):
    row = unactive_test.row[i]
    col = unactive_test.col[i]
    unactive_test_labels[row].append(col)
print('Unactive_Test data processed.')
print(len(unactive_test_labels))

loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []

recall_20_x_unpop = []
recall_20_y_unpop = []
ndcg_20_y_unpop = []
recall_40_y_unpop = []
ndcg_40_y_unpop = []

recall_20_x_unactive = []
recall_20_y_unactive = []
ndcg_20_y_unactive = []
recall_40_y_unactive = []
ndcg_40_y_unactive = []

model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, dropout, batch_user, args.alpha,random_seed,usr_eps_flag,item_eps_flag,usr_loss_flag,item_loss_flag,pop_list,device)
#model.load_state_dict(torch.load('saved_model.pt'))
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=lambda_2,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

def learning_rate_decay(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']*0.98
        if lr > 0.0005:
            param_group['lr'] = lr
    return lr

current_lr = lr

for epoch in range(epoch_no):
    if (epoch+1)%50 == 0:
        torch.save(model.state_dict(),'saved_model_epoch_'+str(epoch)+'.pt')
        torch.save(optimizer.state_dict(),'saved_optim_epoch_'+str(epoch)+'.pt')

    current_lr = learning_rate_decay(optimizer)
        
    e_users = np.random.permutation(adj_norm.shape[0])[:epoch_user]
    batch_no = int(np.ceil(epoch_user/batch_user))

    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    for batch in tqdm(range(batch_no)):
        start = batch*batch_user
        end = min((batch+1)*batch_user,epoch_user)
        batch_users = e_users[start:end]

        # sample pos and neg
        pos = []
        neg = []
        iids = set()
        for i in range(len(batch_users)):
            u = batch_users[i]
            u_interact = train_csr[u].toarray()[0]
            positive_items = np.random.permutation(np.where(u_interact==1)[0])
            negative_items = np.random.permutation(np.where(u_interact==0)[0])
            item_num = min(max_samp,len(positive_items))
            positive_items = positive_items[:item_num]
            negative_items = negative_items[:item_num]
            pos.append(torch.LongTensor(positive_items).cuda(torch.device(device)))
            neg.append(torch.LongTensor(negative_items).cuda(torch.device(device)))
            iids = iids.union(set(positive_items))
            iids = iids.union(set(negative_items))
        iids = torch.LongTensor(list(iids)).cuda(torch.device(device))
        uids = torch.LongTensor(batch_users).cuda(torch.device(device))

        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(uids, iids, pos, neg,active_list = active_list,eps_1 = args.eps_1,eps_2 = args.eps_2,eps_3 = args.eps_3)
        loss.backward()
        optimizer.step()
        #print('batch',batch)

        torch.cuda.empty_cache()

        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

    if epoch % 3 == 0:  # test every 10 epochs
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids)/batch_user))

        all_recall_20 = 0
        all_ndcg_20 = 0
        all_recall_40 = 0
        all_ndcg_40 = 0
        all_recall_20_unpop = 0
        all_ndcg_20_unpop = 0
        all_recall_40_unpop = 0
        all_ndcg_40_unpop = 0
        for batch in tqdm(range(batch_no)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
            predictions = model(test_uids_input,None,None,None,active_list = active_list,eps_1 = args.eps_1,eps_2 = args.eps_2,eps_3 = args.eps_3 ,test=True)
            predictions = np.array(predictions.cpu())

            unpop_predictions = np.copy(predictions)

            mask = np.ones(unpop_predictions.shape, dtype=bool)
            mask[:, unpopular_items] = False
            min_value = np.iinfo(unpop_predictions.dtype).min
            unpop_predictions[mask] = min_value


            #top@20
            recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
            #top@40
            recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

            all_recall_20+=recall_20
            all_ndcg_20+=ndcg_20
            all_recall_40+=recall_40
            all_ndcg_40+=ndcg_40
            
            #top@20
            recall_20_unpop, ndcg_20_unpop = metrics(test_uids[start:end],unpop_predictions,20,test_labels)
            #top@40
            recall_40_unpop, ndcg_40_unpop = metrics(test_uids[start:end],unpop_predictions,40,test_labels)

            all_recall_20_unpop+=recall_20_unpop
            all_ndcg_20_unpop+=ndcg_20_unpop
            all_recall_40_unpop+=recall_40_unpop
            all_ndcg_40_unpop+=ndcg_40_unpop
            
            
            #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
        print('-------------------------------------------')
        print('Test of epoch',epoch,':','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)
        recall_20_x.append(epoch)
        recall_20_y.append(all_recall_20/batch_no)
        ndcg_20_y.append(all_ndcg_20/batch_no)
        recall_40_y.append(all_recall_40/batch_no)
        ndcg_40_y.append(all_ndcg_40/batch_no)
        
        print('-------------------------------------------')
        print('Test of epoch',epoch,':','Recall@20:',all_recall_20_unpop/batch_no,'Ndcg@20:',all_ndcg_20_unpop/batch_no,'Recall@40:',all_recall_40_unpop/batch_no,'Ndcg@40:',all_ndcg_40_unpop/batch_no)
        recall_20_x_unpop.append(epoch)
        recall_20_y_unpop.append(all_recall_20_unpop/batch_no)
        ndcg_20_y_unpop.append(all_ndcg_20_unpop/batch_no)
        recall_40_y_unpop.append(all_recall_40_unpop/batch_no)
        ndcg_40_y_unpop.append(all_ndcg_40_unpop/batch_no)
        
        
        unactive_test_uids = np.array([i for i, category in enumerate(active_list) if category == 'unactive'])
        unactive_batch_no = int(np.ceil(len(unactive_test_uids)/batch_user))
        unactive_recall_20 = 0
        unactive_ndcg_20 = 0
        unactive_recall_40 = 0
        unactive_ndcg_40 = 0
        for batch in range(unactive_batch_no):
            start = batch*batch_user
            end = min((batch+1)*batch_user, len(unactive_test_uids))
            
            test_uids_input = torch.LongTensor(unactive_test_uids[start:end]).cuda(torch.device(device))
            predictions = model(test_uids_input, None, None, None, active_list=active_list, eps_1=args.eps_1, eps_2=args.eps_2, eps_3=args.eps_3,  test=True)
            predictions = np.array(predictions.cpu())
            
            recall_20, ndcg_20 = metrics(unactive_test_uids[start:end], predictions, 20, test_labels)
            recall_40, ndcg_40 = metrics(unactive_test_uids[start:end], predictions, 40, test_labels)
            
            unactive_recall_20 += recall_20
            unactive_ndcg_20 += ndcg_20
            unactive_recall_40 += recall_40
            unactive_ndcg_40 += ndcg_40

        print('-------------------------------------------')
        print(' UNACTIVE test of epoch',epoch,':''Recall@20:', unactive_recall_20/unactive_batch_no, 'Ndcg@20:', unactive_ndcg_20/unactive_batch_no, 'Recall@40:', unactive_recall_40/unactive_batch_no, 'Ndcg@40:', unactive_ndcg_40/unactive_batch_no)
        recall_20_x_unactive.append(epoch)
        recall_20_y_unactive.append(unactive_recall_20/unactive_batch_no)
        ndcg_20_y_unactive.append(unactive_ndcg_20/unactive_batch_no)
        recall_40_y_unactive.append(unactive_recall_40/unactive_batch_no)
        ndcg_40_y_unactive.append(unactive_ndcg_40/unactive_batch_no)

# final test
test_uids = np.array([i for i in range(adj_norm.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_recall_20 = 0
all_ndcg_20 = 0
all_recall_40 = 0
all_ndcg_40 = 0
all_recall_20_unpop = 0
all_ndcg_20_unpop = 0
all_recall_40_unpop = 0
all_ndcg_40_unpop = 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
    predictions = model(test_uids_input,None,None,None,active_list = active_list,eps_1 = args.eps_1,eps_2 = args.eps_2,eps_3 = args.eps_3 ,test=True)
    predictions = np.array(predictions.cpu())

    unpop_predictions = np.copy(predictions)

    mask = np.ones(unpop_predictions.shape, dtype=bool)
    mask[:, unpopular_items] = False
    min_value = np.iinfo(unpop_predictions.dtype).min
    unpop_predictions[mask] = min_value

    #top@20
    recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    #top@40
    recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    all_recall_20+=recall_20
    all_ndcg_20+=ndcg_20
    all_recall_40+=recall_40
    all_ndcg_40+=ndcg_40
    
    #top@20
    recall_20_unpop, ndcg_20_unpop = metrics(test_uids[start:end],unpop_predictions,20,test_labels)
    #top@40
    recall_40_unpop, ndcg_40_unpop = metrics(test_uids[start:end],unpop_predictions,40,test_labels)

    all_recall_20_unpop+=recall_20_unpop
    all_ndcg_20_unpop+=ndcg_20_unpop
    all_recall_40_unpop+=recall_40_unpop
    all_ndcg_40_unpop+=ndcg_40_unpop
    #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
print('-------------------------------------------')
print('Final test:','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)


unactive_test_uids = np.array([i for i, category in enumerate(active_list) if category == 'unactive'])
unactive_batch_no = int(np.ceil(len(unactive_test_uids)/batch_user))

unactive_recall_20 = 0
unactive_ndcg_20 = 0
unactive_recall_40 = 0
unactive_ndcg_40 = 0
for batch in range(unactive_batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user, len(unactive_test_uids))
    
    test_uids_input = torch.LongTensor(unactive_test_uids[start:end]).cuda(torch.device(device))
    predictions = model(test_uids_input, None, None, None, active_list=active_list, eps_1=args.eps_1, eps_2=args.eps_2, eps_3=args.eps_3,  test=True)
    predictions = np.array(predictions.cpu())
    
    recall_20, ndcg_20 = metrics(unactive_test_uids[start:end], predictions, 20, test_labels)
    recall_40, ndcg_40 = metrics(unactive_test_uids[start:end], predictions, 40, test_labels)
    
    unactive_recall_20 += recall_20
    unactive_ndcg_20 += ndcg_20
    unactive_recall_40 += recall_40
    unactive_ndcg_40 += ndcg_40

print('-------------------------------------------')
print('Final test for UNACTIVE users:','Recall@20:', unactive_recall_20/unactive_batch_no, 'Ndcg@20:', unactive_ndcg_20/unactive_batch_no, 'Recall@40:', unactive_recall_40/unactive_batch_no, 'Ndcg@40:', unactive_ndcg_40/unactive_batch_no)



recall_20_x.append('Final')
recall_20_y.append(all_recall_20/batch_no)
ndcg_20_y.append(all_ndcg_20/batch_no)
recall_40_y.append(all_recall_40/batch_no)
ndcg_40_y.append(all_ndcg_40/batch_no)
recall_20_x_unpop.append('Unpop_Final')
recall_20_y_unpop.append(all_recall_20_unpop/batch_no)
ndcg_20_y_unpop.append(all_ndcg_20_unpop/batch_no)
recall_40_y_unpop.append(all_recall_40_unpop/batch_no)
ndcg_40_y_unpop.append(all_ndcg_40_unpop/batch_no)
recall_20_x_unactive.append('Unact_Final')
recall_20_y_unactive.append(unactive_recall_20/unactive_batch_no)
ndcg_20_y_unactive.append(unactive_ndcg_20/unactive_batch_no)
recall_40_y_unactive.append(unactive_recall_40/unactive_batch_no)
ndcg_40_y_unactive.append(unactive_ndcg_40/unactive_batch_no)

metric = pd.DataFrame({
    'epoch':recall_20_x,
    'recall@20':recall_20_y,
    'ndcg@20':ndcg_20_y,
    'recall@40':recall_40_y,
    'ndcg@40':ndcg_40_y,
    'un_pop_epoch':recall_20_x_unpop,
    'un_pop_recall@20':recall_20_y_unpop,
    'un_pop_ndcg@20':ndcg_20_y_unpop,
    'un_pop_recall@40':recall_40_y_unpop,
    'un_pop_ndcg@40':ndcg_40_y_unpop,
    'un_act_epoch':recall_20_x_unactive,
    'un_act_recall@20':recall_20_y_unactive,
    'un_act_ndcg@20':ndcg_20_y_unactive,
    'un_act_recall@40':recall_40_y_unactive,
    'un_act_ndcg@40':ndcg_40_y_unactive,
})



current_t = time.gmtime()
metric.to_csv('log/result_''_'+str(args.usr_eps_flag)+'_'+str(args.item_eps_flag)+'_'+str(args.usr_loss_flag)+'_'+str(args.item_loss_flag)+'_'+args.data+'_'+time.strftime('%Y-%m-%d_%H_%M_%S',current_t)+'_'+str(random_seed)+'_'+str(args.eps_1)+'_'+str(args.eps_2)+'_'+str(args.eps_3)+'_'+str(args.active_threshold)+'_'+str(args.unactive_threshold)+'_'+'.csv')

torch.save(model.state_dict(),'saved_model/saved_model_'+args.data+'_'+time.strftime('%Y-%m-%d_%H_%M_%S',current_t)+'.pt')
torch.save(optimizer.state_dict(),'saved_model/saved_optim_'+args.data+'_'+time.strftime('%Y-%m-%d_%H_%M_%S',current_t)+'.pt')
