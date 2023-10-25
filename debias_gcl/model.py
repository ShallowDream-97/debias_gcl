import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F
import numpy as np
import random
class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, dropout, batch_user, alpha,random_seed,usr_eps_flag,item_eps_flag,usr_loss_flag,item_loss_flag,pop_list,device):
        super(LightGCL,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.temp = temp
        self.lambda_1 = lambda_1
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user
        self.Ws = nn.ModuleList([W_contrastive(d) for i in range(l)])

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt
        self.alpha =alpha
        self.random_seed = random_seed
        self.usr_eps_flag = usr_eps_flag
        self.item_eps_flag = item_eps_flag
        self.usr_loss_flag = usr_loss_flag
        self.item_loss_flag = item_loss_flag
        self.pop_list = pop_list
        self.device = device
        self.set_seeds(self.random_seed) 
    
    def set_seeds(self, seed):
        print(self.usr_eps_flag,self.item_eps_flag,self.usr_loss_flag,self.item_loss_flag)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, uids, iids, pos, neg, active_list,eps_1,eps_2,eps_3,test=False):
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask)
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                self.Z_u_list[layer] = self.act(spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1], self.device))
                self.Z_i_list[layer] = self.act(spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1], self.device))
                
                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = self.act(self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = self.act(self.v_mul_s @ ut_eu)

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer] + self.E_u_list[layer-1]
                self.E_i_list[layer] = self.Z_i_list[layer] + self.E_i_list[layer-1]

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            loss_s = 0
            for l in range(1,self.l+1):
                u_mask = (torch.rand(len(uids))>0.5).float().cuda(self.device)

                gnn_u = nn.functional.normalize(self.Z_u_list[l][uids],p=2,dim=1)
                hyper_u = nn.functional.normalize(self.G_u_list[l][uids],p=2,dim=1)
                hyper_u = self.Ws[l-1](hyper_u)
                pos_score = torch.exp((gnn_u*hyper_u).sum(1)/self.temp)
                neg_score = torch.exp(gnn_u @ hyper_u.T/self.temp).sum(1)
                loss_s_u = ((-1 * torch.log(pos_score/(neg_score+1e-8) + 1e-8))*u_mask).sum()
                loss_s = loss_s + loss_s_u

                i_mask = (torch.rand(len(iids))>0.5).float().cuda(self.device)

                gnn_i = nn.functional.normalize(self.Z_i_list[l][iids],p=2,dim=1)
                hyper_i = nn.functional.normalize(self.G_i_list[l][iids],p=2,dim=1)
                hyper_i = self.Ws[l-1](hyper_i)
                pos_score = torch.exp((gnn_i*hyper_i).sum(1)/self.temp)
                neg_score = torch.exp(gnn_i @ hyper_i.T/self.temp).sum(1)
                loss_s_i = ((-1 * torch.log(pos_score/(neg_score+1e-8) + 1e-8))*i_mask).sum()
                loss_s = loss_s + loss_s_i
            
            # bpr loss
            loss_r = 0
            # for i in range(len(uids)):
            #     u = uids[i]
            #     u_emb = self.E_u[u]
            #     u_pos = pos[i]
            #     u_neg = neg[i]
            #     pos_emb = self.E_i[u_pos]
            #     neg_emb = self.E_i[u_neg]
            #     pos_scores = u_emb @ pos_emb.T
            #     neg_scores = u_emb @ neg_emb.T
            #     bpr = nn.functional.relu(1-pos_scores+neg_scores)
            #     loss_r = loss_r + bpr.sum()
            # loss_r = loss_r/self.batch_user
            
            # 定义噪声级别

            if self.usr_eps_flag:
                loss_r_per_category = {'active': 0.0, 'middle': 0.0, 'unactive': 0.0}
                count_per_category = {'active': 0, 'middle': 0, 'unactive': 0}  # 追踪每种类别的用户数量
                noise_levels = {
                    'active': eps_1,  # example value, adjust as needed
                    'middle': eps_2,  # example value, adjust as needed
                    'unactive': eps_3  # example value, adjust as needed
                    }
                
            if self.item_loss_flag:
                            # 初始化三种类别的loss
                loss_pop = 0
                loss_mid = 0
                loss_unpop = 0
                
                count_pop = 0 
                count_mid = 0
                count_unpop = 0
            
            for i in range(len(uids)):
                u = uids[i]
                u_emb = self.E_u[u]
                if self.usr_eps_flag:
                # 根据用户的类别添加噪声
                    user_category = active_list[u.item()]  # assuming active_list is accessible here
                    random_noise = torch.rand_like(u_emb, device=u_emb.device)
                    u_emb = u_emb + torch.sign(u_emb) * F.normalize(random_noise, dim=-1) * noise_levels[user_category]
                u_pos = pos[i]
                u_neg = neg[i]
                
                if self.item_eps_flag:
                    for p_item, n_item in zip(u_pos, u_neg):
                        pos_emb = self.E_i[p_item]
                        neg_emb = self.E_i[n_item]
                        
                        item_category = self.pop_list[p_item.cpu().numpy()] 
                        if item_category == 'popular':
                            random_noise = torch.rand_like(pos_emb, device=u_emb.device)
                            pos_emb = pos_emb + torch.sign(pos_emb) * F.normalize(random_noise, dim=-1) * eps_1         
                            random_noise = torch.rand_like(pos_emb, device=u_emb.device)
                            neg_emb = neg_emb + torch.sign(neg_emb) * F.normalize(random_noise, dim=-1) * eps_1
                        elif item_category == 'middle':
                            random_noise = torch.rand_like(pos_emb, device=u_emb.device)
                            pos_emb = pos_emb + torch.sign(pos_emb) * F.normalize(random_noise, dim=-1) * eps_2         
                            random_noise = torch.rand_like(pos_emb, device=u_emb.device)
                            neg_emb = neg_emb + torch.sign(neg_emb) * F.normalize(random_noise, dim=-1) * eps_2
                        else:  # 'unpopular'
                            random_noise = torch.rand_like(pos_emb, device=u_emb.device)
                            pos_emb = pos_emb + torch.sign(pos_emb) * F.normalize(random_noise, dim=-1) * eps_3         
                            random_noise = torch.rand_like(pos_emb, device=u_emb.device)
                            neg_emb = neg_emb + torch.sign(neg_emb) * F.normalize(random_noise, dim=-1) * eps_3
                
                pos_emb = self.E_i[u_pos]
                neg_emb = self.E_i[u_neg]
                pos_scores = u_emb @ pos_emb.T
                neg_scores = u_emb @ neg_emb.T
                bpr = nn.functional.relu(1-pos_scores+neg_scores)
                # 根据用户的类别分别+loss
                if self.usr_loss_flag:
                    loss_r_per_category[user_category] += bpr.sum()
                    count_per_category[user_category] += 1  
                    
                elif self.item_loss_flag:
                    item_category = self.pop_list[p_item.cpu().numpy()] 
                    if item_category == 'popular':
                        loss_pop += bpr
                        count_pop += 1
                    elif item_category == 'middle':
                        loss_mid += bpr
                        count_mid += 1
                    else:  # 'unpopular'
                        loss_unpop += bpr
                        count_unpop += 1   
                loss_r = loss_r + bpr.sum()
                
            if self.usr_loss_flag:    
                for category in loss_r_per_category:
                    if count_per_category[category] > 0:
                        loss_r_per_category[category] /= count_per_category[category]
                loss_r_category = sum(loss_r_per_category.values())
                loss_r = loss_r/self.batch_user
            
            # total loss
                loss_ori = loss_r + self.lambda_1 * loss_s
                loss = self.alpha*loss_r_category + self.lambda_1 * loss_s
                print("loss_r_category:",loss)
                print("Loss_ori:",loss_ori)
                return loss, loss_r, loss_s
            
            
            elif self.item_loss_flag:
                loss_r_pop = loss_pop/count_pop
                loss_r_mid = loss_mid/count_mid
                loss_r_unpop = loss_unpop/count_unpop
                loss_item = loss_r_pop + loss_r_mid + loss_r_unpop
                return loss_item,loss_r_pop,loss_s
                
            

class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d,d)))

    def forward(self,x):
        return x @ self.W