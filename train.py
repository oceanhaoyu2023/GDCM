import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models import generator
from data_utils import MyDatasets
from torch.optim import Adam # Using standard Adam optimizer for simplicity
from torch.optim.optimizer import Optimizer, required


# In[28]:


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


# In[29]:


def schedule_sampling(batchsize,eta,aft_len,sampling,stop_it):
    if sampling==True:
        real_input_flag = np.zeros((batchsize,aft_len-1,19,119))
        return real_input_flag
    else:
        if stop_it<50000:
            eta-=0.00002
        random_flip = np.random.random_sample(
            (batchsize, aft_len - 1,180,360))
        true_token = (random_flip < eta)
        true_token[np.where(true_token==True)]=1
        true_token[np.where(true_token==False)]=0
    return eta,true_token


# In[30]:


def Norm_1_torch(y):
    sum=0
    for i in range(y.shape[0]):
        sum=sum+torch.max(torch.norm(y[i],  p=1, dim=0))
    return sum/y.shape[0]


# In[31]:


def schedule_sampling(batchsize,eta,sampling,stop_it):
    if sampling==True:
        real_input_flag = np.zeros((batchsize,7,2,180,360))
#         real_input_flag = np.pad(real_input_flag[:],((0,0),(0,0),(0,0),(6,7),(4,5)),'constant')  
        return real_input_flag
    else:
        if stop_it<50000:
            eta-=0.00002
        random_flip = np.random.random_sample(
            (batchsize,7,2,180,360))
        true_token = (random_flip < eta)
        true_token[np.where(true_token==True)]=1
        true_token[np.where(true_token==False)]=0
#         true_token = np.pad(true_token[:],((0,0),(0,0),(0,0),(6,7),(4,5)),'constant')
    return eta,true_token


# In[32]:


import random


# In[33]:


#mask data load
mask_data = np.load('/data1/表层卫星数据补全/相同时间数据/From_orbital_data/label/AMSR2_water_vapor.npy')


# In[35]:


input_month = 24
# torch.autograd.set_detect_anomaly = True
# val_data_all = torch.FloatTensor(val_data).to(device)
# val_data_all = val_data_all.unsqueeze(1)
for forecast_month in range(1):
    for t in range(5):
        loss_c = []
#             t += 2
        ite_num = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0
        epoch_num = 180
        best_loss=1000000
        eta = 1
#             weight = '../nino 盐度/weight_new/ts_2norm_onlymse_jiaocha_add_global_'+str(input_month)+'_'+str(forecast_month+1)+'_'+str(t)+'.hdf5'
        pred_model = generator(12)
#             model.load_state_dict(torch.load(weight,map_location=device),strict=False)
        pred_model = pred_model.to(device)
#         spaDri = SpaDiscriminator(1,64).to(device)
#         temDri = TemDiscriminator().to(device)
#             model = nn.DataParallel(model,device_ids=[1,0])

        l1_loss, l2_loss, bce_loss = nn.SmoothL1Loss().to(device), nn.MSELoss().to(device), nn.BCELoss().to(device)
#         lpips_dist = lpips.LPIPS(net = 'alex').to(device)
        optimizer = RAdam(pred_model.parameters(),lr=0.00003)
#         optimizer_d = RAdam(spaDri.parameters(),lr=0.00003)
#         optimizer_t = RAdam(temDri.parameters(),lr=0.00003)
        
        mse_min, psnr_max, ssim_max, lpips_min = 99999, 0, 0, 99999
#         valid_mse, valid_psnr, valid_ssim, valid_lpips = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        pred_model.train()
#         discriminator.train()
        for epochs in range(epoch_num):
            for idx, (train_input_temp,label_temp) in enumerate(train_loader):
#                 print(train_input_temp.shape)
#                 print(mask_temp.shape)
#                 train_data_all = torch.empty(batch_size_train,24,64,128)
                m_idx = random.sample(range(0,mask_data.shape[0]-7),batch_size_train)
                era5_data_new = torch.empty((batch_size_train,7,2,180,360))
                
                mask_data_temp = np.empty((batch_size_train,7,2,180,360))
                for i in range(batch_size_train):
                    mask_data_temp[i] = mask_data[m_idx[i]:m_idx[i]+7,:]
#                 mask_data_temp[1] = mask_data[m_idx[1]:m_idx[1]+7,1]
                mask_data_temp = torch.FloatTensor(mask_data_temp)
                eta,mask_real = schedule_sampling(batch_size_train,eta,False,ite_num)
                mask_real = torch.FloatTensor(mask_real)
                mask_real[torch.where(mask_data_temp==1)] = 1
                
                era5_data_new[:,:,0,:] = mask_real[:,:,0,:]*train_input_temp
                era5_data_new[:,:,1,:] = mask_real[:,:,1,:]*train_input_temp
                
                train_data_all = era5_data_new/70
                train_data_all = train_data_all[:,:,0:1,:]
                train_data_all[torch.where(torch.isnan(train_data_all))]=0
#                 train_data_all[:,:,0,:] = train_data_all[:,:,0,:]/30
#                 train_data_all[:,:,1,:] = train_data_all[:,:,1,:]/30
#                 train_data_all[:,:,2,:] = train_data_all[:,:,2,:]/10
#                 train_data_all[:,:,3,:] = train_data_all[:,:,3,:]/10
#                 print(train_data_all.shape)
                train_y = label_temp[:,:,:]            
                train_y[torch.where(torch.isnan(train_y))]=0            
                train_y = train_y/70
                train_data_all = train_data_all.float().to(device)
#                 mask_temp = mask_temp.float().to(device)

                train_y = train_y.float().to(device)
                
#                 eta,mask_real = schedule_sampling(batch_size_train,eta,False,ite_num)
#                 print('eta:',eta)
#                 mask_real = torch.FloatTensor(mask_real).to(device)
                y_pred = pred_model(train_data_all)
                imgs_gen = y_pred[:,0,:,:]
#                 d_gen = spaDri(imgs_gen.detach()[:,:,:,:])
#                 t_gen = temDri(imgs_gen.detach()[:,:,:,:])
#                 Lambda=20
#                 r_loss_sum=0
#                 for i in range(imgs_gen.shape[0]):
#                     result=torch.mul((imgs_gen[i] - train_y[i]), train_y[i])
#                     r_loss = (1 / 64 * 128 * 12) * Lambda * Norm_1_torch(result)
#                     r_loss_sum=r_loss_sum+r_loss
                
                loss_p1 = 1000*l1_loss(imgs_gen,train_y)

                optimizer.zero_grad()
                loss_p1.backward()
                optimizer.step()
                if ite_num%100==0:    
                    loss_c.append(loss_p1.item())
                if ite_num%5000==0:
                    print('loss_G:',loss_p1.item())
                    print('eta:',eta)
#                     print('loss_G_b_true:',bce_loss(d_gen, ones_label).item())
#                     print('loss_G_t_true:',bce_loss(t_gen, ones_label).item())
#                     print('loss_D:',D_loss.item())
#                     print('loss_T:',T_loss.item())
#                     print('loss_D_b_true:',bce_loss(d_gen, zeros_label).item())
#                     print('loss_D_b_false:',bce_loss(d_gt, ones_label).item())
                    print('========================================')
                        
                if ite_num%1000==0:    
                    print('************************************************')
                    torch.save(pred_model.state_dict(), './weight/generator_data_completion_'+str(input_month)+'_'+str(t)+'_water_vapour_1v_mask_lower_orbit1.hdf5')
#                     torch.save(discriminator.state_dict(), './weight/discriminator_ourmodel_frame_1v_'+str(input_month)+'_'+str(t)+'oneone.hdf5')
                ite_num+=1