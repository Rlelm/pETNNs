import torch
import torch.nn as nn
import torch.optim as optim
from quadrature import *
from integration import *
from tnn import *
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
import random
import time

pi = 3.14159265358979323846

import json
import argparse

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    
    argparser.add_argument('--dim', type=int,
                          default=10, help='[3],[10]')
    
    argparser.add_argument('--rcond', type=float,
                          default=1e-10, help='rcond')
    argparser.add_argument('--scheme',
                          default='RK4', help='[Forward_Euler],[A_Euler],[RK4]')
    argparser.add_argument('--num_update', type=int,
                          default=600, help='num of updated parameters in each subnet')
    
    argparser.add_argument('--t', type=float,
                          default=2, help='end time')
    argparser.add_argument('--dt', type=float,
                          default=5e-3, help='time step')
    

    argparser.add_argument('--device', default='cpu')                      
    argparser.add_argument('--seed', type=int,
                           default=3407, help='random seed, [3407]')   
    args = argparser.parse_args()
    print('===> Config:')
    print(json.dumps(vars(args), indent=2))
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
args = get_args()
setup_seed(args.seed)
rcond=args.rcond  
scheme=args.scheme
num_update=args.num_update
t=args.t
dt=args.dt


# ********** choose data type and device **********
dtype = torch.double
device = args.device


# ********** generate quadrature data points **********
# computation domain: [a,b]^dim
a = -1
b = 1
dim = args.dim
# quadrature rule:
# number of quad points
quad = 16
# number of partitions for [a,b]
n = 50
# quad ponits and quad weights.
x, w = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)
x_ob, w_ob= composite_quadrature_1d(quad, a, b, 200, device=device, dtype=dtype)

N, N_ob = len(x), len(x_ob)


# ********** create a neural network model **********


if dim == 2 or dim ==3:
    p = 6
    size = [2, 20, 20, p]
else: 
    p = 10
    size = [2, 30, 30, p]


activation = TNN_Tanh

model = TNN(dim,size,activation,bd=None,grad_bd=None,scaling=True).to(dtype).to(device)
num_para=len(parameters_to_vector(model.parameters()))



# ********** load trained theta0 ***********
load_name="tnn_dim={:d}.pkl".format(dim)
model.load_state_dict(torch.load(load_name))



#*********** sample parameters to be updated per time step ***********
def get_para_list():
    def get_para_id(id, i):
        nowid=0
        for ii in range(1,len(size)):
            if id<size[ii-1]*size[ii]:
                return nowid+id+size[ii-1]*size[ii]*i
            nowid+=dim*size[ii-1]*size[ii]
            id-= size[ii-1]*size[ii]
            if id < size[ii]:
                return nowid+id+size[ii]*i
            
            nowid+=dim*size[ii]
            id-=size[ii]
    
    each_list=[]
    para_list=[]
    sample_id=[]
    nowi=0
    iii=0
    for i in range(1,len(size)):
        for j in range(nowi+iii*size[i-1]*size[i],nowi+(iii+1)*size[i-1]*size[i]):
            sample_id.append(j)
        nowi=nowi+size[i-1]*size[i]
        for j in range(nowi+iii*size[i],nowi+(iii+1)*size[i]):
            sample_id.append(j)
        nowi=nowi+size[i]
        
    each_list=each_list+random.sample(sample_id,num_update)
    random.shuffle(each_list)
    for i in range(dim):
        for j in each_list:
            para_list.append(get_para_id(j,i))
    return each_list,para_list


#*********** evolution ***********
nowt=torch.tensor(0.).to(device).to(dtype)


outl=[]

time_start = time.time()  

while nowt<t:
    
    #compute L2 loss
    F_ob = torch.zeros((dim,dim,N_ob),dtype=dtype,device=device)
    for i in range(dim):
        F_ob[i,0,:] = torch.sin((x_ob+nowt)*torch.pi).to(dtype)
    alpha_F = torch.ones(dim,dtype=dtype,device=device).to(dtype)

    def loss_obs(model, w, x):
        phi = model(w,x,need_grad=0)
        alpha = model.scaling_par()

        part0=  Int2TNN(w, alpha, phi, alpha, phi)

        part1=  Int2TNN(w, alpha_F, F_ob, alpha, phi)

        part2= Int2TNN(w, alpha_F, F_ob, alpha_F, F_ob)
        
        loss= part0 - 2*part1+ part2

        return torch.sqrt(loss)
    
    loss_now=loss_obs(model,w_ob,x_ob)
    print('Time: %.3f, loss: %.6e'%(nowt, loss_now))
    outl.append(loss_now.item())


    # sample parameters to be updated
    each_list,para_list=get_para_list()


    # compute matrix JJ^t and vector Jn
    def getJ(model,w,x):
        alpha=model.scaling_par()
        phi,phi_grad= model(w,x,need_grad=1)

        phi_grad=phi_grad[:,:,:,each_list]
        grad_sum=torch.sum(phi,dim=2)
        grad_model=torch.zeros(phi.size(),device=device,dtype=dtype)
        
        for i in range(dim):
            for j in range(p):
                gradij=torch.autograd.grad(grad_sum[i][j],x,retain_graph=True)
                grad_model[i,j,0:N]=gradij[0]

        Int_outer,Int_prod=Int2TNN(w,alpha,phi,alpha,phi, if_sum=False)
        _,Int_prod_grad=Int2TNN(w,alpha,phi,alpha,grad_model, if_sum=False)
        prod=torch.prod(Int_prod,dim=0)
        
        with torch.no_grad():
            J=torch.zeros(num_update*dim,num_update*dim).to(dtype).to(device)
            Jn=torch.zeros(num_update*dim).to(dtype).to(device)
            phi_self_grad=torch.einsum('k,iakm,ibkn->iabmn',w,phi_grad,phi_grad)
            phi_all_grad=torch.einsum('k,iakm,ibk->iabm',w,phi_grad,phi)
            phi_all_grad_grad=torch.einsum('k,iakm,ibk->iabm',w,phi_grad,grad_model)
            for i1 in range(dim):
                #phi_else_i1=prod/Int_prod[i1,:,:]
                indices = [k for k in range(Int_prod.shape[0]) if (k != i1)]
                phi_else_i1=torch.prod(Int_prod[indices],dim=0)
                J[i1*num_update:(i1+1)*num_update,i1*num_update:(i1+1)*num_update]=torch.einsum('pq,pqmn,pq->mn',Int_outer,phi_self_grad[i1,:,:,:,:],phi_else_i1)
                for i2 in range(i1):
                    phi_else_i1i2=phi_else_i1/Int_prod[i2,:,:]
                    J[i1*num_update:(i1+1)*num_update,i2*num_update:(i2+1)*num_update]=torch.einsum('pq,pq,pqm,pqn->mn',Int_outer,phi_else_i1i2,
                                                                                                    phi_all_grad[i1,:,:,:],phi_all_grad[i2,:,:,:].transpose(1,0))
                    J[i2*num_update:(i2+1)*num_update,i1*num_update:(i1+1)*num_update]=J[i1*num_update:(i1+1)*num_update,
                                                                                         i2*num_update:(i2+1)*num_update].transpose(1,0).detach().clone()
                for i2 in range(dim):
                    if i1==i2:
                        Jn[i1*num_update:(i1+1)*num_update]+=torch.einsum('pqm,pq,pq->m',phi_all_grad_grad[i1,:,:,:],Int_outer,phi_else_i1)
                    else:
                        phi_else_i1i2=phi_else_i1/Int_prod[i2,:,:]
                        Jn[i1*num_update:(i1+1)*num_update]+=torch.einsum('pq,pq,pq,pqm->m',Int_outer,phi_else_i1i2,Int_prod_grad[i2,:,:],phi_all_grad[i1,:,:,:])

        return J,Jn

    
    # compute d(theta)/dt
    def update(rcond=1e-10):
        J,Jn=getJ(model,w,x)
        #J=(J+J.transpose(1,0))/2
        J,Jn=J.detach().cpu().numpy(),Jn.detach().cpu().numpy()
        ll=np.linalg.lstsq(J,Jn, rcond=rcond)[0]
        l1=torch.zeros(num_para,device=device,dtype=dtype)
        
        for id1 in range(num_update*dim):
            i=para_list[id1]
            l1[i]=torch.tensor(ll[id1],device=device,dtype=dtype)
        
        return l1
    

    # different marching schemes
    if scheme=='Forward_Euler':
        l=parameters_to_vector(model.parameters()).clone().detach()
        l1=update(rcond=rcond)
        
        nowt=nowt+dt
        nowl=l+l1*dt
        vector_to_parameters(nowl,model.parameters())

    if scheme=='A_Euler':
        l=parameters_to_vector(model.parameters()).clone().detach()
        l1=update(rcond=rcond)
        
        nowl=l+l1*dt
        nowt=nowt+dt
        vector_to_parameters(nowl,model.parameters())

        l2=update(rcond=rcond)
        
        nowl=l+(l1+l2)*dt/2
        vector_to_parameters(nowl,model.parameters())
    if scheme == 'RK4':

        l=parameters_to_vector(model.parameters()).clone().detach()
        l1=update(rcond=rcond)
        
        nowl=l+l1*dt/2
        nowt=nowt+dt/2
        vector_to_parameters(nowl,model.parameters())

        l2=update(rcond=rcond)

        nowl=l+l2*dt/2
        vector_to_parameters(nowl,model.parameters())

        l3=update(rcond=rcond)

        nowl=l+l3*dt
        nowt=nowt+dt/2
        vector_to_parameters(nowl,model.parameters())

        l4=update(rcond=rcond)

        nowl=l+(l1+2*l2+2*l3+l4)/6*dt
        vector_to_parameters(nowl,model.parameters())

    time_end=time.time()
    
    
    
print('Runtime: %.3f'%(time_end-time_start))
    
file_name="t={:.2f}_{}_num_update={:d}_rcond={:.0e}_seed={:d}.txt".format(t,scheme,num_update,rcond,args.seed)
out_l=np.array(outl)
np.savetxt(file_name,out_l,fmt='%.6e')    
            
file_name="t={:.2f}_{}_num_update={:d}_rcond={:.0e}_seed={:d}.pkl".format(t,scheme,num_update,rcond,args.seed)            
torch.save(model.state_dict(), file_name)
