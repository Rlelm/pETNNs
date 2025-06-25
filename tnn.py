'''Modified from Yifan Wang, Pengzhan Jin and Hehu Xie, Tensor neural network and its numerical integration. arXiv:2207.02754'''
import torch.nn as nn
import torch
import numpy as np

from quadrature import *

from torch.nn.utils import parameters_to_vector, vector_to_parameters

# ********** Activation functions with Derivative **********
# Redefine activation functions and the corresponding local gradient

# tanh(x)
class TNN_Tanh(nn.Module):
    """Tnn_Tanh"""
    def forward(self,x):
        return torch.tanh(x)

    def grad(self,x):
        return 1-torch.tanh(x)**2

# sigmoid(x)
class TNN_Sigmoid(nn.Module):
    """TNN_Sigmoid"""
    def forward(self,x):
        return torch.sigmoid(x)

    def grad(self,x):
        return torch.sigmoid(x)*(1-torch.sigmoid(x))

# sin(x)
class TNN_Sin(nn.Module):
    """TNN_Sin"""
    def forward(self,x):
        return torch.sin(x)

    def grad(self,x):
        return torch.cos(x)

# cos(x)
class TNN_Cos(nn.Module):
    """for TNN_Sin"""
    def forward(self,x):
        return torch.cos(x)

    def grad(self,x):
        return -torch.sin(x)


# ReQU(x)=
#         x^2, x\geq0,
#         0,   x<0.
class TNN_ReQU(nn.Module):
    """docstring for TNN_ReQU"""
    def forward(self,x):
        return x*torch.relu(x)

    def grad(self,x):
        return 2*torch.relu(x)
        




# ********** Network layers **********
# Linear layer for TNN
class TNN_Linear(nn.Module):
    """
    Applies a batch linear transformation to the incoming data:
        input data: x:[dim, n1, N]
        learnable parameters: W:[dim,n2,n1], b:[dim,n2,1]
        output data: y=Wx+b:[dim,n2,N]

    Parameters:
        dim: dimension of TNN
        out_features: n2
        in_features: n1
        bias: if bias needed or not (boolean)
    """
    def __init__(self, dim, out_features, in_features, bias):
        super(TNN_Linear, self).__init__()
        self.dim = dim
        self.out_features = out_features
        self.in_features = in_features

        self.weight = nn.Parameter(torch.empty((self.dim, self.out_features, self.in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((self.dim, self.out_features, 1)))
        else:
            self.bias = None

    def forward(self,x):
        if self.bias==None:
            if self.in_features==1:
                return self.weight*x
            else:
                return self.weight@x
        else:
            if self.in_features==1:
                y=self.weight*x+self.bias
                return y
            else:
                return self.weight@x+self.bias

    def extra_repr(self):
        return 'weight.size={}, bias.size={}'.format(
            [self.dim, self.out_features, self.in_features], [self.dim, self.out_features, 1] if self.bias!=None else []
        )


# Scaling layer for TNN.
class TNN_Scaling(nn.Module):
    """
    Define the scaling parameters.

    size:
        [k,p] for Multi-TNN
        [p] for TNN
    """
    def __init__(self, size):
        super(TNN_Scaling, self).__init__()
        self.size = size
        self.alpha = nn.Parameter(torch.empty(self.size))

    def extra_repr(self):
        return 'size={}'.format(self.size)


# Define extra parameters
class TNN_Extra(nn.Module):
    """
    Define extra parameters.
    """
    def __init__(self, size):
        super(TNN_Extra, self).__init__()
        self.size = size
        self.beta = nn.Parameter(torch.empty(self.size))
        
    def extra_repr(self):
        return 'size={}'.format(self.size)


# ********** TNN architectures **********
# One simple TNN
class TNN(nn.Module):
    """
    Architectures of the simple tensor neural network.
    FNN on each demension has the same size,
    and the input integration points are same in different dinension. 
    TNN values and gradient values at data points are provided.

    Parameters:
        dim: dimension of TNN, number of FNNs
        size: [1, n0, n1, ..., nl, p], size of each FNN
        activation: activation function used in hidden layers
        bd: extra function for boundary condition
        grad_bd: gradient of bd
        initializer: initial method for learnable parameters
    """
    def __init__(self, dim, size, activation, bd=None, grad_bd=None, scaling=True, extra_size=False, initializer='default'):
        super(TNN, self).__init__()
        self.dim = dim
        self.size = size
        self.activation = activation()
        self.bd = bd
        self.grad_bd = grad_bd
        self.scaling = scaling
        self.extra_size = extra_size

        self.p = abs(self.size[-1])

        self.ms = self.__init_modules()
        self.__initialize()

    # Register learnable parameters of TNN module.
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(1, len(self.size)):
            bias = True if self.size[i] > 0 else False
            modules['TNN_Linear{}'.format(i-1)] = TNN_Linear(self.dim,abs(self.size[i]),abs(self.size[i-1]),bias)
        if self.scaling:
            modules['TNN_Scaling'] = TNN_Scaling([self.p])
        if self.extra_size:
            modules['TNN_Extra'] = TNN_Extra(self.extra_size)
        return modules

    # Initialize learnable parameters of TNN module.
    def __initialize(self):
        for i in range(1, len(self.size)):
            for j in range(self.dim):
                #nn.init.orthogonal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:])
                nn.init.xavier_normal_(self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:])
                with torch.no_grad():
                    self.ms['TNN_Linear{}'.format(i-1)].weight[j,:,:]*=self.size[-1]**(1./(len(self.size)))
            if self.size[i] > 0:
                nn.init.constant_(self.ms['TNN_Linear{}'.format(i-1)].bias, 0)
        if self.scaling:
            nn.init.constant_(self.ms['TNN_Scaling'].alpha, 1)
        if self.extra_size:
            nn.init.constant_(self.ms['TNN_Extra'].beta, 1)

    # function to return scaling parameters
    def scaling_par(self):
        if self.scaling:
            return self.ms['TNN_Scaling'].alpha
        else:
            raise NameError('The TNN Module does not have Scaling Parameters')

    # function to return extra parameters
    def extra_par(self):
        if self.extra_size:
            return self.ms['TNN_Extra'].beta
        else:
            raise NameError('The TNN Module does not have Extra Parameters')


    def forward(self,w,x,need_grad=0):
        """
        Parameters:
            w: quadrature weights [N]
            x: quadrature points [N]
            need_grad: if return gradient or not
        
        Returns:
            phi: values of each dimensional FNN [dim, p, N]
            grad_model: gradients (w.r.t. parameters) of each FNN output [dim, p, N, M]
        """
        # Compute values of each one-dimensional input FNN at each quadrature point.
        if need_grad==0:

            # Embedding
            x= torch.cat( [ (torch.cos(x*torch.pi)).unsqueeze(0), (torch.sin(x*torch.pi)).unsqueeze(0)] , 0 )

            # Forward process.
            for i in range(1, len(self.size) - 1):
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                x = self.activation(x)
            phi = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            return phi


        # Compute values and gradient values of each FNN output at each quadrature point simutaneously.
        if need_grad==1:

            # Compute forward and backward process .
                
            # Embedding
            x= torch.cat( [ (torch.cos(x*torch.pi)).unsqueeze(0), (torch.sin(x*torch.pi)).unsqueeze(0)] , 0 )

            list_x=[]
            list_x.append(x.detach().clone().expand(self.dim,x.shape[0],x.shape[1]))
            num=0

            for i in range(1, len(self.size) - 1):
                num+=self.size[i]*(self.size[i-1]+1)
                x = self.ms['TNN_Linear{}'.format(i-1)](x)
                list_x.append(x.clone().detach())
                x = self.activation(x)
                
            x = self.ms['TNN_Linear{}'.format(len(self.size) - 2)](x)
            i+=1
            num+=self.size[i]*(self.size[i-1]+1)

            phi = x
                
            grad_model=torch.zeros(phi.size()+torch.Size([num]),device=x.device,dtype=x.dtype)
            dz=torch.eye(self.size[-1],device=x.device,dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            dz=dz.expand(x.shape[0],x.shape[2],x.shape[1],x.shape[1])

            for i in range(len(self.size)-2,-1,-1):
                grad_model[:,:,:,num-self.size[i+1]:num]=dz.detach().clone().transpose(1,2)
                
                num=num-self.size[i+1]
                if i==0:
                    grad_model[:,:,:,num-self.size[i+1]*self.size[i]:num]=\
                    torch.einsum('iak,ikpb->ipkba',list_x[i],dz).reshape(x.shape[0],self.size[-1],x.shape[2],-1)
                else:
                    grad_model[:,:,:,num-self.size[i+1]*self.size[i]:num]=\
                        torch.einsum('iak,ikpb->ipkba',self.activation(list_x[i]),dz).reshape(x.shape[0],self.size[-1],x.shape[2],-1)
                
                num=num-self.size[i+1]*self.size[i]
                dz=torch.einsum('iba,ikpb->ikpa',self.ms['TNN_Linear{}'.format(i)].weight,dz)
                dz=torch.einsum('ijk,ikpj->ikpj',self.activation.grad(list_x[i]),dz)

            return phi,grad_model

    def extra_repr(self):
        return '{}\n{}'.format('Architectures of one TNN(dim={},rank={}) which has {} FNNs:'.format(self.dim,self.p,self.dim),\
                'Each FNN has size: {}'.format(self.size))

