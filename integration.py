'''Modified from Yifan Wang, Pengzhan Jin and Hehu Xie, Tensor neural network and its numerical integration. arXiv:2207.02754'''

import torch.nn as nn
import torch


# Integration operations for TNN
def normalization(w, phi):
    return torch.prod(torch.sqrt(torch.sum(w*phi**2,dim=2)),dim=0), phi / torch.sqrt(torch.sum(w*phi**2,dim=2)).unsqueeze(dim=-1)


def Int1TNN(w, alpha, phi, if_sum=True):
    """
    Integration for one TNN.

    Paramters:
        w: quad weights [N]
        alpha: scaling parameters [p]
        phi: values of TNN on quad points [dim, p, N]
    Return:
        [1] if_sum=True
        [p] if_sum=False
    """
    if if_sum:
        return torch.sum(alpha*torch.prod(torch.sum(w*phi,dim=2),dim=0))
    else:
        return alpha*torch.prod(torch.sum(w*phi,dim=2),dim=0)


def Int2TNN(w, alpha1, phi1, alpha2, phi2, if_sum=True):
    """
    Integration of prod of two TNNs (L2 inner product of two TNNs).

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [p1]
        phi1: values of TNN1 on quad points [dim, p1, N]
        alpha2: scaling parameters of TNN2 [p2]
        phi2: values of TNN2 on quad points [dim, p2, N]
    Return:
        [1] if_sum=True
        [p1,p2] if_sum=False
    """
    if if_sum:
        return torch.sum(torch.outer(alpha1,alpha2)*torch.prod((w*phi1)@phi2.transpose(1,2),dim=0))
    else:
        return torch.outer(alpha1,alpha2), (w*phi1)@phi2.transpose(1,2)


def Int2TNN_amend_1d(w1, w2, alpha1, phi1, alpha2, phi2, grad_phi1, grad_phi2, if_sum=True):
    """
    Integration of prod of two TNNs and amend each dimension respectively (H1 inner product of two TNNs).

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [p1]
        phi1: values of TNN1 on quad points [dim,p1,N]
        alpha2: scaling parameters of TNN2 [p2]
        phi2: values of TNN2 on quad points [dim,p2,N]
        grad_phi1: gradient values of TNN1 [dim,p1,N]
        grad_phi2: gradient values of TNN2 [dim,p2,N]
    Return:
        [1] if_sum=True
        [p1,p2] if_sum=False
    """
    if if_sum:
        return torch.sum(Int2TNN(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0))
    else:
        return Int2TNN(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0)



def Int3TNN(w, alpha1, phi1, alpha2, phi2, alpha3, phi3, if_sum=True):
    """
    Integration of prod of three TNNs.

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [p1]
        phi1: values of TNN1 on quad points [dim,p1,N]
        alpha2: scaling parameters of TNN2 [p2]
        phi2: values of TNN2 on quad points [dim,p2,N]
        alpha3: scaling parameters of TNN3 [p3]
        phi3: values of TNN3 on quad points [dim,p3,N]
    Return:
        [1] if_sum=True
        [p1,p2,p3] if_sum=False
    """
    if if_sum:
        return torch.sum(torch.einsum('i,j,k->ijk',alpha1,alpha2,alpha3)*torch.prod(torch.einsum('din,djn,dkn->dijk',w*phi1,phi2,phi3),dim=0))
    else:
        return torch.einsum('i,j,k->ijk',alpha1,alpha2,alpha3)*torch.prod(torch.einsum('din,djn,dkn->dijk',w*phi1,phi2,phi3),dim=0)


def Int4TNN(w, alpha1, phi1, alpha2, phi2, alpha3, phi3, alpha4, phi4, if_sum=True):
    """
    Integration of prod of four TNNs.

    Paramters:
        w: quad weights [N]
        alpha1: scaling parameters of TNN1 [p1]
        phi1: values of TNN1 on quad points [dim,p1,N]
        alpha2: scaling parameters of TNN2 [p2]
        phi2: values of TNN2 on quad points [dim,p2,N]
        alpha3: scaling parameters of TNN3 [p3]
        phi3: values of TNN3 on quad points [dim,p3,N]
        alpha4: scaling parameters of TNN4 [p4]
        phi4: values of TNN4 on quad points [dim,p4,N]
    Return:
        [1] if_sum=True
        [p1,p2,p3,p4] if_sum=False
    """
    if if_sum:
        return torch.sum(torch.einsum('i,j,k,l->ijkl',alpha1,alpha2,alpha3,alpha4)*torch.prod(torch.einsum('din,djn,dkn,dln->dijkl',w*phi1,phi2,phi3,phi4),dim=0))
    else:
        return torch.einsum('i,j,k,l->ijkl',alpha1,alpha2,alpha3,alpha4)*torch.prod(torch.einsum('din,djn,dkn,dln->dijkl',w*phi1,phi2,phi3,phi4),dim=0)


