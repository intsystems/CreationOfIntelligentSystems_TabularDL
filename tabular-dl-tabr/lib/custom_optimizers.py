from torch import optim 
import numpy as np
import torch


class MirrorGD(optim.Optimizer):
    def __init__(self, params, lr = 0.01, mirror = lambda x,g,R: x , scale = 0.1):
        defaults = dict(lr = lr, R = None )
        self.mirror = mirror
        self.scale = scale
        self.iter = 0
        super(MirrorGD, self).__init__(params,defaults)
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        if self.iter == 0 :
            for group in self.param_groups:
                norms = []
                for p in group['params']:
                    norm = p.data.norm(p=1) # Говорим, что симплекс будет размера, как первая норма изначальных весов на scale
                    norms.append(norm * self.scale)
                    p.data[p.data < 0] = 0 # Ищбавляемся от отрицательных весов 
                    p.data = norm * p.data / p.data.norm(p=1) # Нормируем , приводя к симплексу
                group['R'] = norms                   
                print('Group R:', group['R'])
        
        self.iter +=1

        for group in self.param_groups:
            R_arr = group['R']
            for e, p in enumerate(group['params']):
                # print(p.data , p.data.shape)
                if p.grad is None:
                    continue
                # Take FW direction
                lr = group['lr']
                R = R_arr[e]

                p.data = self.mirror( p.data , lr * p.grad , R)
                 
        return loss


class ProjectionGD(optim.Optimizer):
    def __init__(self, params, lr = 0.01, proj = lambda x,R: x , scale = 0.1):
        defaults = dict(lr = lr, R = None )
        self.proj = proj
        self.scale = scale
        self.iter = 0
        super(ProjectionGD, self).__init__(params,defaults)
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        if self.iter == 0 :
            for group in self.param_groups:
                norms = []
                for p in group['params']:
                    norm = p.data.norm(p=2)
                    norms.append(norm * self.scale)
                group['R'] = norms                   
                print('Group R:', group['R'])
        
        self.iter +=1

        for group in self.param_groups:
            R_arr = group['R']
            for e, p in enumerate(group['params']):
                # print(p.data , p.data.shape)
                if p.grad is None:
                    continue
                # Take FW direction
                lr = group['lr']
                R = R_arr[e]

                p.data = self.proj( p.data - lr * p.grad , R)
                 
        return loss


class FrankWolfe(optim.Optimizer):
    def __init__(self, params, lr = 0.01, direction="vanila", lmo = lambda x,R: x , scale = 0.1):
        defaults = dict(lr = lr, R = None )
        self.lmo = lmo
        self.direction = direction
        self.iter = 0
        self.scale = scale
        super(FrankWolfe, self).__init__(params,defaults)
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        if self.iter == 0 :
            for group in self.param_groups:
                max_norm = 0
                for p in group['params']:
                    max_norm = max(p.data.norm(p=2), max_norm)
                    # print('New')
                group['R'] = max_norm * self.scale                    
                print('Group R:', group['R'])

        self.iter += 1
        # print(self.iter)
        
        for group in self.param_groups:
            for p in group['params']:
                # print(p.data , p.data.shape)
                if p.grad is None:
                    continue
                # Take FW direction
                lr = group['lr']
                R = group['R']


                # print('s_k_FW:', s_k_FW)
                s_k_FW = self.lmo(p.grad, R)

                if self.direction == 'linesearch':
                    pass 
                elif self.direction == 'vanila':
                    gamma = 2/(self.iter + 2)
                elif self.direction == 'vanila_lr':
                    gamma = 2/(self.iter + 2) * lr
                elif self.direction == 'lr':
                    gamma = lr
                
                # print('Params before step:', p.data)
                p.data += gamma * (s_k_FW - p.data)
                # print('Params after step:', p.data)
                
        return loss



class LBFGS(optim.Optimizer):
    def __init__(self, params , linesearch = False , history = 40 , lr=1e-5):
        defaults = dict(lr=lr)
        super(LBFGS, self).__init__(params, defaults)
        ## One group
        for group in self.param_group:
            p = group['params']        
            self.optimizer = torch.optim.LBFGS(p, lr=lr , history_size= history, line_search_fn=linesearch)
        

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # take sign of gradient
                grad = torch.sign(p.grad)
                # randomise zero gradients to ±1
                if self.rand_zero:
                    grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
                    assert not (grad==0).any()
                # make update
                p.data -= group['lr'] * grad
        return loss


class signSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, rand_zero=True):
        defaults = dict(lr=lr)
        self.rand_zero = rand_zero
        super(signSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                
                if p.grad is None:
                    continue
                # take sign of gradient
                grad = torch.sign(p.grad)
                # randomise zero gradients to ±1
                if self.rand_zero:
                    grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
                    assert not (grad==0).any()
                # make update
                p.data -= group['lr'] * grad
        return loss
