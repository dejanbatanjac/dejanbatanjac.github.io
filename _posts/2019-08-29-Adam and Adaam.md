---
published: true
layout: post
title: Adam and Adaam
---

I wanted to outline that the actual Adam implmentation in PyTorch is little bit different than the [original Adam paper](https://arxiv.org/abs/1412.6980).

The paper suggests Adam should be implemented like this:

>$ p_{t+1} = p_t - lr \frac {m_t} { \sqrt{v_t} + \epsilon} $,
>
>where
>
>$ m_t = \frac {\beta_1 m_{t-1}+ (1-\beta_1)grad_{t-1}} {1-\beta_1^t}$, 
>$ v_t = \frac {\beta_2 v_{t-1}+ (1-\beta_2)grad_{t-1}^2} {1-\beta_2^t}$
>and $lr$ is the learning rate
>
>$grad_t$ is gradient tensor, 
>
>$grad_t^2$ is Hadamar product of gradient tensor
>
>$m_0$ and $v_0$ are 0,
>
>$\beta_1,\beta_2$ are usually 0.9 and 0.99,
>
>and $\epsilon$ is some small number `1e-3` for instance.

However, if we convert the [Adam optimizer from PyTorch](https://pytorch.org/docs/stable/_modules/torch/optim/adam.html) you will note that the implementation is not by the paper.

I will present you the simplified Adam based on the PyTorch implementation with weight decay and other unneeded stuff removed:

    class Adam(Optimizer): #simplified but like in PyTorch

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3):
            
            defaults = dict(lr=lr, betas=betas, eps=eps)
            super(Adam, self).__init__(params, defaults)

        def __setstate__(self, state):
            super(Adam, self).__setstate__(state)            

        def step(self, closure=None):        
            
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data                
                    state = self.state[p]
                    
                    if len(state) == 0:
                        state['step'] = 0                    
                        state['agrad'] = torch.zeros_like(p.data)                    
                        state['agrad2'] = torch.zeros_like(p.data)
                        
                    state['step'] += 1    

                    agrad, agrad2 = state['agrad'], state['agrad2']                
                    beta1, beta2 = group['betas']
                
                    agrad.mul_(beta1).add_(1 - beta1, grad)
                    agrad2.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    denom = agrad2.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(-step_size, agrad, denom)

            return loss

In here `agrad` and `agrad2` are average gradients calculated and also the so called `step_size` is calculated as this:

    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

The previous is based on the fact that $\epsilon$ is very small. If we plan to use bigger $\epsilon$ this would not be correct. 

I would present Adaam, the real Adam by the paper:

    class Adaam(Optimizer):

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3):
            
            defaults = dict(lr=lr, betas=betas, eps=eps)
            super(Adaam, self).__init__(params, defaults)

        def __setstate__(self, state):
            super(Adaam, self).__setstate__(state)

        def step(self, closure=None):        
            
            for group in self.param_groups:
                for p in group['params']:                
                    if p.grad is None:
                        continue
                    grad = p.grad.data                
                    state = self.state[p] 
                    
                    if len(state) == 0:
                        state['step'] = 0                    
                        state['agrad'] = torch.zeros_like(p.data) # grad average                
                        state['agrad2'] = torch.zeros_like(p.data) # Hadamar grad average
                        
                    state['step'] += 1
                    
                    agrad, agrad2 = state['agrad'], state['agrad2'] 
                    beta1, beta2 = group['betas']
                    
                    agrad.mul_(beta1).add_(1 - beta1, grad)
                    agrad2.mul_(beta2).addcmul_(1 - beta2, grad, grad) 

                    bias_1 = 1 - beta1 ** state['step']
                    bias_2 = 1 - beta2 ** state['step'] 
                    
                    agrad = agrad.div(bias_1)
                    agrad2 = agrad2.div(bias_2)
                    
                    denom = agrad2.sqrt().add_(group['eps'])
                
                    p.data.addcdiv_(-group['lr'], agrad, denom)

            return loss

Hope you will find this original Adaam useful.







