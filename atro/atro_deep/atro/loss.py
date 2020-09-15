from collections import OrderedDict

import torch

class MaxHingeLossBinary(torch.nn.Module):
    def __init__(self, use_softplus=True):
        super(MaxHingeLossBinary,self).__init__()
        self.use_softplus = use_softplus

    def forward(self, prediction_out, target):
        """
            prediction_out (B,1): f(x)
        """
        # convert target to (-1, 1)
        target = torch.where(target==1, torch.ones_like(target), -1*torch.ones_like(target)).view(-1)
        target = target.float() # (b)

        A = 1.0 - target*prediction_out.view(-1) # (b)
        max_squared = torch.nn.Softplus()(A) if self.use_softplus else torch.clamp(A, min=0.0)   # (b)

        maxhinge_loss = max_squared.mean()

        # loss information dict
        loss_dict = OrderedDict()

        return maxhinge_loss, loss_dict

class MaxHingeLossBinaryWithRejection(torch.nn.Module):
    def __init__(self, cost:float, alpha:float=None, beta:float=None, use_softplus=True):
        super(MaxHingeLossBinaryWithRejection,self).__init__()
        assert 0 < cost <= 0.5

        self.cost = cost
        self.alpha = alpha if alpha is not None else 1.0
        self.beta = beta if beta is not None else 1.0/(1.0-2.0*cost+1e-16)
        self.use_softplus = use_softplus

        assert self.alpha > 0
        assert self.beta > 0

    def forward(self, prediction_out, rejection_out, target):
        """
        Args
        - prediction_out (B,1): f(x) in [-inf, +inf]
        - rejection_out  (B,1): r(x) in [-1, +1] 
        - target (B)
        """
        # convert target to (-1, 1)
        target = torch.where(target==1, torch.ones_like(target), -1*torch.ones_like(target)).view(-1)
        target = target.float() # (b)

        A = 1.0 + (self.alpha/2.0)*(rejection_out.view(-1) - target*prediction_out.view(-1)) # (b)
        B = self.cost*(1.0 - self.beta*rejection_out.view(-1)) # (b)
        
        if self.use_softplus:
            A_sp = torch.nn.Softplus()(A)
            B_sp = torch.nn.Softplus()(B)
            loss = torch.max(A_sp, B_sp)
        else:
            A_hinge = torch.clamp(A, min=0.0) 
            B_hinge = torch.clamp(B, min=0.0)
            loss = torch.max(A_hinge, B_hinge)

        loss = loss.mean()

        # loss information dict
        loss_dict = OrderedDict()

        return loss, loss_dict
    

# class MaxHingeLoss(torch.nn.Module):
#     def __init__(self, use_weston_mh=False):
#         super(MaxHingeLoss, self).__init__()
#         self.use_weston_mh = use_weston_mh
        
#     def forward(self, prediction_out, target, num_classes:int=10, use_weston_mh=False):
#         """
#             prediction_out (B, #class): f(x)
#         """
#         if not self.use_weston_mh:
#             # convert target to (-1, 1)
#             target = torch.nn.functional.one_hot(target, num_classes=num_classes)
#             target = torch.where(target==1, target, -1*torch.ones_like(target))
#             target = target.float()

#             A = 1.0 - target*prediction_out # (b, #class)
#             zeros = torch.zeros_like(A, dtype=torch.float32, device='cuda', requires_grad=True) # (b, #class)
#             max_squared = torch.max(A, zeros)**2 # (b, #class)
#             # max_squared = torch.max(max_squared, dim=-1)[0]  # (b)
#             max_squared = torch.sum(max_squared, dim=-1)
#         else:
#             prediction_out_y = prediction_out.gather(dim=1, index=target.view(-1,1)).view(-1,1) # (b, 1)
#             A = 1.0 - (prediction_out_y + prediction_out) # (b, #class)
#             zeros = torch.zeros_like(A, dtype=torch.float32, device='cuda', requires_grad=True) # (b, #class)
#             max_squared = torch.max(A, zeros)**2 # (b, #class)
#             max_squared = torch.sum(max_squared, dim=-1) #(b)
        
#         maxhinge_loss = max_squared.mean()

#         # loss information dict
#         loss_dict = OrderedDict()

#         return maxhinge_loss, loss_dict


class MaxHingeLoss(torch.nn.Module):
    def __init__(self, use_softplus=True):
        super(MaxHingeLoss, self).__init__()
        self.use_softplus = use_softplus
        
    def forward(self, prediction_out, target, num_classes:int=10):
        """
        Args
            prediction_out (B,#class)
            target (B)
        """
        prediction_out_t = torch.gather(prediction_out, dim=1, index=target.view(-1,1)) # (B,1)
        A = 1.0 + (prediction_out - prediction_out_t) # (B,#class)

        # create one-hot target to apply torch.masked_select.
        # using torch.where(target_onehot==1, torch.zeros_like(A), A) is not good idea for Softplus.
        B = A.size(0)
        target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes) # (B,#class)
        mask = (torch.ones_like(target_onehot)-target_onehot).bool()
        A = A.masked_select(mask=mask).view(B, num_classes-1)
        A = torch.nn.Softplus()(A) if self.use_softplus else torch.clamp(A, min=0.0) # (B,#class)

        maxhinge_loss = torch.sum(A, dim=-1) # (B)
        maxhinge_loss = maxhinge_loss.mean() # (B)

        # loss information dict
        loss_dict = OrderedDict()

        return maxhinge_loss, loss_dict

class MaxHingeLossWithRejection(torch.nn.Module):
    def __init__(self, cost:float, alpha:float=None, beta:float=None, use_softplus=True):
        super(MaxHingeLossWithRejection, self).__init__()
        assert 0 < cost < 0.5

        self.cost = cost
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 1.0/(1.0-2.0*cost)
        self.use_softplus = use_softplus

        assert self.alpha > 0
        assert self.beta > 0

    def forward(self, prediction_out, rejection_out, target, num_classes:int=10):
        """
            prediction_out (B, #class): f(x)
            rejection_out  (B, 1): r(x)
        """
        prediction_out_t = torch.gather(prediction_out, dim=1, index=target.view(-1,1)) # (B,1)
        A = 1.0 + (prediction_out - prediction_out_t) # (B,#class)



        # convert target to (-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes)
        target = torch.where(target==1, target, -1*torch.ones_like(target))
        target = target.float()

        # braod cast to (b, #class) and take mean about #class.
        A = 1.0 + (self.alpha/2.0)*(rejection_out - target*prediction_out) # (b, #class)
        A = torch.mean(A, dim=-1) # (b)
        B = self.cost*(1.0 - self.beta*rejection_out) # (b, 1)
        B = B.view(-1) # (b)

        # take max and squared
        zeros = torch.zeros_like(A, dtype=torch.float32, device='cuda', requires_grad=True)
        max_squared = torch.max(torch.max(A,B), zeros)**2 # (b)
        max_squared = torch.sum(max_squared, dim=-1) #(b)
        
        maxhinge_loss = max_squared.mean()

        # loss information dict
        loss_dict = OrderedDict()
        loss_dict['A mean'] = A.detach().mean().cpu().item()
        loss_dict['B mean'] = B.detach().mean().cpu().item()

        return maxhinge_loss, loss_dict

class WeightPenalty(torch.nn.Module):
    def __init__(self, norm='l2'):
        super(WeightPenalty, self).__init__()
        self.norm = norm

    def forward(self, model):
        
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                w = m.weight
                if self.norm == 'l2':
                    weight_loss = torch.norm(w)**2
                    break
                else:
                    raise NotImplementedError

        return weight_loss


if __name__ == '__main__':
    criterion = MaxHingeLossWithRejection(0.25)
    
    target = torch.arange(0,8).cuda()

    prediction_out = torch.randn(8,10).cuda()
    rejection_out = torch.randn(8,1).cuda()

    mh_loss, loss_dict = criterion(prediction_out, rejection_out, target)
    print(mh_loss)
    print(loss_dict)