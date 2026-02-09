#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class _Loss(nn.Module):
    def __init__(self, reduction='sum'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. 
        `none`: no reduction will be applied, 
        `mean`: the sum of the output will be divided by the number of elements in the output, 
        `sum`: the output will be summed. 

        Note: size_average and reduce are in the process of being deprecated, 
        and in the meantime,  specifying either of those two args will override reduction. 
        Default: `sum`
        '''
        super().__init__()
        assert(reduction == 'mean' or reduction ==
               'sum' or reduction == 'none')
        self.reduction = reduction

class BPRLoss(_Loss):
    def __init__(self, reduction='sum'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        # ensure reduction in (mean，sum, none)
        super().__init__(reduction)

    def forward(self, model_output, **kwargs):
        '''
        `model_output` (tensor) - column 0 must be the scores of positive bundles/items, column 1 must be the negative.
        '''
        pred, L2_loss, *rest = model_output

        # 加一个resn loss的情况
        if len(rest) == 0:  # BGCN original的情况
            resn_loss = 0
            use_pda = False
            weights = None
        elif len(rest) == 1:  # 加一个resn loss的情况
            resn_loss = rest[0]
            use_pda = False
            weights = None
        elif len(rest) == 2:  # 在bpr loss上reweight pda的情况
            resn_loss = 0
            use_pda, weights = rest[0], rest[1]
        else:
            raise ValueError("rest must be 0 | 1 | 2")

        # BPR loss
        loss = -torch.log(torch.sigmoid(pred[:, 0] - pred[:, 1]))

        # PDA, reweight loss
        if use_pda:
            loss = loss * weights

        # reduction
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        loss += L2_loss / kwargs['batch_size'] if 'batch_size' in kwargs else 0
        loss += resn_loss

        return loss

