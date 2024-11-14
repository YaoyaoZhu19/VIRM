# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None
import random
from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, l2_between_dicts, proj, Nonparametric
)
from domainbed.networks import ResNet

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
    
    
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
    
    
class BSDA(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        
        # BSDA 
        self.bsda_layer = BSDALayer(feature_dim=self.featurizer.n_outputs, 
                                    bsda_lambda=hparams['lambda'], 
                                    bsda_multi=hparams['multi'])
        self.alpha = hparams['alpha']
        self.bsda_kl_weight = hparams['kl_w']
        self.bsda_recon_weight = hparams['recon_w']
        self.hparams = hparams
        self.max_step = 100
        self.set_optimizer()

    def set_optimizer(self):
        all_parameters = list(self.featurizer.parameters()) + list(self.classifier.parameters()) + list(self.bsda_layer.parameters())
        self.optimizer = torch.optim.Adam(
            all_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    
    def update(self, minibatches, unlabeled=None, step=1):
        criterion = F.cross_entropy
        alpha = min(self.alpha * (step / (self.max_step)), self.alpha)
        
        
        # for x, y in minibatches:
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        outputs = self.predict(all_x, is_train=True)
        loss_task, loss_task_tilde, loss_bsda_kl, loss_bsda_recon = self.get_loss(outputs, all_y, criterion, is_train=True)
        loss = loss_task + alpha * (loss_task_tilde + loss_bsda_kl * self.bsda_kl_weight + loss_bsda_recon * self.bsda_recon_weight)
         
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        result = {
            'loss': loss.item(),
            'loss_task': loss_task.item(),
            'loss_task_tildes': loss_task_tilde.item(),
            'loss_bsda_kls': loss_bsda_kl.item(),
            'loss_bsda_recons': loss_bsda_recon.item(),
            'alpha': alpha,
        }
        return result
    
    def get_loss(self, outputs, y, criterion, is_train=False):
        if not is_train:
            return criterion(outputs, y)
        
        (y_hat, y_hat_tilde), (a, a_hat, mu, logvar) = outputs
        loss_task = criterion(y_hat, y)
        loss_task_tilde = criterion(y_hat_tilde, y.repeat(self.bsda_layer.multi, ))
        loss_brsda_kl = self.bsda_layer.calc_kl_loss(mu, logvar)
        loss_brsda_recon = self.bsda_layer.calc_recon_loss(a, a_hat)
        return loss_task, loss_task_tilde, loss_brsda_kl, loss_brsda_recon

    def predict(self, x, is_train=False):
        a = self.featurizer(x)
        y_hat = self.classifier(a)
        if not is_train:
            return y_hat
        
        # BSDA
        m, mu, logvar, a_hat = self.bsda_layer(a)
        a_tilde = self.bsda_layer.calc_a_tilde(a, m)
        y_hat_tilde = self.classifier(a_tilde)
        
        return (y_hat, y_hat_tilde), (a, a_hat, mu, logvar)


class BSDALayer(nn.Module):
    def __init__(self, feature_dim, bsda_lambda=0.8, bsda_multi=10) -> None:
        super().__init__()
        
        self.feature_dim = feature_dim
        self.brsda_lambda = bsda_lambda
        self.multi = bsda_multi
        
        self.logvar = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        
        self.d = nn.Dropout(p=self.brsda_lambda)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
            
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
            
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
        )
    
    def modified_indicator_function(self, x):
        return torch.where(x >= 0, torch.sign(x), -torch.sign(x))

    def calc_a_tilde(self, a, m):
        a = a.repeat(self.multi, 1)
        return a + self.d(m) * self.modified_indicator_function(a)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        std = std.repeat(self.multi, 1)
        eps = torch.randn_like(std, device=std.device)  # TODO test whether this is right
        mu = mu.repeat(self.multi, 1)
        return eps * std + mu
    
    def forward(self, a):
        """
            a: (batch_size, feature_dim)
            m: (batch_size, feature_dim)
            mu: (batch_size, feature_dim)
            logvar: (batch_size, feature_dim)
        """
        x = self.encoder(a)
        
        logvar = self.logvar(x)
        mu = torch.zeros_like(logvar, device=logvar.device)
        
        m = self.reparameterize(mu, logvar)
        a_hat = self.decoder(m)
        
        return m, mu, logvar, a_hat
    
    def calc_kl_loss(self, mu, logvar):
        # MARK mu is zeros
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss
    
    def calc_recon_loss(self, a, a_hat):
        recon_loss = torch.mean((a.repeat(self.multi, 1) - a_hat) ** 2) * 0.5
        return recon_loss


## BSDA + VREx, 只使用原始特征计算VREx
class BSDG(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        # BSDA 
        self.bsda_layer = BSDALayer(feature_dim=self.featurizer.n_outputs, 
                                    bsda_lambda=hparams['lambda'], 
                                    bsda_multi=hparams['multi'])
        self.alpha = hparams['alpha']
        self.bsda_kl_weight = hparams['kl_w']
        self.bsda_recon_weight = hparams['recon_w']
        self.hparams = hparams
        self.max_step = 1000
        self.set_optimizer()

    def set_optimizer(self):
        all_parameters = list(self.featurizer.parameters()) + list(self.classifier.parameters()) + list(self.bsda_layer.parameters())
        self.optimizer = torch.optim.Adam(
            all_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    
    def update(self, minibatches, unlabeled=None, step=1):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0
        criterion = F.cross_entropy
        alpha = min(self.alpha * (step / (self.max_step)), self.alpha)
        
        
        # for x, y in minibatches:
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        outputs = self.predict(all_x, is_train=True)
        loss_task, loss_task_tilde, loss_bsda_kl, loss_bsda_recon = self.get_loss(outputs, all_y, criterion, is_train=True)

        (all_logits, _), (_, _, _, _) = outputs
        # calculate vrex penalty
        nll = 0.
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll
        
        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss_class = mean + penalty_weight * penalty    # vrex loss
        
        loss = loss_class + alpha * (loss_task_tilde + loss_bsda_kl * self.bsda_kl_weight + loss_bsda_recon * self.bsda_recon_weight)
         
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])
        
        result = {
            'loss': loss.item(),
            'loss_task': loss_task.item(),
            'loss_task_tildes': loss_task_tilde.item(),
            'loss_bsda_kls': loss_bsda_kl.item(),
            'loss_bsda_recons': loss_bsda_recon.item(),
            'alpha': alpha,
        }
        return result
    
    def get_loss(self, outputs, y, criterion, is_train=False):
        if not is_train:
            return criterion(outputs, y)
        
        (y_hat, y_hat_tilde), (a, a_hat, mu, logvar) = outputs
        loss_task = criterion(y_hat, y)
        loss_task_tilde = criterion(y_hat_tilde, y.repeat(self.bsda_layer.multi, ))
        loss_brsda_kl = self.bsda_layer.calc_kl_loss(mu, logvar)
        loss_brsda_recon = self.bsda_layer.calc_recon_loss(a, a_hat)
        return loss_task, loss_task_tilde, loss_brsda_kl, loss_brsda_recon

    def predict(self, x, is_train=False):
        a = self.featurizer(x)
        y_hat = self.classifier(a)
        if not is_train:
            return y_hat
        
        # BSDA
        m, mu, logvar, a_hat = self.bsda_layer(a)
        a_tilde = self.bsda_layer.calc_a_tilde(a, m)
        y_hat_tilde = self.classifier(a_tilde)
        
        return (y_hat, y_hat_tilde), (a, a_hat, mu, logvar)

