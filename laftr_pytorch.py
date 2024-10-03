import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# Constants
EPS = 1e-8
HIDDEN_LAYER_SPECS = {
    'enc': [],
    'cla': [],
    'rec': [],
    'aud': [5],
}
CLASS_COEFF = 1.
FAIR_COEFF = 0.
RECON_COEFF = 0.
XDIM = 61
YDIM = 1
ZDIM = 10
ADIM = 1
A_WTS = [1., 1.]
Y_WTS = [1., 1.]
AY_WTS = [[1., 1.], [1., 1.]]
SEED = 0
ACTIV = 'leakyrelu'
HINGE = 0.

class MLP(nn.Module):
    def __init__(self, shapes, activ='leakyrelu'):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(shapes) - 1):
            self.layers.append(nn.Linear(shapes[i], shapes[i+1]))
        
        if activ == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError(f"Activation {activ} not implemented")

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

class AbstractBaseNet(nn.Module, ABC):
    def __init__(self,
                 recon_coeff=RECON_COEFF,
                 class_coeff=CLASS_COEFF,
                 fair_coeff=FAIR_COEFF,
                 xdim=XDIM,
                 ydim=YDIM,
                 zdim=ZDIM,
                 adim=ADIM,
                 hidden_layer_specs=HIDDEN_LAYER_SPECS,
                 seed=SEED,
                 hinge=HINGE,
                 **kwargs):
        super(AbstractBaseNet, self).__init__()
        self.recon_coeff = recon_coeff
        self.class_coeff = class_coeff
        self.fair_coeff = fair_coeff
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.adim = adim
        self.hidden_layer_specs = hidden_layer_specs
        self.seed = seed
        self.hinge = hinge
        torch.manual_seed(self.seed)
        
        self._define_vars()

    @abstractmethod
    def _define_vars(self):
        pass

    @abstractmethod
    def _get_latents(self, inputs):
        pass

    @abstractmethod
    def _get_class_logits(self, latents):
        pass

    @abstractmethod
    def _get_sensitive_logits(self, latents):
        pass

    @abstractmethod
    def _get_recon_inputs(self, latents):
        pass

    @abstractmethod
    def _get_class_loss(self, pred, target):
        pass

    @abstractmethod
    def _get_recon_loss(self, pred, target):
        pass

    @abstractmethod
    def _get_aud_loss(self, pred, target):
        pass

    @abstractmethod
    def _get_loss(self):
        pass

    @abstractmethod
    def _get_class_preds_from_logits(self, logits):
        pass

    @abstractmethod
    def _get_aud_preds_from_logits(self, logits):
        pass

    def forward(self, X, Y, A):
        self.Z = self._get_latents(X)
        self.Y_hat_logits = self._get_class_logits(self.Z)
        self.Y_hat = self._get_class_preds_from_logits(self.Y_hat_logits)
        self.A_hat_logits = self._get_sensitive_logits(self._get_aud_inputs())
        self.A_hat = self._get_aud_preds_from_logits(self.A_hat_logits)
        self.X_hat = self._get_recon_inputs(self.Z)
        self.class_loss = self._get_class_loss(self.Y_hat, Y)
        self.recon_loss = self._get_recon_loss(self.X_hat, X)
        self.aud_loss = self._get_aud_loss(self.A_hat, A)
        self.loss = self._get_loss()
        self.class_err = classification_error(Y, self.Y_hat)
        self.aud_err = classification_error(A, self.A_hat)
        return self.loss

    def _get_aud_inputs(self):
        return self.Z

# Utility functions
def classification_error(target, pred):
    pred_class = torch.round(pred)
    return 1.0 - torch.mean((target == pred_class).float())

def cross_entropy(target, pred, weights=None, eps=EPS):
    if weights is None:
        weights = torch.ones_like(pred)
    return -torch.squeeze(weights * (target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps)))



class DemParGan(AbstractBaseNet):
    def _define_vars(self):
        assert(
            isinstance(self.hidden_layer_specs, dict) and
            all([net_name in self.hidden_layer_specs for net_name in ['enc', 'cla', 'aud', 'rec']])
        )

    def _get_latents(self, inputs):
        mlp = MLP(shapes=[self.xdim] + self.hidden_layer_specs['enc'] + [self.zdim], activ=ACTIV)
        return mlp(inputs)

    def _get_class_logits(self, latents):
        mlp = MLP(shapes=[self.zdim] + self.hidden_layer_specs['cla'] + [self.ydim], activ=ACTIV)
        return mlp(latents)

    def _get_sensitive_logits(self, latents):
        mlp = MLP(shapes=[self.zdim] + self.hidden_layer_specs['aud'] + [self.adim], activ=ACTIV)
        return mlp(latents)

    def _get_recon_inputs(self, latents):
        mlp = MLP(shapes=[self.zdim + 1] + self.hidden_layer_specs['rec'] + [self.xdim], activ=ACTIV)
        Z_and_A = torch.cat([self.Z, self.A], dim=1)
        return mlp(Z_and_A)

    def _get_class_loss(self, Y_hat, Y):
        return cross_entropy(Y, Y_hat)

    def _get_recon_loss(self, X_hat, X):
        return torch.mean(torch.square(X - X_hat), dim=1)

    def _get_aud_loss(self, A_hat, A):
        return cross_entropy(A, A_hat)

    def _get_weight_decay(self):
        return sum(torch.sum(p.pow(2.0)) for p in self.parameters())

    def _get_loss(self):
        return torch.mean(torch.stack([
            self.class_coeff * self.class_loss,
            self.recon_coeff * self.recon_loss,
            -self.fair_coeff * self.aud_loss
        ]))

    def _get_class_preds_from_logits(self, logits):
        return torch.sigmoid(logits)

    def _get_aud_preds_from_logits(self, logits):
        return torch.sigmoid(logits)


class EqOddsUnweightedGan(DemParGan):
    def _get_aud_inputs(self):
        return torch.cat([self.Z, self.Y], dim=1)

    def _get_sensitive_logits(self, inputs):
        mlp = MLP(shapes=[self.zdim + 1 * self.ydim] + self.hidden_layer_specs['aud'] + [self.adim], activ=ACTIV)
        return mlp(inputs)


class WassGan(AbstractBaseNet):
    def _get_class_loss(self, Y_hat, Y):
        return wass_loss(Y, Y_hat)

    def _get_aud_loss(self, A_hat, A):
        return wass_loss(A, A_hat)

    def _get_class_preds_from_logits(self, logits):
        return logits

    def _get_aud_preds_from_logits(self, logits):
        return logits


def wass_loss(target, pred):
    return torch.squeeze(torch.abs(target - pred))


class DemParWassGan(WassGan, DemParGan):
    def _get_class_loss(self, Y_hat, Y):
        return WassGan._get_class_loss(self, Y_hat, Y)

    def _get_aud_loss(self, A_hat, A):
        return WassGan._get_aud_loss(self, A_hat, A)


class WeightedGan(AbstractBaseNet):
    def __init__(self, *args, A_weights=A_WTS, Y_weights=Y_WTS, AY_weights=AY_WTS, **kwargs):
        self.A_weights = A_weights
        self.Y_weights = Y_weights
        self.AY_weights = AY_weights
        super().__init__(*args, **kwargs)

    def forward(self, X, Y, A):
        super().forward(X, Y, A)
        self.unweighted_aud_loss = self._get_aud_loss(self.A_hat, A)
        self.aud_loss = self._get_weighted_aud_loss(self.unweighted_aud_loss, self.A_weights, self.Y_weights, self.AY_weights)
        self.loss = self._get_loss()
        return self.loss

    @abstractmethod
    def _get_weighted_class_loss(self, L, A_wts, Y_wts, AY_wts):
        pass

    @abstractmethod
    def _get_weighted_recon_loss(self, L, A_wts, Y_wts, AY_wts):
        pass

    @abstractmethod
    def _get_weighted_aud_loss(self, L, A_wts, Y_wts, AY_wts):
        pass


class WeightedDemParGan(WeightedGan, DemParGan):
    def _weight_loss(self, L, A_wts, Y_wts, AY_wts):
        A0_wt, A1_wt = A_wts
        wts = A0_wt * (1. - self.A) + A1_wt * self.A
        wtd_L = L * torch.squeeze(wts)
        return wtd_L

    def _get_weighted_class_loss(self, L, A_wts, Y_wts, AY_wts):
        return self._weight_loss(L, A_wts, Y_wts, AY_wts)

    def _get_weighted_recon_loss(self, L, A_wts, Y_wts, AY_wts):
        return self._weight_loss(L, A_wts, Y_wts, AY_wts)

    def _get_weighted_aud_loss(self, L, A_wts, Y_wts, AY_wts):
        return self._weight_loss(L, A_wts, Y_wts, AY_wts)