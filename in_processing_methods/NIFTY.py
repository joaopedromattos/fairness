import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    SAGEConv,
    DeepGraphInfomax,
    JumpingKnowledge,
)
import os
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from torch.nn.utils import spectral_norm
from torch_geometric.utils import dropout_adj, convert

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import argparse
import numpy as np
import scipy.sparse as sp
from torch_geometric.typing import SparseTensor

from torch_geometric.data import Data


from NCNC import CN0LinkPredictor, CNLinkPredictor
from SEAL import SEALDataset, GCN_SEAL


class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes, base_model="standard"):
        super(Classifier, self).__init__()
        
        self.base_model = base_model

        # Classifier projector
        if base_model == 'classifier':
            self.predictor = spectral_norm(nn.Linear(ft_in, nb_classes))
        elif base_model == 'gae':
            self.predictor = CN0LinkPredictor(in_channels=ft_in, hidden_channels=ft_in, out_channels=nb_classes, num_layers=2, dropout=0.5)
        elif base_model == 'ncn':
            self.predictor = CNLinkPredictor(in_channels=ft_in, hidden_channels=ft_in, out_channels=nb_classes, num_layers=2, dropout=0.5)
        elif base_model == 'seal':
            self.predictor = nn.Linear(ft_in, 2)

    def forward(self, x, adj, tar_ei):
        ret = None
        if self.base_model == 'standard':
            ret = self.predictor(x)
        elif self.base_model == 'gae':
            ret = self.predictor.multidomainforward(x, adj, tar_ei)
        elif self.base_model == 'ncn':
            adj = SparseTensor.from_edge_index(adj, sparse_sizes=(x.size(0), x.size(0)))
            ret = self.predictor.multidomainforward(x, adj, tar_ei)
        elif self.base_model == 'seal':
            ret = self.predictor(x)
            
        return ret


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x


class GIN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            spectral_norm(nn.Linear(nfeat, nhid)),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            spectral_norm(nn.Linear(nhid, nhid)),
        )
        self.conv1 = GINConv(self.mlp1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class JK(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(JK, self).__init__()
        self.conv1 = spectral_norm(GCNConv(nfeat, nhid))
        self.convx = spectral_norm(GCNConv(nhid, nhid))
        self.jk = JumpingKnowledge(mode="max")
        self.transition = nn.Sequential(
            nn.ReLU(),
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        xs = []
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        xs.append(x)
        for _ in range(1):
            x = self.convx(x, edge_index)
            x = self.transition(x)
            xs.append(x)
        x = self.jk(xs)
        return x


class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(SAGE, self).__init__()

        # Implemented spectral_norm in the sage main file
        # ~/anaconda3/envs/PYTORCH/lib/python3.7/site-packages/torch_geometric/nn/conv/sage_conv.py
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = "mean"
        self.transition = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(nhid), nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = "mean"

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        return x


class Encoder_DGI(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Encoder_DGI, self).__init__()
        self.hidden_ch = nhid
        self.conv = spectral_norm(GCNConv(nfeat, self.hidden_ch))
        self.activation = nn.PReLU()

    def corruption(self, x, edge_index):
        # corrupted features are obtained by row-wise shuffling of the original features
        # corrupted graph consists of the same nodes but located in different places
        return x[torch.randperm(x.size(0))], edge_index

    def summary(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x


class GraphInfoMax(nn.Module):
    def __init__(self, enc_dgi):
        super(GraphInfoMax, self).__init__()
        self.dgi_model = DeepGraphInfomax(
            enc_dgi.hidden_ch, enc_dgi, enc_dgi.summary, enc_dgi.corruption
        )

    def forward(self, x, edge_index):
        pos_z, neg_z, summary = self.dgi_model(x, edge_index)
        return pos_z


class Encoder(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, max_z:int, base_model="gcn", k: int = 2
    ):
        super(Encoder, self).__init__()
        self.base_model = base_model
        if self.base_model == "gcn":
            self.conv = GCN(in_channels, out_channels)
        elif self.base_model == "gin":
            self.conv = GIN(in_channels, out_channels)
        elif self.base_model == "sage":
            self.conv = SAGE(in_channels, out_channels)
        elif self.base_model == "infomax":
            enc_dgi = Encoder_DGI(nfeat=in_channels, nhid=out_channels)
            self.conv = GraphInfoMax(enc_dgi=enc_dgi)
        elif self.base_model == "jk":
            self.conv = JK(in_channels, out_channels)
        elif self.base_model == "seal":
            self.conv = GCN_SEAL(num_features=in_channels, hidden_channels=out_channels, num_layers=3, max_z=max_z)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, z=None, batch=None, edge_weight=None, node_id=None) -> torch.Tensor:
        if self.base_model == "seal":
            print("Inside encoder")
            import code
            code.interact(local={**locals(), **globals()})
            return self.conv(z, edge_index, batch, x, edge_weight, node_id)
        else:
            return self.conv(x, edge_index)
        


class NIFTY(torch.nn.Module):
    def __init__(
        self,
        adj,
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        sens,
        sens_idx,
        edge_splits,
        dataset_name,
        num_hidden=16,
        num_proj_hidden=16,
        lr=0.0001,
        weight_decay=1e-5,
        drop_edge_rate_1=0.1,
        drop_edge_rate_2=0.1,
        drop_feature_rate_1=0.1,
        drop_feature_rate_2=0.1,
        encoder="gcn",
        decoder='standard',
        sim_coeff=0.5,
        nclass=2,
        device="cuda",
    ):
        super(NIFTY, self).__init__()

        self.device = device

        # self.edge_index = convert.from_scipy_sparse_matrix(sp.coo_matrix(adj.to_dense().numpy()))[0]
        self.edge_index = adj.coalesce().indices()
        self.decoder_name = decoder
        self.encoder_name = encoder
        self.dataset_name = dataset_name
        self.encoder = Encoder(
            in_channels=features.shape[1], out_channels=num_hidden, base_model=encoder, max_z=features.shape[0]
        ).to(device)
        # model = SSF(encoder=encoder, num_hidden=args.hidden, num_proj_hidden=args.proj_hidden, sim_coeff=args.sim_coeff,
        # nclass=num_class).to(device)
        self.val_edge_index_1 = dropout_adj(
            self.edge_index.to(device), p=drop_edge_rate_1
        )[0]
        self.val_edge_index_2 = dropout_adj(
            self.edge_index.to(device), p=drop_edge_rate_2
        )[0]
        self.val_x_1 = drop_feature(
            features.to(device), drop_feature_rate_1, sens_idx, sens_flag=False
        )
        self.val_x_2 = drop_feature(features.to(device), drop_feature_rate_2, sens_idx)

        self.sim_coeff = sim_coeff
        # self.encoder = encoder
        self.labels = labels

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.sens_idx = sens_idx
        self.drop_edge_rate_1 = self.drop_edge_rate_2 = 0
        self.drop_feature_rate_1 = self.drop_feature_rate_2 = 0

        # Projection
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_proj_hidden)),
            nn.BatchNorm1d(num_proj_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(num_proj_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
        )

        # Prediction
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(inplace=True),
        )
        self.fc4 = spectral_norm(nn.Linear(num_hidden, num_hidden))

        # Classifier
        self.c1 = Classifier(ft_in=num_hidden, nb_classes=nclass, base_model=decoder)

        for m in self.modules():
            self.weights_init(m)

        par_1 = (
            list(self.encoder.parameters())
            + list(self.fc1.parameters())
            + list(self.fc2.parameters())
            + list(self.fc3.parameters())
            + list(self.fc4.parameters())
        )
        par_2 = list(self.c1.parameters()) + list(self.encoder.parameters())
        self.optimizer_1 = optim.Adam(par_1, lr=lr, weight_decay=weight_decay)
        self.optimizer_2 = optim.Adam(par_2, lr=lr, weight_decay=weight_decay)
        self = self.to(device)

        self.features = features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.labels = self.labels.to(device)
        
        # Adds all edges and labels as attributes
        for edge_set in edge_splits.keys():
            setattr(self, f'{edge_set}_edge_index', torch.cat([edge_splits[edge_set]['edge'].t(), edge_splits[edge_set]['edge_neg'].t()], dim=-1))
            setattr(self, f'{edge_set}_edge_labels', F.one_hot(torch.cat([torch.ones(edge_splits[edge_set]['edge'].size(0)), 
                                                                          torch.zeros(edge_splits[edge_set]['edge_neg'].size(0))]).long(), 
                                                               num_classes=2).float().to(self.device))
            
        if encoder == 'seal' and decoder == 'seal':
            data = Data(x=self.features, edge_index=self.edge_index, y=self.labels, train_mask=self.idx_train, val_mask=self.idx_val, test_mask=self.idx_test)
            self.seal_dataset = SEALDataset(root='data', data=data.cpu(), split_edge=edge_splits, num_hops=2, split='train').to(device)
            
        

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, z=None, batch=None, edge_weight=None, node_id=None) -> torch.Tensor:
        return self.encoder(x, edge_index, z, batch, edge_weight, node_id)

    def projection(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        return z

    def prediction(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        return z

    def classifier(self, x, adj=None, tar_ei=None):
        return self.c1(x, adj, tar_ei)

    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()
        x = x.div(val.unsqueeze(dim=1).expand_as(x))
        return x

    def D_entropy(self, x1, x2):
        x2 = x2.detach()
        return (
            -torch.max(F.softmax(x2), dim=1)[0]
            * torch.log(torch.max(F.softmax(x1), dim=1)[0])
        ).mean()

    def D(self, x1, x2):  # negative cosine similarity
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, e_1, e_2, idx):

        # projector
        p1 = self.projection(z1)
        p2 = self.projection(z2)

        # predictor
        h1 = self.prediction(p1)
        h2 = self.prediction(p2)

        # classifier
        c1 = self.classifier(z1)

        l1 = self.D(h1[idx], p2[idx]) / 2
        l2 = self.D(h2[idx], p1[idx]) / 2
        l3 = F.cross_entropy(c1[idx], z3[idx].squeeze().long().detach())

        return self.sim_coeff * (l1 + l2), l3

    def forwarding_predict(self, emb, adj=None, edge_index=None):

        # projector
        p1 = self.projection(emb)

        # predictor
        h1 = self.prediction(p1)

        # classifier
        c1 = self.classifier(emb, adj, edge_index)

        return c1

    def linear_eval(self, emb, labels, idx_train, idx_test):
        x = emb.detach()
        classifier = nn.Linear(in_features=x.shape[1], out_features=2, bias=True)
        classifier = classifier.to("cuda")
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=0.001, weight_decay=1e-4
        )
        for i in range(1000):
            optimizer.zero_grad()
            preds = classifier(x[idx_train])
            loss = F.cross_entropy(preds, labels[idx_train])
            loss.backward()
            optimizer.step()
            # if i%100==0:
            #     print(loss.item())
        classifier.eval()
        preds = classifier(x[idx_test]).argmax(dim=1)
        correct = (preds == labels[idx_test]).sum().item()
        return preds, correct / preds.shape[0]

    def ssf_validation(self, x_1, edge_index_1, x_2, edge_index_2, y):
        z1 = self.forward(x_1, edge_index_1)
        z2 = self.forward(x_2, edge_index_2)

        # projector
        p1 = self.projection(z1)
        p2 = self.projection(z2)

        # predictor
        h1 = self.prediction(p1)
        h2 = self.prediction(p2)

        l1 = self.D(h1[self.idx_val], p2[self.idx_val]) / 2
        l2 = self.D(h2[self.idx_val], p1[self.idx_val]) / 2
        sim_loss = self.sim_coeff * (l1 + l2)

        # classifier
        c1 = self.classifier(z1, edge_index_1, self.valid_edge_index)
        c2 = self.classifier(z2, edge_index_2, self.valid_edge_index)

        # Binary Cross-Entropy
        l3 = (
            F.binary_cross_entropy_with_logits(
                c1, self.valid_edge_labels
            )
            / 2
        )
        l4 = (
            F.binary_cross_entropy_with_logits(
                c2, self.valid_edge_labels
            )
            / 2
        )

        return sim_loss, l3 + l4

    def fit_GNN(self, epochs=300):
        best_loss = 100
        for epoch in range(epochs + 1):

            sim_loss = 0

            self.train()
            self.optimizer_2.zero_grad()
            edge_index_1 = self.edge_index
            x_1 = self.features

            # classifier
            z1 = self.forward(x_1, edge_index_1)
            c1 = self.classifier(z1)

            # Binary Cross-Entropy
            cl_loss = F.binary_cross_entropy_with_logits(
                c1[self.idx_train],
                self.labels[self.idx_train].unsqueeze(1).float().to(self.device),
            )

            cl_loss.backward()
            self.optimizer_2.step()

            # Validation
            self.eval()
            z_val = self.forward(self.features, self.edge_index)
            c_val = self.classifier(z_val)
            val_loss = F.binary_cross_entropy_with_logits(
                c_val[self.idx_val],
                self.labels[self.idx_val].unsqueeze(1).float().to(self.device),
            )

            # if epoch % 100 == 0:
            #     logger.info(f"[Train] Epoch {epoch}: train_c_loss: {cl_loss:.4f} | val_c_loss: {val_loss:.4f}")

            if (val_loss) < best_loss:
                self.val_loss = val_loss.item()

                best_loss = val_loss
                if not os.path.exists("data"):
                    os.makedirs("data")
                torch.save(self.state_dict(), f"data/weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt")

    def fit(self, epochs=300):

        # Train model
        t_total = time.time()
        best_loss = 100
        best_acc = 0
        
        for epoch in tqdm(range(epochs + 1)):
            t = time.time()

            sim_loss = 0
            cl_loss = 0
            rep = range(1, 2) if self.encoder_name != "seal" else self.seal_dataset
            for _ in rep:
                self.train()
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                edge_index_1 = dropout_adj(self.edge_index, p=self.drop_edge_rate_1)[0]
                edge_index_2 = dropout_adj(self.edge_index, p=self.drop_edge_rate_2)[0]
                x_1 = drop_feature(
                    self.features,
                    self.drop_feature_rate_1,
                    self.sens_idx,
                    sens_flag=False,
                )
                x_2 = drop_feature(
                    self.features, self.drop_feature_rate_2, self.sens_idx
                )
                
                if self.encoder_name == 'seal':
                    z1 = self.forward(x_1, edge_index_1, z=_.z, batch=_.batch, edge_weight=_.edge_weight, node_id=_.node_id)
                    z2 = self.forward(x_2, edge_index_2, z=_.z, batch=_.batch, edge_weight=_.edge_weight, node_id=_.node_id)
                else:
                    z1 = self.forward(x_1, edge_index_1)
                    z2 = self.forward(x_2, edge_index_2)

                # projector
                p1 = self.projection(z1)
                p2 = self.projection(z2)

                # predictor
                h1 = self.prediction(p1)
                h2 = self.prediction(p2)

                l1 = self.D(h1[self.idx_train], p2[self.idx_train]) / 2
                l2 = self.D(h2[self.idx_train], p1[self.idx_train]) / 2
                sim_loss += self.sim_coeff * (l1 + l2)

            (sim_loss / _).backward()
            self.optimizer_1.step()

            # classifier
            z1 = self.forward(x_1, edge_index_1)
            z2 = self.forward(x_2, edge_index_2)
            c1 = self.classifier(z1, edge_index_1, self.train_edge_index)
            c2 = self.classifier(z2, edge_index_2, self.train_edge_index)

            # Binary Cross-Entropy
            l3 = (
                F.binary_cross_entropy_with_logits(
                    c1,
                    self.train_edge_labels,
                )
                / 2
            )
            l4 = (
                F.binary_cross_entropy_with_logits(
                    c2,
                    self.train_edge_labels,
                )
                / 2
            )

            cl_loss = (1 - self.sim_coeff) * (l3 + l4)
            cl_loss.backward()
            self.optimizer_2.step()
            loss = sim_loss / _ + cl_loss

            # Validation
            self.eval()
            val_s_loss, val_c_loss = self.ssf_validation(
                self.val_x_1,
                self.val_edge_index_1,
                self.val_x_2,
                self.val_edge_index_2,
                self.labels,
            )
            emb = self.forward(self.val_x_1, self.val_edge_index_1)
            output = self.forwarding_predict(emb, self.val_edge_index_1, self.valid_edge_index)
            preds = (output.squeeze() > 0).type_as(self.labels)
            # auc_roc_val = roc_auc_score(
            #     self.valid_edge_labels.detach().cpu().numpy(),
            #     output.detach().cpu().numpy(),
            # )

            if epoch % 50 == 0:
                print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss/_):.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f}")

            if (val_c_loss + val_s_loss) < best_loss:
                self.val_loss = val_c_loss.item() + val_s_loss.item()

                print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
                best_loss = val_c_loss + val_s_loss
                if not os.path.exists("data"):
                    os.makedirs("data")
                torch.save(self.state_dict(), f"data/weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt")

    def predict_GNN(self):

        self.load_state_dict(torch.load(f"data/weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt"))
        self.eval()
        emb = self.forward(
            self.features.to(self.device), self.edge_index.to(self.device)
        )
        output = self.forwarding_predict(emb)

        output_preds = (
            (output.squeeze() > 0)
            .type_as(self.labels)[self.idx_test]
            .detach()
            .cpu()
            .numpy()
        )

        labels = self.labels.detach().cpu().numpy()
        idx_test = self.idx_test

        F1 = f1_score(labels[idx_test], output_preds, average="micro")
        ACC = accuracy_score(
            labels[idx_test],
            output_preds,
        )
        try:
            AUCROC = roc_auc_score(labels[idx_test], output_preds)
        except:
            AUCROC = "N/A"

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = (
            self.predict_sens_group(output_preds, idx_test)
        )

        SP, EO = self.fair_metric(
            output_preds,
            self.labels[idx_test].detach().cpu().numpy(),
            self.sens[idx_test].detach().cpu().numpy(),
        )

        return (
            ACC,
            AUCROC,
            F1,
            ACC_sens0,
            AUCROC_sens0,
            F1_sens0,
            ACC_sens1,
            AUCROC_sens1,
            F1_sens1,
            SP,
            EO,
        )

    def predict(self):
        global evaluating
        evaluating = True

        self.load_state_dict(torch.load(f"data/weights_ssf_{self.dataset_name}_{self.encoder_name}_{self.decoder_name}.pt"))
        self.eval()
        emb = self.forward(
            self.features.to(self.device), self.edge_index.to(self.device)
        )
        output = self.forwarding_predict(emb, self.edge_index, self.test_edge_index)
        
        counter_features = self.features.clone()
        counter_features[:, self.sens_idx] = 1 - counter_features[:, self.sens_idx]
        counter_output = self.forwarding_predict(
            self.forward(
                counter_features.to(self.device), self.edge_index.to(self.device)
            ), self.edge_index, self.test_edge_index
        )
        noisy_features = self.features.clone() + torch.ones(
            self.features.shape
        ).normal_(0, 1).to(self.device)
        noisy_output = self.forwarding_predict(
            self.forward(
                noisy_features.to(self.device), self.edge_index.to(self.device)
            ), self.edge_index, self.test_edge_index
        )
        
        return output, counter_output, noisy_output

        # Report
        output_preds = (output.squeeze() > 0).type_as(self.labels)

        counter_output_preds = (counter_output.squeeze() > 0).type_as(self.labels)
        noisy_output_preds = (noisy_output.squeeze() > 0).type_as(self.labels)
        auc_roc_test = roc_auc_score(
            self.test_edge_labels.cpu().numpy(),
            output.detach().cpu().numpy(),
        )
        counterfactual_fairness = 1 - (
            output_preds.eq(counter_output_preds)[self.idx_test].sum().item()
            / self.idx_test.shape[0]
        )
        robustness_score = 1 - (
            output_preds.eq(noisy_output_preds)[self.idx_test].sum().item()
            / self.idx_test.shape[0]
        )

        parity, equality = self.fair_metric(
            output_preds[self.idx_test].cpu().numpy(),
            self.labels[self.idx_test].cpu().numpy(),
            self.sens[self.idx_test].numpy(),
        )
        f1_s = f1_score(
            self.labels[self.idx_test].cpu().numpy(),
            output_preds[self.idx_test].cpu().numpy(),
        )

        output_preds = (
            (output.squeeze() > 0)
            .type_as(self.labels)[self.idx_test]
            .detach()
            .cpu()
            .numpy()
        )

        labels = self.labels.detach().cpu().numpy()
        idx_test = self.idx_test

        F1 = f1_score(labels[idx_test], output_preds, average="micro")
        ACC = accuracy_score(
            labels[idx_test],
            output_preds,
        )
        try:
            AUCROC = roc_auc_score(labels[idx_test], output_preds)
        except:
            AUCROC = "N/A"

        ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1 = (
            self.predict_sens_group(output_preds, idx_test)
        )

        SP, EO = self.fair_metric(
            output_preds,
            self.labels[idx_test].detach().cpu().numpy(),
            self.sens[idx_test].detach().cpu().numpy(),
        )

        return (
            ACC,
            AUCROC,
            F1,
            ACC_sens0,
            AUCROC_sens0,
            F1_sens0,
            ACC_sens1,
            AUCROC_sens1,
            F1_sens1,
            SP,
            EO,
        )
        # print report
        # print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
        # print(f'Parity: {parity} | Equality: {equality}')
        # print(f'F1-score: {f1_s}')
        # print(f'CounterFactual Fairness: {counterfactual_fairness}')
        # print(f'Robustness Score: {robustness_score}')

    def fair_metric(self, pred, labels, sens):

        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
        equality = abs(
            sum(pred[idx_s0_y1]) / sum(idx_s0_y1)
            - sum(pred[idx_s1_y1]) / sum(idx_s1_y1)
        )
        return parity.item(), equality.item()

    def predict_sens_group(self, output, idx_test):
        # pred = self.lgreg.predict(self.embs[idx_test])
        pred = output
        result = []
        for sens in [0, 1]:
            F1 = f1_score(
                self.labels[idx_test][self.sens[idx_test] == sens]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test] == sens],
                average="micro",
            )
            ACC = accuracy_score(
                self.labels[idx_test][self.sens[idx_test] == sens]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test] == sens],
            )
            try:
                AUCROC = roc_auc_score(
                    self.labels[idx_test][self.sens[idx_test] == sens]
                    .detach()
                    .cpu()
                    .numpy(),
                    pred[self.sens[idx_test] == sens],
                )
            except:
                AUCROC = "N/A"
            result.extend([ACC, AUCROC, F1])

        return result


def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1 - x[:, sens_idx]

    return x
