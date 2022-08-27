import math
from re import S
from numpy.core.numeric import ones_like
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import pdb
from scipy import stats
from torch.utils.data.dataset import Dataset
from config import BATCH_SIZE, DEVICE, NUM_WORKERS
from NetworkDataset import *
from torch import linalg as la

class Strategy:
    def __init__(self, X, Y, transforms, train_valid_raio = 0.8):
        self._selected_indices = np.zeros(len(X), dtype=bool);
        self._X = np.array(X);
        self._Y = np.array(Y);
        self._transforms = transforms;
        self._train_valid_split_ratio = train_valid_raio;
        pass

    def query(self, n, model):
        pass

    def get_random_initial_data(self, n):
        cnt = 0;
        indices = [];
        while cnt!= n:
            r = np.random.randint(0, len(self._X));
            while r in indices:
                r = np.random.randint(0, len(self._X));
            indices.append(r);
            cnt+=1;
        
        self._selected_indices[indices] = True;
        return self._X[self._selected_indices], self._Y[self._selected_indices]
    
    def get_total_selected_data(self):
        return np.sum(self._selected_indices);
    
    def train_valid_split(self, model, X, Y, loss_fn):
        model.train();
        loader = DataLoader(NetworkDataset(X, Y, self._transforms), batch_size = 1, shuffle= True, num_workers=NUM_WORKERS);
        embeddings = np.zeros((len(X), model.get_num_weight_parameters()), dtype=np.float);
        with torch.enable_grad():
            for x,y,idx in loader:
                x,y  = x.to(DEVICE), y.to(DEVICE);
                cout, out = model(x);
                loss = loss_fn(cout, y.float());
                loss.backward();
                grads = [];
                for name, W in model.named_parameters():
                    if 'weight' in name:
                        grads.append(la.norm(W.grad.view(-1), dim = 0, ord= 2).item());
                embeddings[idx] = grads;
                model.zero_grad();
        
        selected_indices = self.get_cluster_centers(embeddings, math.floor(self._train_valid_split_ratio * len(X)), None);
        total_indices = np.zeros(len(X), dtype=bool);
        total_indices[selected_indices] = True;
        X_train= X[total_indices];
        X_valid = X[~total_indices];
        Y_train=  Y[total_indices];
        Y_valid = Y[~total_indices];
        return X_train, X_valid, Y_train, Y_valid;
    
    def get_cluster_centers(self, X, K, data):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1

        return indsAll
    
class BadgeStrategy(Strategy):
    def __init__(self, X, Y, transforms):
        super().__init__(X, Y, transforms);

    def query(self, n, model):
        idx_unlabeled = np.arange(len(self._X), )[~self._selected_indices];
        grad_embeddings = self.get_grad_embeddings(idx_unlabeled, model);
        new_indices = self.get_cluster_centers(grad_embeddings, n, self._X[idx_unlabeled])
        self._selected_indices[new_indices] = True;
        return self._X[self._selected_indices], self._Y[self._selected_indices];

    def get_grad_embeddings(self, idx_unlabeled, model):
        model.eval();
        embed_dim = model.get_embedd_dim();
        embeddings = np.zeros((len(idx_unlabeled), embed_dim));
        loader = DataLoader(NetworkDataset(self._X[idx_unlabeled], self._Y[idx_unlabeled], self._transforms), batch_size= BATCH_SIZE, shuffle= True, num_workers= NUM_WORKERS);
        with torch.no_grad():
            for x , y, idxs in loader:
                x,y  = x.to(DEVICE), y.to(DEVICE);
                cout, out = model(x);
                out = out.data.cpu().numpy()
                batchProbs = torch.sigmoid(cout).cpu().numpy();
                cls = batchProbs > 0.5;
                cls = np.where(cls==0, -1, cls);
                for j in range(len(y)):
                    embedd = (1-batchProbs[j]) * cls[j]*out[j];
                    embedd = np.sum(embedd, axis=(1,2), keepdims=False);
                    embeddings[idxs[j]] = embedd;
        
        return embeddings;

class RandomSampling(Strategy):
    def __init__(self, X, Y, transforms, train_valid_raio=0.8):
        super().__init__(X, Y, transforms, train_valid_raio);
    
    def query(self, n, model):
        cnt = 0;
        indices = [];
        while cnt!= n:
            r = np.random.randint(0, len(self._X));
            #while number is selcted now or before
            while r in indices or self._selected_indices[r] is True:
                r = np.random.randint(0, len(self._X));
            indices.append(r);
            cnt+=1;
        
        self._selected_indices[indices] = True;
        return self._X[self._selected_indices], self._Y[self._selected_indices]
    