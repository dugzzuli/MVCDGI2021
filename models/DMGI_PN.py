import torch

from utils.utils import mkdir

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator, Attention
import numpy as np
np.random.seed(0)
from evaluate import evaluate
from models import LogReg
import pickle as pkl
from tqdm import tqdm

class DMGI_PN(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self,f):
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        cnt_wait = 0;
        best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        iters=tqdm(range(self.args.nb_epochs))
        accMax=-1
        nmiMax = -1
        ariMax=-1
        curepoch=-1
        for epoch in iters:

            model.train()
            xent_loss = None
            model.train()
            optimiser.zero_grad()
            idx = np.random.permutation(self.args.nb_nodes)

            shuf = [feature[:, idx, :] for feature in features]
            shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

            lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
            lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

            result = model(features, adj, shuf, self.args.sparse, None, None, None)
            logits = result['logits']

            for view_idx, logit in enumerate(logits):
                if xent_loss is None:
                    xent_loss = b_xent(logit, lbl)
                else:
                    xent_loss += b_xent(logit, lbl)

            loss = xent_loss

            



            if loss < best:
                best = loss
                cnt_wait = 0

                # torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder,0))
            else:
                cnt_wait =+ 1

            if cnt_wait == self.args.patience:
                break

            loss.backward()
            optimiser.step()

            # Evaluation
            if(epoch%10)==0:
                # print(loss)
                model.eval()

                nmi,acc,ari,stdacc,stdnmi,stdari=evaluate(result["PN"].data.detach(), self.idx_train, self.labels, self.args.device)
                if(accMax<acc):
                    accMax=acc
                    nmiMax=nmi
                    ariMax=ari
                    curepoch=epoch
                retxt="loss:{} epoch:{} acc:{} nmi:{} accMax:{} nmiMax:{} ariMax:{} curepoch:{}".format(loss.item(),epoch,acc,nmi,accMax,nmiMax,ariMax,curepoch)
                iters.set_description(retxt)
                # f.write("loss:{} epoch:{} acc:{} nmi:{} accMax:{} nmiMax:{} curepoch:{}".format(loss.item(),epoch,acc,nmi,accMax,nmiMax,curepoch))
                # f.write('\n')
                # print()

            # if ((epoch+1) % 200) == 0:
            #     emdH=model.H.data.detach()
            #     adjzeros=self.KNN(emdH,emdH)
            #     adjzeros=adjzeros.float()
            #     adj = [adjzeros.to(self.args.device) for adj_ in self.adj]
                # pass



        # model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder,0)))

        # Evaluation
        model.eval()
        nmi,acc,ari,stdacc,stdnmi,stdari=evaluate(result["PN"].data.detach(), self.idx_train, self.labels, self.args.device)
        return nmi,acc,ari,stdacc,stdnmi,stdari,retxt

    
class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList([GCN(hid, args.hid_units, args.activation, args.drop_prob, args.isBias) for _,hid in zip(range(args.nb_graphs),self.args.dims)])

        self.disc=Discriminator(args.hid_units)
        # self.disc=nn.ModuleList([Discriminator(args.hid_units) for _ in range(self.args.View_num)])


        if(self.args.isMeanOrCat=='Mean'):
            self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        else:
            self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units * self.args.View_num))
        self.readout_func = self.args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):
        h_1_all = []; h_2_all = []; c_all = []; logits = []
        result = {}

        for i in range(self.args.nb_graphs):
            h_1 = self.gcn[i](feature[i], adj[i], sparse)


            # how to readout positive summary vector
            c = self.readout_func(h_1)
            c = self.args.readout_act_func(c)  # equation 9
            h_2 = self.gcn[i](shuf[i], adj[i], sparse)


            logit = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)

        result['logits'] = logits
        



        
        if (self.args.isMeanOrCat == 'Mean'):
            h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all), 0).unsqueeze(0)
        else:
            h_1_all = torch.cat(h_1_all,2).squeeze().unsqueeze(0)
            h_2_all = torch.cat(h_2_all,2).squeeze().unsqueeze(0)
            
        result["PN"]=h_1_all

        return result
