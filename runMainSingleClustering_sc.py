import numpy as np

from utils import process
from utils.utils import mkdir

np.random.seed(0)
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse
import os
import yaml
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, pairwise,adjusted_rand_score
def acc_val(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def run_kmeans_yypred(y_pred, y):
    
    NMI_list = []
    ACM_list = []
    ARI_list=[]
    for i in range(1):
        acc = acc_val(np.array(y), np.array(y_pred))
        ACM_list.append(acc)
        s1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari=adjusted_rand_score(y, y_pred)
        ARI_list.append(ari)
        NMI_list.append(s1)

    nmiscore = sum(NMI_list) / len(NMI_list)
    acc = sum(ACM_list) / len(ACM_list)
    ari=sum(ARI_list)/len(ARI_list)

    return nmiscore,acc,ari,np.std(ACM_list),np.std(NMI_list),np.std(ARI_list)

def runMain(config,f,args):
    rownetworks, truefeatures_list, labels, idx_train=process.load_data_mv(args,Unified=False)
    
    args.rownetworks, args.truefeatures_list, args.labels, args.idx_train=rownetworks, truefeatures_list, labels, idx_train
    from sklearn.cluster import KMeans
    
    for view,data in enumerate(args.truefeatures_list):
        NMI_list=[]
        ACM_list=[]
        ARI_list=[]
        for count in range(5):
            for index, gamma in enumerate((100)): #((0.01, 0.1,0.5, 1, 10))
                y_pred = SpectralClustering(n_clusters=np.shape(args.labels)[1], gamma=gamma,affinity='nearest_neighbors').fit_predict(data)

                nmi,acc,ari,stdacc,stdnmi,stdari=run_kmeans_yypred(y_pred,np.argmax(args.labels,1))
                NMI_list.append(nmi)
                ACM_list.append(acc)
                ARI_list.append(ari)
                                
                print("view:{}\tgamma:{}\tACC:{}\tNMI:{}\tARI:{}".format(view+1,gamma,acc, nmi,ari))
            
            # f.write("view:{}\tgamma:{}\tACC:{}\tNMI:{}\n".format(view+1,gamma,acc, nmire))
            # f.flush()
        
        # nmiscore = sum(NMI_list) / len(NMI_list)
        # acc = sum(ACM_list) / len(ACM_list)
        # ari=sum(ARI_list)/len(ARI_list)
            
        # with open(filePath, 'a+') as f:
            
        #     result = "View:{}, acc:{},nmi:{}，Ari:{},stdnmi:{},stdacc:{},stdari:{}".format((view+1),acc, nmiscore, ari, np.std(ACM_list), np.std(NMI_list), np.std(ARI_list))

        #     print(result)
        #     f.write(result)
        #     f.write("\n")
        #     f.flush()
            
    
    


if __name__ == '__main__':

    d=['BBCSport'] #['Reuters','yale_mtv','MSRCv1','3sources','small_Reuters','small_NUS','BBC','BBCSport'] # ['BBCSport','yale_mtv','MSRCv1','3sources']
    for data in d:
        for link in ['Cat']:
            config = yaml.load(open("configMainML.yaml", 'r'))
            
            # input arguments
            parser = argparse.ArgumentParser(description='DMGI')
            parser.add_argument('--embedder', nargs='?', default='DMGI')
            parser.add_argument('--dataset', nargs='?', default=data)
            parser.add_argument('--View_num',default=config[data]['View_num'])
            parser.add_argument('--sc', type=float, default=1.0, help='GCN self connection') #config[data]['sc']
            parser.add_argument('--Weight', nargs='?', default=False)
            
            args, unknown = parser.parse_known_args()
                        
            print(args)

            resultsDir = 'baseline/Spectral/{}'.format(args.dataset)
            mkdir(resultsDir)
            
            filePath = os.path.join(resultsDir, '{}_{}_sc.{}.txt'.format(args.dataset,config[args.dataset]['norm'],args.sc))
    
            
            with open(filePath, 'w+') as f:
                f.write("SC:{}\n".format(args.sc))
                runMain(config,f,args)
                f.flush()

