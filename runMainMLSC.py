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

def runMain(config,f,args):
    rownetworks, truefeatures_list, labels, idx_train=process.load_data_mv(args,Unified=False)
    
    args.rownetworks, args.truefeatures_list, args.labels, args.idx_train=rownetworks, truefeatures_list, labels, idx_train

    for lr in config[args.dataset]['lr']:
        for l2_coef in config[args.dataset]['l2_coef']:
            for reg_coef in config[args.dataset]['reg_coef']:
                for hid in config[args.dataset]['hid']:
                    args.lr = lr
                    args.hid_units = hid
                    args.l2_coef = l2_coef
                    args.dataset = args.dataset
                    args.reg_coef = reg_coef

                    print(args)

                    from models import DMGI
                    embedder = DMGI(args)
                    try:
                        nmi, acc, ari, stdacc, stdnmi, stdari, retxt = embedder.training(f)
                        result = "hid_units:{},lr:{},l2_coef:{},reg_coef:{},acc:{},nmi:{}，Ari:{},stdnmi:{},stdacc:{},stdari:{}".format(
                            hid, lr, l2_coef, reg_coef, acc, nmi, ari, stdacc, stdnmi, stdari)

                        f.write(retxt)
                        f.write('\n')
                        f.write(result)
                        f.write('\n')
                        f.write('\n')
                        f.flush()
                    except Exception as e:
                        print(e)


if __name__ == '__main__':

    d=['BBCSport'] #['Reuters','yale_mtv','MSRCv1','3sources','small_Reuters','small_NUS','BBC','BBCSport'] # ['BBCSport','yale_mtv','MSRCv1','3sources']
    for data in d:
        for link in ['Mean']:
            config = yaml.load(open("configMainMLSC_Mean.yaml", 'r'))
            
            # input arguments
            parser = argparse.ArgumentParser(description='DMGI')
            parser.add_argument('--embedder', nargs='?', default='DMGI')
            parser.add_argument('--dataset', nargs='?', default=data)
            parser.add_argument('--View_num',default=config[data]['View_num'])
            parser.add_argument('--norm',default=config[data]['norm'])
            parser.add_argument('--nb_epochs', type=int, default=config[data]['nb_epochs'])
            parser.add_argument('--sc', type=float, help='GCN self connection', default=10) #config[data]['sc']
            parser.add_argument('--gpu_num', type=int, default=0)
            parser.add_argument('--drop_prob', type=float, default=0.2)
            parser.add_argument('--patience', type=int, default=100)
            parser.add_argument('--nheads', type=int, default=1)
            parser.add_argument('--activation', nargs='?', default='leakyrelu')
            
            parser.add_argument('--isBias',default=False)
            
            parser.add_argument('--isAttn',  default=False)
            
            parser.add_argument('--isMeanOrCat', nargs='?', default=link) #config[data]['isMeanOrCat']
            parser.add_argument('--Weight', nargs='?', default=config['Weight'])
            args, unknown = parser.parse_known_args()
                        
            print(args)

            resultsDir = 'baseline/DMGIVisulazation/{}/{}'.format(args.isMeanOrCat,args.dataset)
            mkdir(resultsDir)
            
            filePath = os.path.join(resultsDir, '{}_{}_{}_{}_sc.{}.txt'.format('Y 'if args.Weight else 'N',args.dataset,args.isMeanOrCat,config[args.dataset]['norm'],args.sc))
    

            # with open(filePath, 'w+') as f:
            #     for sc in [1,1,1,1,1]:
            #         args.sc=sc
            #         f.write("SC:{}\n".format(args.sc))
            #         runMain(config,f,args)
            #         f.flush()
            
            with open(filePath, 'w+') as f:
                f.write("SC:{}\n".format(args.sc))
                runMain(config,f,args)
                f.flush()

