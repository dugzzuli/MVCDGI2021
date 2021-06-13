import os,sys

sys.path.append("./")  

import numpy as np
import linecache as liner
from utils.utils import mkdir

if __name__ == '__main__':
    
    accList=[]
    nmiList=[]
    ariList=[]
    sclist=[0,1 ,2 ,3 ,4 ,5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    linkType="Cat"
    datasetname='small_Reuters'
    for sc in sclist:
        txtpath="baseline_knn_15/DMGIAdjuSC/{}/{}/N_{}_Cat_False_sc.{}.0.txt".format(linkType,datasetname,datasetname,sc)
        cach_data=liner.getlines(txtpath)
        
        for line in cach_data:
            
            if(line.startswith('loss')):
                arrs=line.split(' ')
                
                temp=arrs[4].split(':')[1]
                print(float(temp))
                accList.append(float(temp))
                temp=arrs[5].split(':')[1]
                print(float(temp))
                nmiList.append(float(temp))
                temp=arrs[6].split(':')[1]
                print(float(temp))
                ariList.append(float(temp))
                
                print(arrs)
    
    print(accList)
    print(nmiList)
    print(ariList)
    savedir='baseline_knn_15/SCAdjust/{}/{}/'.format(datasetname,linkType)
    mkdir(savedir)
    np.savetxt(savedir+"acc.csv",accList,delimiter=',',encoding='utf-8')
    np.savetxt(savedir+"nmi.csv",nmiList,delimiter=',',encoding='utf-8')
    np.savetxt(savedir+"ari.csv",ariList,delimiter=',',encoding='utf-8')
    
    