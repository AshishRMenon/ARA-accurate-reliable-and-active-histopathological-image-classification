import numpy as np
from matplotlib import pyplot as plt
import os
from skimage import io
import pathlib
from sklearn.manifold import TSNE
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import time
import copy
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import pandas as pd



import torch
import torchvision
from torchvision import datasets,models,transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.datasets as dset



parser = argparse.ArgumentParser(description='Process args for patch extraction')
parser.add_argument("--root_dir", type=str, help="Root directory", default="./tissue_images/")
parser.add_argument("--model_load_path", type=str, default="./")
args = parser.parse_args()



class ImageFolder_Mod(dset.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self,root,transform):
        super(ImageFolder_Mod,self).__init__(root,transform)
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super().__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        imgname= path.split('/')[-1]
        #print(f"path {path} index {index}")
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,imgname))
        return tuple_with_path

  
    
def test_model(model,dataloader,dataset):
    since=time.time()
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model.eval()
    running_loss_aux = 0
    running_loss_out = 0
    running_corrects_aux = 0
    running_corrects_out = 0
    
    epoch_list =[]
    epoch_train = []
    epoch_loss_out =[]
    epoch_loss_aux = []
    epoch_acc_out = []
    epoch_acc_aux = []
    ap = []
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pred = {}
    gt ={}
    precision = {}
    recall = {}
    average_precision = {}
    img_raw_tsne=[]
    image_embedding = {}
    epoch_acc_out_test = []
    img_raw_tsne=[]
                
    label_pred = []
    for key in range(8):
        running_corrects_out_test = 0
        for i , data in enumerate (dataloader):
            images=data[0].to(device)
            #print(images)
            label_binarized = label_binarize(data[1].numpy(), classes=[0,1,2,3,4,5,6,7])
            output_main = model(images)
            
            output_main = output_main.detach()
            output_main_pred = F.softmax(output_main)
            _,pred_out = torch.max(output_main_pred,1)
                
            if key ==0:
                for k in range(len(data[2])):
                    image_raw = Image.open(data[2][k]).convert('L')
                    image_raw = np.array(image_raw)
                    image_raw = image_raw.flatten()
                    img_raw_tsne.append(image_raw)
                    image_embedding[data[3][k]] = output_main[k].detach().cpu().numpy()
                    label_pred.append(pred_out.cpu().numpy()[k])
                    

                df1 = pd.DataFrame(list(image_embedding.keys()))
                df2 = pd.DataFrame(image_embedding.values())
                df3 = pd.DataFrame(label_pred)

                print(np.array(img_raw_tsne).shape)
                tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
                tsne_results = tsne.fit_transform(np.array(img_raw_tsne))
                
                df_raw_tsne_val = pd.DataFrame(tsne_results)
                
                df = pd.DataFrame(np.column_stack((df1,df2,df3,df_raw_tsne_val)))
                df.columns = ['Image Name','f1','f2','f3','f4','f5','f6','f7','f8','label','tsne1','tsne2','tsne3']
                df.to_excel('output_emb.xlsx', engine='xlsxwriter')
                

            predictions_binary = (pred_out == key) * 1
            label_binary = (data[1].to(device)== key) * 1
            output_main_pred = output_main_pred.cpu().numpy()
            pred [i] = output_main_pred
            gt[i] = label_binarized
            #running_corrects_out_test+=torch.sum(pred_out == data[1].to(device).data)
            running_corrects_out_test+=torch.sum(predictions_binary == label_binary)
        epoch_acc_out_test.append((running_corrects_out_test.double()/len(dataset)).cpu().item())
    
    GT = np.concatenate((gt[0], gt[1],gt[2],gt[3]), axis=0)
    PR = np.concatenate((pred[0], pred[1],pred[2],pred[3]), axis=0)
    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(GT[:,i], PR[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], thresholds = precision_recall_curve(GT[:,i], PR[:, i])
        average_precision[i] = average_precision_score(GT[:,i], PR[:, i])

    return roc_auc,fpr,tpr,precision,recall,average_precision




if __name__ == '__main__':

    T = {'train': transforms.Compose([ transforms.Resize(224),
                                       #transforms.RandomHorizontalFlip(),
                                       #transforms.RandomVerticalFlip(),
                                       #transforms.RandomRotation((0,90)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))       
                                     ]) ,
          
        'val': transforms.Compose([  transforms.Resize(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))       
                                     ]) ,
          
        'test': transforms.Compose([   transforms.Resize(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))       
                                     ]) 
        }

    root = args.root_dir

    datasets={x: ImageFolder_Mod( os.path.join(root,x), T[x] ) for x in os.listdir(root) }
    dataloaders = {}
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'] , shuffle=True, batch_size=128 , num_workers=16)

    datalength= { x : len(datasets[x]) for x in os.listdir(root)}
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    class_names=datasets['test'].classes
    num_ftrs = 1000
    model_ft = nn.Sequential(models.squeezenet1_1(), nn.Linear(num_ftrs,len(class_names)))


    model_ft = model_ft.to(device)


    print('Loading Model', flush=True)
    model_ft.load_state_dict(torch.load(args.model_load_path))
    print('Model Loaded', flush=True)

    
    roc,fpr,tpr,pr , re , avg_pr = test_model(model_ft,dataloaders['test'],datasets['test'])


    classes = datasets['train'].classes



    ax = plt.subplot(111)

    for i in range(8):
        ax.plot(fpr[i], tpr[i], lw=1.5,  label=classes[i]+ " " + "AUC=" + str(roc[i]))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    #fig.savefig('samplefigure', bbox_inches='tight')
    plt.savefig('Output_ROC_curve.jpg',bbox_inches='tight')
    #plt.show()



    ax = plt.subplot(111)

    for i in range(8):
        ax.plot(re[i],pr[i],label=classes[i]+ " " + "AP=" + str(avg_pr[i]))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    #fig.savefig('samplefigure', bbox_inches='tight')
    plt.savefig('Output_PR_curve.jpg',bbox_inches='tight')
    #plt.show()

    print(sum(list(roc.values()))/8)
    print(sum(list(avg_pr.values()))/8)
    sum1 = 0
    for i in range(8):
        sum1+=len(pr[i])
        print('Precision',pr[i])
        print('\n')
        print('Recall', re[i])