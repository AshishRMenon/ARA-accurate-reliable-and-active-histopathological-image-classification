import os 
import numpy as np
#from PIL import Image
import numbers
import random
import pandas as pd
from skimage import io, transform
from matplotlib import pyplot as plt
import glob
import concurrent.futures
import multiprocessing
import random as rd
import copy
import time
from sklearn.manifold import TSNE
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from sklearn.preprocessing import label_binarize



import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset , DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets as dset
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models,datasets,transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import argparse



parser = argparse.ArgumentParser(description='Process args for patch extraction')
parser.add_argument("--root_dir", type=str, help="Root directory", default="./tissue_images/")
parser.add_argument("--root_dir_copy", type=str, help="Destination directory", default="./tissue_images_copy/")
parser.add_argument("--active_learning_dir", type=str, default="./ara_active_expts")
parser.add_argument("--random_transfer_dir", type=str, default="./ara_random_expts")
parser.add_argument("--log_dir", type=str, default="./logs/ara_resnet")
parser.add_argument("--num_train_epochs", type=int, default=100)
parser.add_argument("--num_test_runs", type=int, default=10)
parser.add_argument("--num_varaiational_dropouts", type=int, default=50)
args = parser.parse_args()



class dataset_forming():
    def __init__(self,data_set_folder_original=args.root_dir,data_set_folder=args.root_dir_copy,expt_folder=None,active_train_path=None,original_train_path=None):
        self.expt_folder  = expt_folder
        self.data_set_folder_original  = data_set_folder_original
        self.data_set_folder = data_set_folder
        self.active_train_path  = os.path.join(self.expt_folder,'train')
        self.active_remain_path  = os.path.join(self.expt_folder,'remain')
        self.active_test_path = os.path.join(self.expt_folder,'test')
        
        #self.initial_split()
    
    def create_original_dataset_copy(self):
        self.custom_mkdir(self.data_set_folder)
        os.system("cp -a {}/* {}".format(self.data_set_folder_original,self.data_set_folder)) 
    
    def reset_dir(self):
        os.system("find {} -type f -name '*.tif' -delete".format(self.expt_folder))

    def create_dir(self):
        self.custom_mkdir(self.active_train_path)
        self.custom_mkdir(self.active_remain_path)
        self.custom_mkdir(self.active_test_path)
        class_names = os.listdir(self.data_set_folder_original)
        for class_name in class_names:
            self.custom_mkdir(os.path.join(self.active_remain_path, class_name))
            self.custom_mkdir(os.path.join(self.active_train_path, class_name))
            self.custom_mkdir(os.path.join(self.active_test_path, class_name))




    def initial_split(self):
        class_names = os.listdir(self.data_set_folder)

        for class_name in class_names:
            current_class_path = os.path.join(self.data_set_folder, class_name)
            active_class_path_remain = os.path.join(self.active_remain_path, class_name)
            active_class_path_train = os.path.join(self.active_train_path, class_name)
            active_class_path_test = os.path.join(self.active_test_path, class_name)
            os.system("shuf -zn562 -e {}/*.tif | xargs -0 mv -t {}".format(current_class_path,active_class_path_remain))
            os.system("shuf -zn63 -e {}/*.tif | xargs -0 mv -t {}".format(current_class_path,active_class_path_test))
            os.system("shuf -zn40 -e {}/*.tif | xargs -0 mv -t {}".format(active_class_path_remain,active_class_path_train))


    def active_learning_split(self,list_remain_images,class_list):
        os.system("mv {} {}".format(os.path.join(self.active_remain_path,class_list,list_remain_images),os.path.join(self.active_train_path, class_list)))


    def custom_mkdir(self,path):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)




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

# def init_kernel(m):
    
#     if isinstance(m, nn.Conv2d):
#         print(m)
#         # Initialize kernels of Conv2d layers as kaiming normal
#         nn.init.kaiming_normal_(m.weight)
#         # Initialize biases of Conv2d layers at 0
#         nn.init.zeros_(m.bias)
    
#     if isinstance(m, nn.Linear):
#         print(m)
#         # Initialize kernels of Conv2d layers as kaiming normal
#         nn.init.xavier_uniform_(m.weight)
#         # Initialize biases of Conv2d layers at 0
#         nn.init.zeros_(m.bias)



# class model_ara(nn.Module):
  
#     def __init__(self):
#         super(model_ara, self).__init__()
#         self.conv = nn.Sequential(nn.Conv2d(3,64,kernel_size=7,stride=4,padding=2),nn.BatchNorm2d(64),nn.LeakyReLU(0.1),nn.MaxPool2d(2))
#         self.res = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.LeakyReLU(0.1))
#         self.avgpool = nn.AvgPool2d(2)
#         self.dropout_layer = nn.Sequential(nn.Linear(64,32),nn.LeakyReLU(0.1),nn.Dropout(0.5),nn.Linear(32,8))
#         self.softmax = nn.Softmax()
#         self.conv.apply(init_kernel)
#         self.res.apply(init_kernel)
#         self.dropout_layer.apply(init_kernel)
        
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)    


#     def forward(self,input):
#         x = self.conv(input) 
#         #--------------------------------------------------#


#         #------------Residual Block1------------------------#
#         for i in range(4):
#             x1 = self.res(x)
#             x = x+x1

#         x = self.avgpool(x)
        
        
#         #--------Aux_output (dropout1)--------------------------#
#         x_aux = F.adaptive_avg_pool2d(x, (1, 1))
#         x_aux = x_aux.view(-1,x_aux.size(1)*x_aux.size(2)*x_aux.size(3))
#         x_aux = self.dropout_layer(x_aux)

        



#         #-----------Residual Block2-----------------------#
#         for i in range(3):
#             x1 = self.res(x)
#             x = x+x1

#         x = self.avgpool(x)

        
#         #--------Actual_output--------------#
#         x_main = F.adaptive_avg_pool2d(x, (1, 1))
#         x_main = x_main.view(-1,x_main.size(1)*x_main.size(2)*x_main.size(3))
#         x_main = self.dropout_layer(x_main)
#         return x_aux,x_main 

def make_weights_for_balanced_classes(images, nclasses): 
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 
    
    
class get_dataloader():
    def __init__(self,dataroot,is_train=None,is_remain=None):
        self.dataroot = dataroot
        self.is_train = is_train
        self.is_remain = is_remain
        self.dataloader_specs()

    def dataloader_specs(self):
        if self.is_train:
            self.dataset = ImageFolder_Mod(root= self.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(224),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomVerticalFlip(p=0.5),
                                   transforms.RandomRotation(90),
                                   transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                   ]))
            
            # For unbalanced dataset we create a weighted sampler                       
            weights = make_weights_for_balanced_classes(self.dataset.imgs, len(self.dataset.classes))                                                                
            weights = torch.DoubleTensor(weights)                                       
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                

            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=32, sampler=sampler, num_workers=8)

        if self.is_remain:
            
            self.dataset = ImageFolder_Mod(root= self.dataroot,
                   transform=transforms.Compose([
                       transforms.Resize(224),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                       ]))            
            
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=128, shuffle=False, num_workers=16)


        else:
            self.dataset = ImageFolder_Mod(root=self.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                   ]))
            self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=128, shuffle=False, num_workers=16)


        
def compute_table_entropy(table):
        EPS = 1e-6
        table = np.clip(table, EPS, 1 - EPS)
        return (-table * np.log(table)).sum(axis=1)
        


def train_test(model_obj,n_epochs,dataloaders_train,criterion,dataloaders_test):
    y_train = []
    path = []
    global step_active
    global acc_expt
    global step_expt
    global best_model_track
    step_active += 1
    model = copy.deepcopy(model_obj)
    model = model.to(device)
    best_acc = 0

    loss_fn = criterion

    optimizer = optim.Adam(model.parameters(),lr=0.00005, weight_decay=0.005)
    scheduler_val = optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,mode='max',patience=5)
    scheduler_train = optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,mode='min',patience=5)

    monitor_value_out = 0
    monitor_value_aux = 0
    monitor_count = 0
    acc_epoch = []
    loss_epoch = []
    for epoch in range(n_epochs):
        start_time = time.time()

        running_loss_aux_train=0
        running_loss_out_train=0
        running_corrects_aux_train=0
        running_corrects_out_train=0

        model.train()
        length_dataset_train=0
        lenth_dataset_test=0
        running_loss=0
        running_corrects=0
        for j, data in enumerate(dataloaders_train):
            images = data[0]
            length_dataset_train+=images.shape[0]
            images=images.to(device)
            labels = data[1]
            labels=labels.to(device)
            optimizer.zero_grad()        
            with torch.set_grad_enabled(True):

                outputs=model(images)


                _,predictions = torch.max(F.softmax(outputs),1)
                
    
                    
                loss=criterion(outputs,labels)
                

                loss.backward()
                optimizer.step()
            running_loss+=loss.item()*images.size(0)
            running_corrects+=torch.sum(predictions == labels.data)


        epoch_loss_train=running_loss/length_dataset_train
        epoch_acc_train=running_corrects.double()/length_dataset_train
 
        model.eval()
        running_loss_test=0
        running_corrects_test=0
        for j ,data in enumerate(dataloaders_test):
            images = data[0]
            lenth_dataset_test+=images.shape[0]
            images=images.to(device)
            #print(images)
            labels = data[1]
            labels=labels.to(device)
            optimizer.zero_grad()        
            with torch.set_grad_enabled(False):

                outputs=model(images)


                _,predictions = torch.max(F.softmax(outputs),1)
                

                loss = criterion(outputs,labels)
                
            running_loss_test+=loss.item()*images.size(0)
            running_corrects_test+=torch.sum(predictions == labels.data)
            
        epoch_loss_test=running_loss_test/lenth_dataset_test
        epoch_acc_test=running_corrects_test.double()/lenth_dataset_test
        scheduler_val.step(epoch_acc_test)
        if epoch_acc_test>best_acc:
            best_acc=epoch_acc_test
            best_model_wts=copy.deepcopy(model.state_dict())
            #best_model_track[best_acc] = copy.deepcopy(model.state_dict())
            #torch.save(model.state_dict(),'/home/ashishmenon/ARA/Pytorch_imp/trained_acc_fold_{}.pth'.format(i))


        #scheduler.step(loss)
        acc_epoch.append(epoch_acc_test.item())
        loss_epoch.append(epoch_loss_test)
        print('epoch {} Loss_Train: {:.4f} Loss_Test: {:.4f} Acc_test: {:.4f} '.format(
            epoch, epoch_loss_train, epoch_loss_test, epoch_acc_test),flush=True) 
    

        writer.add_scalars('Loss',{'test_loss{}'.format(step_active):epoch_loss_test}, epoch)
        writer.close()
        writer.add_scalars('Acc',{'Test_acc{}'.format(step_active):epoch_acc_test}, epoch)
        writer.close()
        if (epoch_acc_test==monitor_value_out ):
            monitor_count+=1
        else:
            monitor_count=0
            monitor_value_out = epoch_acc_test
            
        if monitor_count>5:
            print('SATURATED for more than 5 epochs returning model')
            try:
                model.load_state_dict(best_model_wts)
                break
            except:
                return model_obj
    
        
        
    model.load_state_dict(best_model_wts)
    acc_expt.append(sum(acc_epoch)/len(acc_epoch))
    step_expt.append(step_active)
    writer.add_scalar('Acc_steps',max(acc_epoch), step_active)
    writer.close()
    writer.add_scalar('Loss_steps',min(loss_epoch), step_active)
    writer.close()
    print('***************************** \n')
    print(list(zip(acc_expt,step_expt)))
    return model

    
            
def active_learning_entropy_tranfer(model_obj , testruns , variationalruns , measure_name, dataloader_active, dataloader_train, data_mover):  
    model = copy.deepcopy(model_obj)
    model = model.to(device)
    all_u = []
    all_p = []
        
    for test in range(testruns):
        entropy_list ={}
        classes_list = {}
        classes = dataloader_active.dataset.classes
        overall_pred = {}
        recall=[]
        fbeta=[]
        running_corrects=0
        
        for v in range(variationalruns):
            model.eval()
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
            length_dataset_active=0
            for j,data in enumerate(dataloader_active):
                images = data[0]
                length_dataset_active+=images.shape[0]
                reqd_list = list(data[3])
                labels = data[1].to(device)
                index = list(data[1].cpu().numpy())
                y_true = data[1].cpu().view(-1).numpy()
                input_to_model = data[0].to(device)
                with torch.no_grad():
                    pred_actual = model(input_to_model)
                    pred_actual = F.softmax(pred_actual)
                    _,predictions=torch.max(pred_actual,1)

                if v==0 and j==0:
                    predictions_over_variations = pred_actual.cpu().numpy().copy()
                    
                else:
                    predictions_over_variations = np.concatenate((predictions_over_variations,pred_actual.cpu().numpy().copy()),axis=0)
                    
                    

                for k in range(len(data[2])):
                    try:
                        
                        classes_list[reqd_list[k]] = classes[index[k]]
                        overall_pred[reqd_list[k]] += pred_actual.detach().cpu()[k].numpy()


                    except Exception:
                        classes_list[reqd_list[k]] = classes[index[k]]
                        overall_pred[reqd_list[k]] = pred_actual.detach().cpu()[k].numpy()
           
                
                
        
        predictions_over_variations = predictions_over_variations.reshape(variationalruns,-1,8)
        mean_entropy = np.array([compute_table_entropy(table) for table in predictions_over_variations]).mean(axis=0)
        entropy_of_mean = compute_table_entropy(predictions_over_variations.mean(axis=0))
        predictions_overall = list(predictions_over_variations.mean(axis=0))
        
        if measure_name == 'Entropy':
            uncertainity = entropy_of_mean # Entropy
        elif measure_name == 'BALD':
            uncertainity =  entropy_of_mean - mean_entropy # BALD
        
        all_u.append(uncertainity)
        all_p.append(predictions_overall)

    overall_uncertainity = np.array(all_u).mean(axis=0)
    overall_prediction = np.array(all_p).mean(axis=0)
    df1 = pd.DataFrame(list(classes_list.keys()))
    df2 = pd.DataFrame(list(classes_list.values()))
    df3 = pd.DataFrame(list(overall_uncertainity))
    df4 = pd.DataFrame(list(overall_prediction))
    df = pd.DataFrame(np.column_stack((df1,df2,df3,df4)))
    df.columns = ['Image Name','actual_class','uncertainity' ,classes[0], classes[1], classes[2], classes[3], classes[4], classes[5], classes[6], classes[7]]
    a = df.columns[3:]
    predictions = np.array(df.iloc[:,3:])
    df['predicted_Class'] = list(a[predictions.argmax(axis=1)])
    df = df.iloc[:,0:3]
    df['predicted_class'] = list(a[predictions.argmax(axis=1)])
         
    df = df.sort_values(by='uncertainity',ascending=False)
    df.to_excel('output{}.xlsx'.format(step_active), engine='xlsxwriter')
    df[0:160].apply(lambda x: data_mover.active_learning_split(x['Image Name'] ,x['actual_class']), axis=1)
    
    
    

class run_model():
    def __init__(self,expt_data,expt,train_epochs=1,test_runs=1,variation_dropout_calls=1,uncertainity_measure='Entropy'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.expt = expt
        self.expt_data = expt_data
        self.testruns = testruns
        self.variation_dropout_calls = variation_dropout_calls
        self.uncertainity_measure = uncertainity_measure
        num_ftrs = 1000
        self.model = nn.Sequential(models.squeezenet1_1(pretrained=True), nn.Linear(num_ftrs,8))
        self.model = self.model.to(device)        
        self.criterion = nn.CrossEntropyLoss()
        self.train_epochs = train_epochs
    
    def train(self):
        global step_active
        data_mover = dataset_forming(expt_folder=self.expt_data)
        data_mover.reset_dir() #Remove all images files in the train remain and valid once in 3 steps of active learning
        data_mover.create_original_dataset_copy() #Make the copy of original dataset once in every 3 steps learning
        data_mover.create_dir()
        data_mover.initial_split()#Initial split only once for every 3 steps of active learning

        dataloader_train = get_dataloader(dataroot = data_mover.active_train_path , is_train = True )
        dataloader_remain = get_dataloader(dataroot = data_mover.active_remain_path, is_remain = True )
        dataloader_test =  get_dataloader(dataroot = data_mover.active_test_path , is_train = False )
        cnt=0
        while(len(glob.glob(str(data_mover.active_remain_path)+'/*/*.tif')) !=0):
            print(len(glob.glob(str(data_mover.active_remain_path)+'/*/*.tif')))
            self.model = train_test(self.model,self.train_epochs,dataloader_train.data_loader,self.criterion,dataloader_test.data_loader)
            if self.expt == 'Active_Learning':
                active_learning_entropy_tranfer(self.model, self.testruns, self.variation_dropout_calls , self.uncertainity_measure, dataloader_remain.data_loader,dataloader_train.data_loader,data_mover)
            if self.expt == 'Random':
                p = glob.glob('/ssd_scratch/cvit/ashishmenon/ara_random_expts/remain/**/*.tif')
                random.shuffle(p)
                for i in p[0:160]:
                    data_mover.active_learning_split(i.split('/')[7],i.split('/')[6])
                    
                    
            dataloader_train = get_dataloader(dataroot = data_mover.active_train_path , is_train = True )
            dataloader_remain = get_dataloader(dataroot = data_mover.active_remain_path, is_remain = True )
            dataloader_test =  get_dataloader(dataroot = data_mover.active_test_path , is_train = False )
            cnt+=1
            

if __name__=='__main__':

    if not os.path.exists(args.active_learning_dir):
        os.makedirs(args.active_learning_dir)

    if not os.path.exists(args.random_transfer_dir):
        os.makedirs(args.random_transfer_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    rd.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TRAIN_RATIO = 0.9
    step_active=0
    acc_expt = []
    step_expt= []
    best_model_track ={}
    writer = SummaryWriter(log_dir=args.log_dir)
    print(args.active_learning_dir)
    try1 = run_model(args.active_learning_dir, 'Active_Learning' , train_epochs = args.num_train_epochs,test_runs=args.num_test_runs , variational_dropout_calls=args.variation_dropout_calls )
    try1.train()
