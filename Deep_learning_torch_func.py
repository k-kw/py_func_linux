import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import torch
import numpy as np
import torch.nn as nn

import os
import os.path as osp

#mean of list
import statistics

#SSIM
from skimage.metrics import structural_similarity as ssim

#Data Augumentation

#assign Gaussian Noise Layer
class GaussianNoise(nn.Module):
    def __init__(self, stddev, device):
        super().__init__()
        self.stddev = stddev
        self.device = device
    
    def forward(self, x):
        return x + torch.autograd.Variable(torch.randn(x.size()).to(self.device) * self.stddev)


#Mixup
def mixup_data(x, y, device, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    # if use_cuda:
    #     index = torch.randperm(batch_size).cuda()
    # else:
    #     index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)






#Deep learning preprocess Normalization
def tensor_norm_DL(tensor, mean = None, std = None):
  if mean == None:
    mean = torch.mean(tensor)
  if std == None:
    std = torch.std(tensor)
  re_tensor = (tensor - mean)/std
  return re_tensor, mean, std

#Max-Min normalization
def tensor_norm(tensor, max = None, min = None):
  if max == None:
     max = torch.max(tensor)
  if min == None:
    min = torch.min(tensor)
  re_tensor = (tensor - min)/(max - min)
  return re_tensor, max, min



#Create and save confusion-matrix
def create_cfmat(labels,predictions,matrix_save_path,vmax,figsy,figsx,fontsize):
  import seaborn as sns
  from sklearn.metrics import confusion_matrix

  _, ax = plt.subplots(figsize = (figsy, figsx))
  #plt.rcParams["font.size"] = fontsize

  cm = confusion_matrix(labels,predictions)
  sns.heatmap(cm,square=True,cmap='Blues',annot=True,fmt='d',ax=ax,vmax=vmax,vmin=0,annot_kws={"size":fontsize})
  
  plt.savefig(matrix_save_path)





#Displaying learning-curv_ver2
def learning_curv_ver2( fig_w, fig_h, labelfontsize, scalefontsize, tls = None, vls = None, tas = None, vas = None):
    """
    you must prepare vls.
    """
    rcparams_dic = {
        'figure.figsize': (fig_w,fig_h),
        'axes.labelsize': labelfontsize,
        'xtick.labelsize': scalefontsize,
        'ytick.labelsize': scalefontsize,
    }
    plt.rcParams.update(rcparams_dic)

    #if you have list of accuracy late
    if (tas != None) or (vas != None):
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        plt.subplot(1,2,1)
  
    if tls != None:
        plt.plot(range(1, 1 + len(tls)), tls, label="training")
    plt.plot(range(1, 1 + len(vls)), vls, label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if (vas != None) or (tas != None):
        plt.subplot(1,2,2)
        if tas != None:
            plt.plot(range(1, 1 + len(tas)), tas, label="training")
        plt.plot(range(1, 1 + len(vas)), vas, label="validation")
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.legend()

    plt.show()







#Weight decay
#L1
def L1norm(model, alpha, loss):
    l1 = torch.tensor(0., requires_grad=True)
    for w in model.parameters():
        #sum of norm of weight
        l1 = l1 + torch.norm(w, 1)
    loss = loss + alpha*l1
    return loss

#L2
def L2norm(model, lamda, loss):
    l2 = torch.tensor(0., requires_grad=True)
    for w in model.parameters():
        #sum of power of weight
        l2 = l2 + torch.norm(w)**2
    loss = loss + lamda*l2/2
    return loss



#学習終了判定(Early stopping)
#最後からdeciosion_num番目までの平均損失値が、decision_meanを下回ると終了させる
def endmean(val_loss_list, decision_num, decision_mean, disp_epoch):
    lastloss=val_loss_list[(-1*decision_num):]

    if(statistics.mean(lastloss)<=decision_mean):
        print(f'mean loss from last to decision_num was below decision_mean, so ended at epoch{disp_epoch}.\n')
        endflg=True
    else:
        endflg=False
    return endflg


#評価・訓練関数
#Deep Learning validation(image classification)
def val_model(dataloader, model, device, lossfunc, predict_label_list_true):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    val = []

    if (predict_label_list_true == True):
        predict_list = []
        label_list = []
  
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            loss = lossfunc(outputs, labels)
            val_loss += loss.item()
            #_, predicted = torch.max(outputs.data, 1)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (predict_label_list_true == True):
                for i in range(len(predicted)):
                    tmp = int(predicted[i])
                    predict_list.append(tmp)
                for i in range(len(labels)):
                    tmp = int(labels[i])
                    label_list.append(tmp)

        acc = float(correct)/total
        val_loss = val_loss/len(dataloader.dataset)
        val.append(acc)
        val.append(val_loss)

        if (predict_label_list_true == True):
            val.append(predict_list)
            val.append(label_list)

    return val

#Deep Learning(image classification)
#MIXUP
def train_model_mixup(dlt, dlv, model, lossfunc, optimizer, maxepochs, device, 
    mean_or_improve = None, decision_num = 10, decision_mean = None, #学習終了条件, 
    #meanのとき最後からdicision_numまでの平均lossがdecision_meanを下回ったら終了, improveのときdecision_numだけ改善がなければ終了
    L1 = False, alpha = None, L2 = False, lamda = None, #正則化
    mixalpha = 1.0, #MIXUP alpha
    scheduler = None, #学習率スケジューラ
    modelsavedir = None, saveepoch = 100 #モデル保存、ディレクトリ、エポック
    ):


    t1=time.time()
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    endflg=False

    if(mean_or_improve=="improve"):
        notimpv_cnt=0

    for epoch in range(maxepochs):
        model.train()
        for dat_train in dlt:
            inputs, labels = dat_train
            inputs, labels = inputs.to(device), labels.to(device)
            
            #mixup
            inputs, laba, labb, mixlamda = mixup_data(inputs, labels, device, mixalpha)            
            inputs, laba, labb = Variable(inputs),  Variable(laba), Variable(labb)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss_mix_func = mixup_criterion(laba, labb, mixlamda)
            loss = loss_mix_func(lossfunc, outputs)

            #正則化(weight decay)
            if L2:
                loss = L2norm(model, lamda, loss)
            elif L1:
                loss = L1norm(model, alpha, loss)


            loss.backward()
            optimizer.step()
        
        if(scheduler != None):
            scheduler.step()

        #modelの評価 
        val_val = val_model(dlv, model, device, lossfunc, False)
        val_train = val_model(dlt, model, device, lossfunc, False)
        train_loss_list.append(val_train[1])
        val_loss_list.append(val_val[1])
        train_acc_list.append(val_train[0])
        val_acc_list.append(val_val[0])

        
        #最小損失の更新
        if(epoch==0):
            notimpv_cnt=0
            minloss=val_val[0]
            flg=True
        elif(minloss>val_val[0]):
            notimpv_cnt=0
            minloss=val_val[0]
            flg=True
        else:
            notimpv_cnt+=1
            flg=False

        #modelを保存
        if (modelsavedir != None):
            if(epoch+1==saveepoch):
                os.makedirs(modelsavedir, exist_ok=True)
                mdpath = osp.join(modelsavedir, "sota.pth")
                torch.save(model.state_dict(), mdpath)
            elif(epoch+1>saveepoch and flg):
                mdpath = osp.join(modelsavedir, "sota.pth")
                torch.save(model.state_dict(), mdpath)


        print(f'---------------------------epoch{epoch+1}------------------------------')
        print(f'train_loss{val_train[1]:.4f} ,train_acc{val_train[0]:.4f}, val_loss{val_val[1]:.4f} ,val_acc{val_val[0]:.4f}')
        t2=time.time()
        caltime=(t2-t1)/60
        print(f'epochtime:{caltime:.4f} minutes')
        t1=time.time()

        #終了判定
        #平均を指定した場合
        if(mean_or_improve=="mean"):
            endflg = endmean(val_loss_list, decision_num, decision_mean, epoch+1)
            if(endflg):
                break
            
        #改善を指定した場合
        elif(mean_or_improve=="improve" and epoch>=1):
            if(notimpv_cnt>=decision_num):
                print(f'loss was not improved in decision_num, so ended at epoch{epoch+1}.\n')
                endflg=True
            if(endflg):
                break       

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list




#Deep Learning(image classification)
def train_model_ver3(dlt, dlv, model, lossfunc, optimizer, maxepochs, device, 
    mean_or_improve = None, decision_num = 10, decision_mean = None, #学習終了条件, 
    #meanのとき最後からdicision_numまでの平均lossがdecision_meanを下回ったら終了, improveのときdecision_numだけ改善がなければ終了
    L1 = False, alpha = None, L2 = False, lamda = None,
    scheduler = None, 
    gausnoise = False, stddev = 0.01,
    modelsavedir = None, saveepoch = 100):
    """
    transfer displaying learning_curv from this function.
    select weight decay option
    L1 : bool
    L2 : bool
    alpha and lamda are parameter
    """
    t1=time.time()
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    endflg=False


    if(mean_or_improve=="improve"):
        notimpv_cnt=0

    #ガウシアンノイズ付与レイヤを定義
    if (gausnoise):
        gn = GaussianNoise(stddev, device)

    for epoch in range(maxepochs):
        model.train()
        for dat_train in dlt:
            inputs, labels = dat_train
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            #ガウシアンノイズを付与
            if (gausnoise):
                inputs = gn(inputs)


            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossfunc(outputs, labels)

            #正則化(weight decay)
            if L2:
                loss = L2norm(model, lamda, loss)
            elif L1:
                loss = L1norm(model, alpha, loss)


            loss.backward()
            optimizer.step()

        if(scheduler != None):
            scheduler.step()

        val_val = val_model(dlv, model, device, lossfunc, False)
        val_train = val_model(dlt, model, device, lossfunc, False)
        train_loss_list.append(val_train[1])
        val_loss_list.append(val_val[1])
        train_acc_list.append(val_train[0])
        val_acc_list.append(val_val[0])

        #最小損失の更新
        if(epoch==0):
            notimpv_cnt=0
            minloss=val_val[1]
            flg=True
        elif(minloss>val_val[1]):
            notimpv_cnt=0
            minloss=val_val[1]
            flg=True
        else:
            notimpv_cnt+=1
            flg=False

        #modelを保存
        if (modelsavedir != None):
            if(epoch+1==saveepoch):
                os.makedirs(modelsavedir, exist_ok=True)
                mdpath = osp.join(modelsavedir, "sota.pth")
                torch.save(model.state_dict(), mdpath)
            elif(epoch+1>saveepoch and flg):
                mdpath = osp.join(modelsavedir, "sota.pth")
                torch.save(model.state_dict(), mdpath)
        
        #進捗表示
        print(f'----------------------------epoch{epoch+1}------------------------------')
        print(f'train_loss{val_train[1]:.4f} ,train_acc{val_train[0]:.4f}, val_loss{val_val[1]:.4f} ,val_acc{val_val[0]:.4f}')
        t2=time.time()
        caltime=(t2-t1)/60
        print(f'epochtime:{caltime:.4f} minutes')
        t1=time.time()

        #終了判定
        #平均を指定した場合
        if(mean_or_improve=="mean"):
            endflg = endmean(val_loss_list, decision_num, decision_mean, epoch+1)
            if(endflg):
                break
            
        #改善を指定した場合
        elif(mean_or_improve=="improve" and epoch>=1):
            if(notimpv_cnt>=decision_num):
                print(f'loss was not improved in decision_num, so ended at epoch{epoch+1}.\n')
                endflg=True
            if(endflg):
                break

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list








#Deep Learning validation(decode)
def val_decode_model(dataloader, model, device, lossfunc):
  model.eval()
  val_loss = 0
  val = []
  
  with torch.no_grad():
    for inputs, origin in dataloader:
      inputs, origin = inputs.to(device), origin.to(device)
      inputs, origin = Variable(inputs), Variable(origin)
      outputs = model(inputs)
      loss = lossfunc(outputs, origin)
      val_loss += loss.item()

    
    val_loss = val_loss/len(dataloader.dataset)
    val.append(val_loss)
  
  return val


#Deep Learning(decode)
def train_decode_model_mixup(dlt, dlv, model, lossfunc, optimizer, maxepochs, device, 
    mean_or_improve = None, decision_num = 10, decision_mean = None, #学習終了条件, 
    #meanのとき最後からdicision_numまでの平均lossがdecision_meanを下回ったら終了, improveのときdecision_numだけ改善がなければ終了
    mixalpha = 1.0, scheduler = None, 
    modelsavedir = None, saveepoch = 100):
    t1=time.time()
    train_loss_list=[]
    val_loss_list=[]
    endflg=False

    if(mean_or_improve=="improve"):
        notimpv_cnt=0

    for epoch in range(maxepochs):
        model.train()
        for dat_train in dlt:
            inputs, origin = dat_train
            inputs, origin = inputs.to(device), origin.to(device)
            # inputs, origin = Variable(inputs), Variable(origin)
            inputs, laba, labb, mixlamda = mixup_data(inputs, origin, device, mixalpha)
            inputs, laba, labb = Variable(inputs), Variable(laba), Variable(labb)

            optimizer.zero_grad()
            outputs = model(inputs)

            # loss = lossfunc(outputs, origin)
            loss_mix_func = mixup_criterion(laba, labb, mixlamda)
            loss = loss_mix_func(lossfunc, outputs)
            loss.backward()
            optimizer.step()
        
        if(scheduler != None):
            scheduler.step()

        
        val_val = val_decode_model(dlv, model, device, lossfunc)
        val_train = val_decode_model(dlt, model, device, lossfunc)
        train_loss_list.append(val_train[0])
        val_loss_list.append(val_val[0])

        #最小損失の更新
        if(epoch==0):
            notimpv_cnt=0
            minloss=val_val[0]
            flg=True
        elif(minloss>val_val[0]):
            notimpv_cnt=0
            minloss=val_val[0]
            flg=True
        else:
            notimpv_cnt+=1
            flg=False

        #modelを保存
        if (modelsavedir != None):
            if(epoch+1==saveepoch):
                os.makedirs(modelsavedir, exist_ok=True)
                mdpath = osp.join(modelsavedir, "sota.pth")
                torch.save(model.state_dict(), mdpath)
            elif(epoch+1>saveepoch and flg):
                mdpath = osp.join(modelsavedir, "sota.pth")
                torch.save(model.state_dict(), mdpath)
        
        #進捗表示
        t2=time.time()
        caltime=(t2-t1)/60
        print(f'--------------------------epoch{epoch+1}--------------------------------')
        print(f'epochtime:{caltime:.4f} minutes\n, train_loss:{val_train[0]*1000:.4f}, val_loss:{val_val[0]*1000:.4f}')
        t1=time.time()

        #終了判定
        #平均を指定した場合
        if(mean_or_improve=="mean"):
            endflg = endmean(val_loss_list, decision_num, decision_mean, epoch+1)
            if(endflg):
                break
            
        #改善を指定した場合
        elif(mean_or_improve=="improve" and epoch>=1):
            if(notimpv_cnt>=decision_num):
                print(f'loss was not improved in decision_num, so ended at epoch{epoch+1}.\n')
                endflg=True
            if(endflg):
                break

    return train_loss_list, val_loss_list



#Deep Learning(decode)
def train_decode_model_ver2(dlt, dlv, model, lossfunc, optimizer, maxepochs, device, scheduler = None, 
    mean_or_improve = None, decision_num = 10, decision_mean = None, #学習終了条件, 
    #meanのとき最後からdicision_numまでの平均lossがdecision_meanを下回ったら終了, improveのときdecision_numだけ改善がなければ終了
    gausnoise = False, stddev = 0.01, 
    modelsavedir = None, saveepoch = 100):
    """
    transfer displaying learning_curv from this function.
    """
    t1=time.time()
    train_loss_list=[]
    val_loss_list=[]
    endflg=False

    #ガウシアンノイズ付与レイヤを定義
    if (gausnoise):
        gn = GaussianNoise(stddev, device)
    
    if(mean_or_improve=="improve"):
        notimpv_cnt=0

    for epoch in range(maxepochs):
        model.train()
        for dat_train in dlt:
            inputs, origin = dat_train
            inputs, origin = inputs.to(device), origin.to(device)
            inputs, origin = Variable(inputs), Variable(origin)

            #ガウシアンノイズを付与
            if (gausnoise):
                inputs = gn(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossfunc(outputs, origin)
            loss.backward()
            optimizer.step()

        
        if(scheduler != None):
            scheduler.step()

        val_val = val_decode_model(dlv, model, device, lossfunc)
        val_train = val_decode_model(dlt, model, device, lossfunc)
        train_loss_list.append(val_train[0])
        val_loss_list.append(val_val[0])

        #最小損失の更新
        if(epoch==0):
            notimpv_cnt=0
            minloss=val_val[0]
            flg=True
        elif(minloss>val_val[0]):
            notimpv_cnt=0
            minloss=val_val[0]
            flg=True
        else:
            notimpv_cnt+=1
            flg=False

        #modelを保存
        if (modelsavedir != None):
            if(epoch+1==saveepoch):
                os.makedirs(modelsavedir, exist_ok=True)
                mdpath = osp.join(modelsavedir, "sota.pth")
                torch.save(model.state_dict(), mdpath)
            elif(epoch+1>saveepoch and flg):
                mdpath = osp.join(modelsavedir, "sota.pth")
                torch.save(model.state_dict(), mdpath)

        #進捗表示
        t2=time.time()
        caltime=(t2-t1)/60
        print(f'-----------------------------epoch{epoch+1}--------------------------------')
        print(f'epochtime:{caltime:.4f} minutes\n train_loss:{val_train[0]*1000:.4f}, val_loss:{val_val[0]*1000:.4f}')
        t1=time.time()

        #終了判定
        #平均を指定した場合
        if(mean_or_improve=="mean"):
            endflg = endmean(val_loss_list, decision_num, decision_mean, epoch+1)
            if(endflg):
                break
            
        #改善を指定した場合
        elif(mean_or_improve=="improve" and epoch>=1):
            if(notimpv_cnt>=decision_num):
                print(f'loss was not improved in decision_num, so ended at epoch{epoch+1}.\n')
                endflg=True
            if(endflg):
                break

    return train_loss_list, val_loss_list





#convert tensor into numpy, 255までにクリップする
def tensor_to_numpy(input_tensor, normTrue):
  output_numpy = input_tensor.to('cpu').detach().numpy().copy()
  if normTrue:
    output_numpy *= 255
  output_numpy = np.round(output_numpy)
  output_numpy = np.clip(output_numpy, a_min = 0, a_max = 255)
  output_numpy = output_numpy.astype(np.uint8)
  return output_numpy

#test a decode model and check a output-image
def test_decode_model_and_check_img_ver3(dataloader, img_width, img_height, model, device, figwidth, figheighgt, from_num, to_num, save_dir_path,
datanorm = False, correctimgnorm = False):
    """
    datanorm: Bool 
    正規化した復元画像を出力するよう訓練したモデルはTrue
    画素値をそのまま出力するよう訓練したモデルはFalse
    """
    import numpy as np
    import cv2
    #評価モード
    model.eval()
    img_num = 1
    displength=to_num-from_num+1

    
    plt.rcParams["figure.figsize"] = (figwidth, figheighgt)
    fig = plt.figure()

    #PSNR配列
    psnrs = []
    #SSIM配列
    ssims = []

    with torch.no_grad():
        for inputs, origin in dataloader:
            inputs, origin = inputs.to(device), origin.to(device)
            inputs = Variable(inputs)
            outputs = model(inputs)
            for i in range(len(outputs)):
                output_img_array = tensor_to_numpy(outputs[i], datanorm)
                output_img_array = np.resize(output_img_array,(img_height,img_width))

                origin_img_array = tensor_to_numpy(origin[i], correctimgnorm)
                origin_img_array = np.resize(origin_img_array,(img_height,img_width))

                #PSNR算出
                psnr = cv2.PSNR(output_img_array, origin_img_array)
                psnrs.append(psnr)

                #SSIM算出
                ssimvalue = ssim(output_img_array, origin_img_array)
                ssims.append(ssimvalue)

                if from_num <= img_num and img_num <= to_num:    
                    plt.subplot(2,displength,img_num-from_num+1)
                    plt.imshow(origin_img_array)
                    plt.gray()
                    plt.subplot(2,displength,img_num-from_num+1+displength)
                    plt.imshow(output_img_array)
                    plt.gray()
                img_num += 1

    plt.show()
    os.makedirs(save_dir_path, exist_ok=True)
    savepath = osp.join(save_dir_path, "restore_result.jpg")
    fig.savefig(savepath)

    return psnrs, ssims



if __name__=='__main__':
    print('Functions related Deep Learning ')
