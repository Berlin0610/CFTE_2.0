import os, sys
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.keypoint_detector import KPDetector
import time
import random
import pandas as pd
import collections
import itertools
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *

import cv2


def load_checkpoints(scalenumber,config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    kp_detector.load_state_dict(checkpoint['kp_detector'+'_'+str(scalenumber)],strict=False) ####
    
    if not cpu:
        kp_detector = DataParallelWithCallback(kp_detector)
    kp_detector.eval()

    return  kp_detector


def RawReader_planar(FileName, ImgWidth, ImgHeight, NumFramesToBeComputed):
    
    f   = open(FileName, 'rb')
    frames  = NumFramesToBeComputed
    width   = ImgWidth
    height  = ImgHeight
    data = f.read()
    f.close()
    data = [int(x) for x in data]

    data_list=[]
    n=width*height
    for i in range(0,len(data),n):
        b=data[i:i+n]
        data_list.append(b)
    x=data_list

    listR=[]
    listG=[]
    listB=[]
    for k in range(0,frames):
        R=np.array(x[3*k]).reshape((width, height)).astype(np.uint8)
        G=np.array(x[3*k+1]).reshape((width, height)).astype(np.uint8)
        B=np.array(x[3*k+2]).reshape((width, height)).astype(np.uint8)
        listR.append(R)
        listG.append(G)
        listB.append(B)
    return listR,listG,listB

def splitlist(list): 
    alist = []
    a = 0 
    for sublist in list:
        try: 
            for i in sublist:
                alist.append (i)
        except TypeError: 
            alist.append(sublist)
    for i in alist:
        if type(i) == type([]):
            a =+ 1
            break
    if a==1:
        return printlist(alist) 
    if a==0:
        return alist  


if __name__ == "__main__":
   
    parser = ArgumentParser()

    frames=250 #250
  
    
    scalenumber=4 # 4/6/8
    
    scalefactor=1/scalenumber
    width=64*scalenumber
    height=64*scalenumber 
    
    Qstep=40 ##16

 
    modeldir = 'temporal_adaptive_2ref'  
    config_path='./checkpoint/'+modeldir+'/vox-256.yaml'
    checkpoint_path='./checkpoint/'+modeldir+'/0099-checkpoint.pth.tar'
    
    kp_detector = load_checkpoints(scalenumber,config_path, checkpoint_path, cpu=False)
        
    
    
#     seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020',
#              '021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040']    
    
    
#     qplist=['32','37','42','47','52'] 


    seqlist=['001']
    qplist=['32']
    
    model_dirname='./experiment/'+modeldir+"/"+"Qstep_"+str(Qstep)+"/"
    
    
    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):             
            
            original_seq='./Testrgb/'+str(width)+'/'+str(seq)+'_'+str(width)+'x'+str(width)+'.rgb'  #'_1_8bit.rgb'       
            
            
            listR,listG,listB=RawReader_planar(original_seq,width, height,frames)
        
            driving_kp =model_dirname+'/kp/'+str(width)+'/'+seq+'_QP'+str(QP)+'/'    
            os.makedirs(driving_kp,exist_ok=True)     # the frames to be compressed by vtm                 


            dir_enc =model_dirname+'/enc/'+str(width)+'/'+seq+'_QP'+str(QP)+'/'
            os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm                 
            
            f_org=open(original_seq,'rb')
            ref_rgb_list=[]
            
            ref_kp_list=[]
            ref_multiper_list=[] 
            
            seq_kp_integer=[]
            
            start=time.time() 
            
            sum_bits = 0
            for frame_idx in range(0, frames):            
                
                frame_idx_str = str(frame_idx).zfill(4)   
                
                img_input=np.fromfile(f_org,np.uint8,3*height*width).reshape((3,height,width))  #RGB
                
                if frame_idx in [0]:      # I-frame                        
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.rgb','w')
                    img_input.tofile(f_temp)
                    f_temp.close()
                                                           
                    os.system("./vtm/encode.sh "+dir_enc+'frame'+frame_idx_str+" "+QP+" "+str(width)+" "+str(height))   ########################

                    bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits
                    
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_rec.rgb','rb')
                    img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                    ref_rgb_list.append(img_rec)
                    
                    
                    img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                                      
                    with torch.no_grad(): 
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                        reference = reference.cuda()    # require GPU                        
                        kp_reference, multiscale_perceptural_representation_reference = kp_detector(reference,scalefactor) ################
            
                        ref_kp_list.append(kp_reference)
                        ref_multiper_list.append(multiscale_perceptural_representation_reference)    

                        kp_value = kp_reference['value']

                        kp_value_list = kp_value.tolist()
                        kp_value_list = str(kp_value_list)
                        kp_value_list = "".join(kp_value_list.split())

                        with open(driving_kp+'/frame'+frame_idx_str+'.txt','w')as f:
                            f.write(kp_value_list)  
                            
                        kp_value_frame=json.loads(kp_value_list)
                        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
                        seq_kp_integer.append(kp_value_frame)                                         
                        
                        

                else:
                    
                    interframe = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                    interframe = resize(interframe, (width, height))[..., :3]

                    with torch.no_grad(): 
                        interframe = torch.tensor(interframe[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                        interframe = interframe.cuda()    # require GPU                  
                        kp_interframe, multiscale_perceptural_representation_interframe = kp_detector(interframe,scalefactor) ################
                        
                        
                        ###Quantization
                        kp_value = kp_interframe['value']

                        kp_value_list = kp_value.tolist()
                        kp_value_list = str(kp_value_list)
                        kp_value_list = "".join(kp_value_list.split())
                        
                        with open(driving_kp+'/frame'+frame_idx_str+'.txt','w')as f:
                            f.write(kp_value_list)  
                            
                        kp_value_frame=json.loads(kp_value_list)
                        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
                        seq_kp_integer.append(kp_value_frame)            
                        
                        
                        
            rec_sem=[]
            for frame in range(1,frames):
                frame_idx = str(frame).zfill(4)
                if frame==1:
                    rec_sem.append(seq_kp_integer[0])
                    
                    ### residual
                    kp_difference=(np.array(seq_kp_integer[frame])-np.array(seq_kp_integer[frame-1])).tolist()
                    ## quantization

                    #kp_difference=[int((i+1)*Qstep) for i in kp_difference]
                    kp_difference=[i*Qstep for i in kp_difference]
                    kp_difference= list(map(round, kp_difference[:]))

                    frame_idx = str(frame).zfill(4)
                    bin_file=driving_kp+'/frame'+str(frame_idx)+'.bin'
                    
                    final_encoder_expgolomb(kp_difference,bin_file)     

                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits          
                    
                    #### decoding for residual
                    res_dec = final_decoder_expgolomb(bin_file)
                    res_difference_dec = data_convert_inverse_expgolomb(res_dec)   
                    
                    ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

                    res_difference_dec=[i/Qstep for i in res_difference_dec]
                    rec_semantics=(np.array(res_difference_dec)+np.array(rec_sem[frame-1])).tolist()
                    rec_sem.append(rec_semantics)
                    
                else:
                    
                    
                    ### residual
                    kp_difference=(np.array(seq_kp_integer[frame])-np.array(rec_sem[frame-1])).tolist()
                    
                    ## quantization
                    kp_difference=[i*Qstep for i in kp_difference]
                    kp_difference= list(map(round, kp_difference[:]))

                    frame_idx = str(frame).zfill(4)
                    bin_file=driving_kp+'/frame'+str(frame_idx)+'.bin'
                    final_encoder_expgolomb(kp_difference,bin_file)     

                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits          
                    
                    #### decoding for residual
                    res_dec = final_decoder_expgolomb(bin_file)
                    res_difference_dec = data_convert_inverse_expgolomb(res_dec)   
                    
                    ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame
                    res_difference_dec=[i/Qstep for i in res_difference_dec]
                    rec_semantics=(np.array(res_difference_dec)+np.array(rec_sem[frame-1])).tolist()                  
                    rec_sem.append(rec_semantics)

            end=time.time()
            print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))                        
                        
                        
                        
