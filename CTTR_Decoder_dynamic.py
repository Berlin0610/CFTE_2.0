import os, sys
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
import json
import time
import cv2
from arithmetic.value_decoder import *
from flowvisual import *

def rgb4442yuv420(dir1, dir2):    
    os.makedirs(dir2,exist_ok=True)
    files=glob.glob(os.path.join(dir1,'*.rgb'))
    files.sort()
    for file in files:  
        f=open(file,'rb')
        file_name=file.split('/')[-1]
        file_name=os.path.splitext(file_name)[0]
        tar_path=dir2+file_name+'.yuv'
        yuvfile=open(tar_path, mode='w')

        width=256
        height=256
        framenum=50
        for idx in range(framenum):
            R=np.fromfile(f,np.uint8,width*height).reshape((height,width))
            G=np.fromfile(f,np.uint8,width*height).reshape((height,width))   
            B=np.fromfile(f,np.uint8,width*height).reshape((height,width))
            image_rgb = np.zeros((height,width,3), 'uint8')
            image_rgb[..., 0] = R
            image_rgb[..., 1] = G
            image_rgb[..., 2] = B
            image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
            image_yuv[:,:,0].tofile(yuvfile)
            image_yuv[:,:,1].tofile(yuvfile)
            image_yuv[:,:,2].tofile(yuvfile)
        
    print('done')


def load_checkpoints(scalenumber,config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['common_params'])
    
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'+'_'+str(scalenumber)],strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector'+'_'+str(scalenumber)],strict=False) ####
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


def make_prediction(reference_frame, kp_reference, kp_current, generator, scalefactor,relative=False, adapt_movement_scale=False, cpu=False):
        
    kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
                               kp_driving_initial=kp_reference, use_relative_movement=relative,
                               use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    
    out = generator(reference_frame, scalefactor,kp_reference, kp_norm, reference_frame, kp_reference)
        
    prediction=np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]
    return prediction
    
    


def check_reference(ref_kp_list, kp_current):
    diff_list=[]
    for idx in range(0, len(ref_kp_list)):    
        dif = (ref_kp_list[idx]['value'] - kp_current['value']).abs().mean()
        diff_list.append(dif)
    #print(diff_list)
    return diff_list


def getFileName(path):
    codedframe_index=[]
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == '.bin':
            codedframe_index.append(i)
    return codedframe_index
            
            
if __name__ == "__main__":
    parser = ArgumentParser()
            
    modeldir = 'temporal_adaptive_2ref'  
    config_path='./checkpoint/'+modeldir+'/vox-256.yaml'
    checkpoint_path='./checkpoint/'+modeldir+'/0099-checkpoint.pth.tar'
    
    frames=250
    
    Qstep= 25 #16
    max_ref_num=3  ###3/4/5
    threshold = 0.25 ###0.2/0.25/0.3/0.35/0.4/0.45
    
    scalenumber=4 #4/6/8
    scalefactor=1/scalenumber
    width=64*scalenumber
    height=64*scalenumber
    
    generator, kp_detector = load_checkpoints(scalenumber,config_path, checkpoint_path, cpu=False)    
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)      
    num_channel = config['common_params']['num_kp']  ###channel
    N_size=int(width*scalefactor/16)   
    
        

    
#     seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020',
#              '021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040']    
#     qplist=['32','37','42','47','52'] 


    seqlist=['002']
    qplist=['32']
    
    
    model_dirname='./experiment-dynamic/'+modeldir+"/"+"Qstep_"+str(Qstep)+"_Numref_"+str(max_ref_num)+"_Threshold_"+str(threshold)+"/"

   
    totalResult=np.zeros((len(seqlist)+1,len(qplist)))
    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):             
            
            original_seq='./Testrgb/'+str(width)+'/'+str(seq)+'_'+str(width)+'x'+str(width)+'.rgb'  #'_1_8bit.rgb'       

            driving_kp =model_dirname+'/kp/'+str(width)+'/'+seq+'_QP'+str(QP)+'/'   
            dir_dec=model_dirname+'/dec/'+str(width)+'/'   
            os.makedirs(dir_dec,exist_ok=True)     # the real decoded video  
            decode_seq=dir_dec+seq+'_QP'+str(QP)+'.rgb'
            
            dir_enc =model_dirname+'/enc/'+str(width)+'/'+seq+'_QP'+str(QP)+'/'
            os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm     
            
            savetxt=model_dirname+'/resultBit/'+str(width)+'/'
            os.makedirs(savetxt,exist_ok=True)         
 
            
            start=time.time() 
            gene_time = 0

            sum_bits = 0

            f_org=open(original_seq,'rb')
            f_dec=open(decode_seq,'w') 
            ref_rgb_list=[]
            seq_kp_integer=[]
            ref_norm_list=[]
            ref_kp_list=[]
            
            codedframe_index=getFileName(dir_enc)
            
            if len(codedframe_index)==1:   
                
                frame_idx_name=os.path.splitext(codedframe_index[0])[0]   
            
                os.system("./vtm/decode.sh "+dir_enc+frame_idx_name) 

                bin_file=dir_enc+frame_idx_name+'.bin'
                bits=os.path.getsize(bin_file)*8
                sum_bits += bits

                f_temp=open(dir_enc+frame_idx_name+'_dec.rgb','rb')
                img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                img_rec.tofile(f_dec) 

                ref_rgb_list.append(img_rec)         
                
                img_rec = resize(img_rec, (3, height, width))                                      
                with torch.no_grad(): 
                    reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                    reference = reference.cuda()    # require GPU

                    kp_reference, multiscale_perceptural_representation_reference = kp_detector(reference,scalefactor) ################                            
                    ref_norm_list.append(reference)
                    ref_kp_list.append(kp_reference)      
                    
                    kp_value = kp_reference['value']
                    kp_value_list = kp_value.tolist()
                    kp_value_list = str(kp_value_list)
                    kp_value_list = "".join(kp_value_list.split())

                    ref_idx_tensor=torch.FloatTensor([0]).cuda()  
                    ref_idx_tensor=ref_idx_tensor.int()
                    ref_idx_list=ref_idx_tensor.tolist()
                    ref_idx_list=str(ref_idx_list)
                    ref_idx_list="".join(ref_idx_list.split())   

                    kp_value_frame=json.loads(kp_value_list)
                    kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))

                    ref_idx_frame=json.loads(ref_idx_list)
                    kp_integer=kp_value_frame+ref_idx_frame #16+1 
                    seq_kp_integer.append(kp_integer)                                         

            else:

                for refnum in range(0,1):
                    frame_idx_name=os.path.splitext(codedframe_index[refnum])[0]   

                    os.system("./vtm/decode.sh "+dir_enc+frame_idx_name) 

                    bin_file=dir_enc+frame_idx_name+'.bin'
                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits

                    f_temp=open(dir_enc+frame_idx_name+'_dec.rgb','rb')
                    img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                    
                    img_rec.tofile(f_dec) 

                    ref_rgb_list.append(img_rec)
                    
                    img_rec = resize(img_rec, (3, height, width))                                      
                    with torch.no_grad(): 
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                        reference = reference.cuda()    # require GPU

                        kp_reference, multiscale_perceptural_representation_reference = kp_detector(reference,scalefactor) ################                            
                        ref_norm_list.append(reference)
                        ref_kp_list.append(kp_reference)
                        
                        kp_value = kp_reference['value']
                        kp_value_list = kp_value.tolist()
                        kp_value_list = str(kp_value_list)
                        kp_value_list = "".join(kp_value_list.split())

                        ref_idx_tensor=torch.FloatTensor([0]).cuda()  
                        ref_idx_tensor=ref_idx_tensor.int()
                        ref_idx_list=ref_idx_tensor.tolist()
                        ref_idx_list=str(ref_idx_list)
                        ref_idx_list="".join(ref_idx_list.split())   

                        kp_value_frame=json.loads(kp_value_list)
                        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))

                        ref_idx_frame=json.loads(ref_idx_list)
                        kp_integer=kp_value_frame+ref_idx_frame #16+1 
                        seq_kp_integer.append(kp_integer)                         
                        
                        
                        
                for refnum in range(1, len(codedframe_index)):
                    frame_idx_name=os.path.splitext(codedframe_index[refnum])[0]
                    
                    #  read the rec frame (yuv420) and convert to rgb444
                    f_temp=open(dir_enc+frame_idx_name+'_rec.yuv','rb')
                    
                    bin_file=dir_enc+frame_idx_name+'.bin'
                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits                    
                    
                    img_rec=np.fromfile(f_temp,np.uint8,3*height*width//2)    # skip the refence frame
                    img_rec_Y = np.fromfile(f_temp,np.uint8,height*width).reshape((height,width))
                    img_rec_U = np.fromfile(f_temp,np.uint8,height*width//4).reshape((height//2, width//2))
                    img_rec_V = np.fromfile(f_temp,np.uint8,height*width//4).reshape((height//2, width//2))
                    img_rec_U=np.repeat(img_rec_U,2,axis=1)
                    img_rec_U=np.repeat(img_rec_U,2,axis=0)                        
                    img_rec_V=np.repeat(img_rec_V,2,axis=1)
                    img_rec_V=np.repeat(img_rec_V,2,axis=0)  
                    img_rec = np.array([img_rec_Y, img_rec_U, img_rec_V])   # 3xHxW
                    img_rec = img_rec.transpose(1, 2, 0)    # HxWx3
                    img_rec = cv2.cvtColor(img_rec, cv2.COLOR_YUV2RGB)
                    img_rec = img_rec.transpose(2, 0, 1)   # 3xHxW
                    #img_rec.tofile(f_dec)

                    ref_rgb_list.append(img_rec) 
                    
                    
                    img_rec = resize(img_rec, (3, height, width))                                      
                    with torch.no_grad(): 
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                        reference = reference.cuda()    # require GPU

                        kp_reference, multiscale_perceptural_representation_reference = kp_detector(reference,scalefactor) ################                            
                        ref_norm_list.append(reference)
                        ref_kp_list.append(kp_reference)
                        
                            
            for frame_idx in range(1, frames):            

                # check whether refresh reference
                frame_index=str(frame_idx).zfill(4)
                bin_save=driving_kp+'/frame'+frame_index+'.bin'            
                kp_dec = final_decoder_expgolomb(bin_save)
                
                
                bits=os.path.getsize(bin_save)*8
                sum_bits += bits                

                ## decoding residual
                kp_difference = data_convert_inverse_expgolomb(kp_dec)
                ## inverse quanzation
                kp_difference_dec=[i/Qstep for i in kp_difference[:16]]+[i for i in kp_difference[16:17]]
                kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  

                
                kp_previous=seq_kp_integer[frame_idx-1]
                kp_previous= eval('[%s]'%repr(kp_previous).replace('[', '').replace(']', '').replace("'", ""))  

                
                kp_integer, kp_integer_matrix,ref_idx=listformat_adptivenew(kp_previous, kp_difference_dec, num_channel,N_size)  #####           
                seq_kp_integer.append(kp_integer)

                kp_integer_matrix=json.loads(str(kp_integer_matrix))
                kp_current_value=torch.Tensor(kp_integer_matrix).to('cuda:0')          
                dict={}
                dict['value']=kp_current_value  
                kp_current=dict 
            
                reference = ref_norm_list[int(ref_idx)]
                kp_reference = ref_kp_list[int(ref_idx)]
                ref_img=ref_rgb_list[int(ref_idx)]

                gene_start = time.time()

                prediction = make_prediction(reference, kp_reference, kp_current, generator,scalefactor) #######################


                gene_end = time.time()
                gene_time += gene_end - gene_start
                pre=(prediction*255).astype(np.uint8)  
                
                
                frame_flag_name=os.path.splitext(codedframe_index[int(ref_idx)])[0].strip('frame')
                if frame_flag_name==frame_index:
                    ref_img.tofile(f_dec)
                else:
                    pre.tofile(f_dec)                      


            f_dec.close()     
            end=time.time()
            
            totalResult[seqIdx][qpIdx]=sum_bits           
            print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))
    
    # summary the bitrate
    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            totalResult[-1][qp]+=totalResult[seq][qp]
        totalResult[-1][qp] /= len(seqlist)
    
    print(totalResult)
    np.set_printoptions(precision=5)
    totalResult = totalResult/1000
    seqlength = frames/25
    totalResult = totalResult/seqlength
    np.savetxt(savetxt+str(width)+'.txt', totalResult, fmt = '%.5f')              
