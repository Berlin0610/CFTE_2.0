# -*- coding: utf-8 -*-
from torch import nn
import torch
import torch.nn.functional as F
from modules.util import *
import numpy as np
from torch.autograd import grad
from .GDN import GDN
import math
from modules.vggloss import *
from modules.spynet import *
#from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, temporaldiscriminator,train_params, common_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        #self.scale_factor = train_params['scale_factor']
        self.scales = train_params['scales']
        self.temperature =train_params['temperature']
        self.out_channels =common_params['num_kp'] 
        self.num_ref = common_params['num_ref']
        self.num_temporal = common_params['num_temporal']        
        self.disc_scales = self.discriminator.scales
        self.num_channels = common_params['num_channels']
        
        # self.down = AntiAliasInterpolation2d(generator.num_channels, self.scale_factor)    
            
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()
            

    def forward(self, x, scalefactor):
        
        #heatmap_source = self.kp_extractor(x['source'], scalefactor)
        heatmap_source, multiscale_perceptural_representation_source = self.kp_extractor(x['source'], scalefactor)
        
        loss_values_temporal={}
        generated_temporal={}      
        
        for driving_num in range(0,self.num_temporal) :    
            
            #heatmap_driving = self.kp_extractor(x['driving'+'_'+str(driving_num)], scalefactor)            
            heatmap_driving, multiscale_perceptural_representation_driving = self.kp_extractor(x['driving'+'_'+str(driving_num)], scalefactor)         
            

            if self.num_ref > 1:
                heatmap_source_more = self.kp_extractor(x['source_more'], scalefactor)
                generated = self.generator(x['source'], scalefactor, heatmap_source,heatmap_driving,x['source_more'],heatmap_source_more)
                generated.update({'heatmap_source':heatmap_source,'heatmap_driving':heatmap_driving,'heatmap_source_more':heatmap_source_more}) 
            else:  
                generated = self.generator(x['source'], scalefactor, heatmap_source,heatmap_driving)
                generated.update({'heatmap_source':heatmap_source,'heatmap_driving':heatmap_driving}) 

            
            loss_values = {}

            pyramide_real = self.pyramid(x['driving'+'_'+str(driving_num)])
            pyramide_generated = self.pyramid(generated['prediction'])


            if scalefactor != 1:
                self.down = AntiAliasInterpolation2d(self.num_channels, scalefactor).cuda()
                driving_image_downsample = self.down(x['driving'+'_'+str(driving_num)])
            else:
                driving_image_downsample = x['driving'+'_'+str(driving_num)]
            pyramide_real_downsample = self.pyramid(driving_image_downsample)             
            
#             driving_image_downsample = self.down(x['driving'+'_'+str(driving_num)])    ### [3,64,64]   ##########################
#             pyramide_real_downsample = self.pyramid(driving_image_downsample) 
            sparse_deformed_generated=generated['sparse_deformed']  ### [3,64,64]
            sparse_pyramide_generated = self.pyramid(sparse_deformed_generated)      

            #################       
            if scalefactor != 1:
                self.down = AntiAliasInterpolation2d(self.num_channels, scalefactor).cuda()
                source_image_downsample = self.down(x['source'])    ### [3,64,64]   #############source_image_downsample
            else:
                source_image_downsample = x['source']
            optical_flow_generated=generated['deformation'].to(device)   
            
            optical_flow_real= spynet_estimate(source_image_downsample,driving_image_downsample).to(device)
            optical_flow_real=optical_flow_real.permute(0, 2, 3, 1) 
            
            if self.num_ref > 1:      
                if scalefactor != 1:
                    self.down = AntiAliasInterpolation2d(self.num_channels, scalefactor).cuda()
                    source_image_downsample_more = self.down(x['source_more'])    ### [3,64,64]   #############source_image_downsample
                else:
                    source_image_downsample_more = x['source_more']                
                optical_flow_generated_more=generated['deformation_more']     
                optical_flow_real_more= spynet_estimate(source_image_downsample_more,driving_image_downsample).to(device)
                optical_flow_real_more=optical_flow_real_more.permute(0, 2, 3, 1)
                
            ### Perceptual Loss---Initial
            if sum(self.loss_weights['perceptual_initial']) != 0:
                value_total = 0
                for scale in [1, 0.5, 0.25]:
                    x_vgg = self.vgg(sparse_pyramide_generated['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real_downsample['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual_initial']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual_initial'][i] * value

                loss_values['initial'] = value_total   

                if self.num_ref > 1:
                    sparse_deformed_generated_more=generated['sparse_deformed_more']  ### [3,64,64]
                    sparse_pyramide_generated_more = self.pyramid(sparse_deformed_generated_more)           

                    value_total = 0
                    for scale in [1, 0.5, 0.25]:
                        x_vgg = self.vgg(sparse_pyramide_generated_more['prediction_' + str(scale)])
                        y_vgg = self.vgg(pyramide_real_downsample['prediction_' + str(scale)])

                        for i, weight in enumerate(self.loss_weights['perceptual_initial']):
                            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                            value_total += self.loss_weights['perceptual_initial'][i] * value

                    loss_values['initial'] += value_total 

#             #### optical flow loss 
            
            if self.loss_weights['optical_flow'] != 0:
                value = torch.abs(optical_flow_real.to(device).detach()-optical_flow_generated.to(device).detach()).mean()
                value = value.requires_grad_()
                value_total = self.loss_weights['optical_flow'] * value            
                loss_values['optical_flow'] = value_total   
                
                if self.num_ref > 1:
                    value = torch.abs(optical_flow_real.to(device).detach()-optical_flow_generated.to(device).detach()).mean()
                    value = value.requires_grad_()
                    value_total = self.loss_weights['optical_flow'] * value            
                    loss_values['optical_flow'] += value_total                     
                    
            ### Perceptual Loss---Final
            if sum(self.loss_weights['perceptual_final']) != 0:
                value_total = 0
                for scale in self.scales:
                    x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual_final']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual_final'][i] * value

                loss_values['prediction'] = value_total

                if self.num_ref > 1:
                    pyramide_generated_more = self.pyramid(generated['prediction_more'])
                    pyramide_generated_fusion = self.pyramid(generated['prediction_fusion'])

                    value_total = 0
                    for scale in self.scales:
                        x_vgg = self.vgg(pyramide_generated_more['prediction_' + str(scale)])
                        y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                        for i, weight in enumerate(self.loss_weights['perceptual_final']):
                            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                            value_total += self.loss_weights['perceptual_final'][i] * value

                    loss_values['prediction'] += value_total

                    value_total = 0
                    for scale in self.scales:
                        x_vgg = self.vgg(pyramide_generated_fusion['prediction_' + str(scale)])
                        y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                        for i, weight in enumerate(self.loss_weights['perceptual_final']):
                            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                            value_total += self.loss_weights['perceptual_final'][i] * value

                    loss_values['fusion'] = value_total


            ### GAN adversial Loss
            if self.loss_weights['generator_gan'] != 0:

                discriminator_maps_generated = self.discriminator(pyramide_generated)
                discriminator_maps_real = self.discriminator(pyramide_real)     

                value_total = 0
                for scale in self.disc_scales:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    value_total += self.loss_weights['generator_gan'] * value
                loss_values['gen_gan'] = value_total

                if sum(self.loss_weights['feature_matching']) != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            if self.loss_weights['feature_matching'][i] == 0:
                                continue
                            value = torch.abs(a - b).mean()
                            value_total += self.loss_weights['feature_matching'][i] * value
                        loss_values['feature_matching'] = value_total
                        
         
            loss_values_temporal['driving'+'_'+str(driving_num)]=loss_values
            generated_temporal['driving'+'_'+str(driving_num)]=generated
        return loss_values_temporal, generated_temporal
    
    
    
class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, temporaldiscriminator,train_params,common_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        
        self.num_temporal = common_params['num_temporal']   

    def forward(self, x, generated):        
        
        loss_values_temporal={}
        
        for driving_num in range(0,self.num_temporal) :           
        
            pyramide_real = self.pyramid(x['driving'+'_'+str(driving_num)]) ######################        

            pyramide_generated = self.pyramid(generated['driving'+'_'+str(driving_num)]['prediction'].detach())
            


            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)

            loss_values = {}

            if self.loss_weights['generator_gan'] != 0:
                value_total = 0
                for scale in self.scales:
                    key = 'prediction_map_%s' % scale
                    value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
                    value_total += self.loss_weights['discriminator_gan'] * value.mean()
                loss_values['disc_spatial_gan'] = value_total
            else:
                loss_values['disc_spatial_gan'] = {}
                
            loss_values_temporal['driving'+'_'+str(driving_num)]=loss_values
        return loss_values_temporal
    
    
class DiscriminatorFullModel_Temporal(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator,temporaldiscriminator, train_params,common_params):
        super(DiscriminatorFullModel_Temporal, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.temporaldiscriminator = temporaldiscriminator #######################       
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        
        self.num_temporal = common_params['num_temporal']   

    def forward(self, x, generated):        
              
        pyramide_real_dict={}
        pyramide_generated_dict={}
        for driving_num in range(0,self.num_temporal) :           
        
            pyramide_real = self.pyramid(x['driving'+'_'+str(driving_num)]) ######################        

            pyramide_generated = self.pyramid(generated['driving'+'_'+str(driving_num)]['prediction'].detach())
            
            pyramide_real_dict['driving'+'_'+str(driving_num)]=pyramide_real   
            pyramide_generated_dict['driving'+'_'+str(driving_num)]=pyramide_generated   
        
        #print(pyramide_real_dict)
        
        original_temporal_pyramide={}
        generated_temporal_pyramide={}
        for scale in self.scales:
            original_temporal = torch.cat([ pyramide_real_dict['driving'+'_'+str(driving_num)]['prediction_'+str(scale)] for driving_num in range(0,self.num_temporal)  ],1)
            generated_temporal = torch.cat([ pyramide_generated_dict['driving'+'_'+str(driving_num)]['prediction_'+str(scale)] for driving_num in range(0,self.num_temporal)  ],1)        
       
            original_temporal_pyramide['prediction_'+str(scale)]=original_temporal
            generated_temporal_pyramide['prediction_'+str(scale)]=generated_temporal        
        
        
        discriminator_maps_generated = self.temporaldiscriminator(generated_temporal_pyramide)
        discriminator_maps_real = self.temporaldiscriminator(original_temporal_pyramide)
        
        loss_values = {}

        if self.loss_weights['generator_gan'] != 0:
            value_total = 0
            for scale in self.scales:               
                
                key = 'prediction_map_%s' % scale
                value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
                value_total += self.loss_weights['discriminator_temporal_gan'] * value.mean()
            loss_values['disc_temporal_gan'] = value_total
        else:
            loss_values['disc_temporal_gan'] = {}
                
        return loss_values    

    
