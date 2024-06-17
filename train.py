from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import *
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback
from frames_dataset import *
from torch.autograd import Variable

def train(config, generator, discriminator, temporaldiscriminator, kp_detector, checkpoint, log_dir, mode, device_ids):  
    
    train_params = config['train_params']
    common_params = config['common_params']
    num_temporal =config['common_params']['num_temporal']
    scalerange = config['common_params']['scale_range']
    loss_weights = train_params['loss_weights']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_temporaldiscriminator = torch.optim.Adam(temporaldiscriminator.parameters(), lr=train_params['lr_temporaldiscriminator'], betas=(0.5, 0.999))    
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(scalenumber, checkpoint, generator, discriminator, temporaldiscriminator,kp_detector,
                                      optimizer_generator, optimizer_discriminator,optimizer_temporaldiscriminator, optimizer_kp_detector)
    else:
        start_epoch = 0
        

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_temporaldiscriminator = MultiStepLR(optimizer_temporaldiscriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)    
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))


    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, temporaldiscriminator,train_params, common_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, temporaldiscriminator,train_params, common_params) ### spatial
    discriminator_full_temporal = DiscriminatorFullModel_Temporal(kp_detector, generator, discriminator, temporaldiscriminator,train_params, common_params) ### temporal
    

    if torch.cuda.is_available():
        generator_full = torch.nn.DataParallel(generator_full, device_ids=device_ids).cuda()#.to(device_ids[0])
        discriminator_full = torch.nn.DataParallel(discriminator_full, device_ids=device_ids).cuda()#.to(device_ids[0])  
        discriminator_full_temporal = torch.nn.DataParallel(discriminator_full_temporal, device_ids=device_ids).cuda()#.to(device_ids[0])  

        
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):

            scaletimes=len(scalerange)
            out_ckp=[]
            for scalenumber in scalerange:

                dataset = FramesDataset(scalenumber, is_train=(mode == 'train'), **config['dataset_params'], **config['common_params'])   ########################################   
                #### load data
                if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
                    dataset = DatasetRepeater(dataset, train_params['num_repeats'])
                #print(dataset[0]['driving'].shape)
                dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

                scalefactor = 1/ scalenumber
                #scalefactor = 0.25            
                
                for x in dataloader:

                    losses_generator, generated = generator_full(x, scalefactor)  #####   
                    losses_discriminator = discriminator_full(x, generated) ###spatial
                    losses_discriminator_temporal = discriminator_full_temporal(x, generated) ### temporal               
                    LOSS_gene=0
                    LOSS_disc=0
                    LOSSES={}
                    for driving_num in range(0,num_temporal) : 

                        ### Generator loss                    
                        loss_gene_values = [val.mean() for val in losses_generator['driving'+'_'+str(driving_num)].values()]
                        loss_gene = sum(loss_gene_values)
                        LOSS_gene+=loss_gene
                        
                        ### Spatial Discriminator loss
                        if loss_weights['generator_gan'] != 0:
                            loss_disc_values = [val.mean() for val in losses_discriminator['driving'+'_'+str(driving_num)].values()]
                            loss_disc = sum(loss_disc_values)
                            LOSS_disc+=loss_disc

                            losses_generator['driving'+'_'+str(driving_num)].update(losses_discriminator['driving'+'_'+str(driving_num)])
                            
                        losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator['driving'+'_'+str(driving_num)].items()}
                        for key in LOSSES.keys()| losses.keys():
                            LOSSES[key] = sum([d.get(key, 0) for d in (LOSSES, losses)])

 
                    ### Temporal Discriminator loss
                    if loss_weights['generator_gan'] != 0:
                        loss_temp_disc_values = [val.mean() for val in losses_discriminator_temporal.values()]
                        LOSS_temp_disc_values = sum(loss_temp_disc_values)

                        LOSSES.update({key: value.mean().detach().data.cpu().numpy() for key, value in losses_discriminator_temporal.items()})

                    ########
                    optimizer_generator.zero_grad()
                    optimizer_kp_detector.zero_grad()
                    LOSS_gene.backward(retain_graph=True)  ####
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()
                    
                    if loss_weights['generator_gan'] != 0:
                        ########
                        optimizer_discriminator.zero_grad()
                        LOSS_disc.backward(retain_graph=True) #####
                        optimizer_discriminator.step()
                        optimizer_discriminator.zero_grad()
                        ########
                        optimizer_temporaldiscriminator.zero_grad()
                        LOSS_temp_disc_values.backward(retain_graph=True) #########
                        optimizer_temporaldiscriminator.step()
                        optimizer_temporaldiscriminator.zero_grad()


                    logger.log_iter(losses=LOSSES)          


                scheduler_generator.step()
                scheduler_discriminator.step()
                scheduler_kp_detector.step()
                
                logger.log_epoch_vis(epoch,scalenumber, num_temporal, inp=x, out=generated)  
                
                ckp_dict={'generator'+'_'+str(scalenumber): generator,
                          'discriminator'+'_'+str(scalenumber): discriminator,
                          'kp_detector'+'_'+str(scalenumber): kp_detector,
                          'temporaldiscriminator'+'_'+str(scalenumber): temporaldiscriminator,                          
                          'optimizer_generator'+'_'+str(scalenumber): optimizer_generator,
                          'optimizer_discriminator'+'_'+str(scalenumber): optimizer_discriminator,
                          'optimizer_kp_detector'+'_'+str(scalenumber): optimizer_kp_detector,
                          'optimizer_temporaldiscriminator'+'_'+str(scalenumber): optimizer_temporaldiscriminator}

                out_ckp.append(ckp_dict)
            
            out_ckp_dict={}
            for i in out_ckp:
                out_ckp_dict.update(i)            
            logger.log_epoch_ckp(epoch,scaletimes, out_ckp_dict)     
            
            
            
            
            
def retrain(config, generator, discriminator, temporaldiscriminator, kp_detector, checkpoint, start_epoch, log_dir, mode, device_ids):
        
    train_params = config['train_params']
    common_params = config['common_params']
    num_temporal =config['common_params']['num_temporal']
    scalerange = config['common_params']['scale_range']
    loss_weights = train_params['loss_weights']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_temporaldiscriminator = torch.optim.Adam(temporaldiscriminator.parameters(), lr=train_params['lr_temporaldiscriminator'], betas=(0.5, 0.999))    
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))




    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):

            scaletimes=len(scalerange)
            out_ckp=[]
            for scalenumber in scalerange:

                Logger.load_cpk(scalenumber, checkpoint, generator, discriminator, temporaldiscriminator,kp_detector,
                                optimizer_generator, optimizer_discriminator, optimizer_temporaldiscriminator, optimizer_kp_detector)

                scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                                  last_epoch=start_epoch - 1)
                scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                                      last_epoch=start_epoch - 1)
                scheduler_temporaldiscriminator = MultiStepLR(optimizer_temporaldiscriminator, train_params['epoch_milestones'], gamma=0.1,
                                                      last_epoch=start_epoch - 1)    
                scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                                    last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

                generator_full = GeneratorFullModel(kp_detector, generator, discriminator, temporaldiscriminator,train_params, common_params)
                discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, temporaldiscriminator,train_params, common_params) ### spatial
                discriminator_full_temporal = DiscriminatorFullModel_Temporal(kp_detector, generator, discriminator, temporaldiscriminator,train_params, common_params) ### temporal
                if torch.cuda.is_available():
                    generator_full = torch.nn.DataParallel(generator_full, device_ids=device_ids).cuda()#.to(device_ids[0])
                    discriminator_full = torch.nn.DataParallel(discriminator_full, device_ids=device_ids).cuda()#.to(device_ids[0])  
                    discriminator_full_temporal = torch.nn.DataParallel(discriminator_full_temporal, device_ids=device_ids).cuda()#.to(device_ids[0])                  

                
                
                dataset = FramesDataset(scalenumber, is_train=(mode == 'train'), **config['dataset_params'], **config['common_params'])   ########################################   
                #### load data
                if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
                    dataset = DatasetRepeater(dataset, train_params['num_repeats'])
                #print(dataset[0]['driving'].shape)
                dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

                scalefactor = 1/ scalenumber
                #scalefactor = 0.25            
                
                for x in dataloader:

                    losses_generator, generated = generator_full(x, scalefactor)  #####   
                    losses_discriminator = discriminator_full(x, generated) ###spatial
                    losses_discriminator_temporal = discriminator_full_temporal(x, generated) ### temporal               
                    LOSS_gene=0
                    LOSS_disc=0
                    LOSSES={}
                    for driving_num in range(0,num_temporal) : 

                        ### Generator loss                    
                        loss_gene_values = [val.mean() for val in losses_generator['driving'+'_'+str(driving_num)].values()]
                        loss_gene = sum(loss_gene_values)
                        LOSS_gene+=loss_gene

                        ### Spatial Discriminator loss
                        if loss_weights['generator_gan'] != 0:
                            loss_disc_values = [val.mean() for val in losses_discriminator['driving'+'_'+str(driving_num)].values()]
                            loss_disc = sum(loss_disc_values)
                            LOSS_disc+=loss_disc

                            losses_generator['driving'+'_'+str(driving_num)].update(losses_discriminator['driving'+'_'+str(driving_num)])
                            
                        losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator['driving'+'_'+str(driving_num)].items()}
                        for key in LOSSES.keys()| losses.keys():
                            LOSSES[key] = sum([d.get(key, 0) for d in (LOSSES, losses)])

 
                    ### Temporal Discriminator loss
                    if loss_weights['generator_gan'] != 0:
                        loss_temp_disc_values = [val.mean() for val in losses_discriminator_temporal.values()]
                        LOSS_temp_disc_values = sum(loss_temp_disc_values)

                        LOSSES.update({key: value.mean().detach().data.cpu().numpy() for key, value in losses_discriminator_temporal.items()})

                    #print(LOSSES)
                    ########
                    optimizer_generator.zero_grad()
                    optimizer_kp_detector.zero_grad()
                    LOSS_gene.backward(retain_graph=True)  ####
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()
                    
                    if loss_weights['generator_gan'] != 0:
                        ########
                        optimizer_discriminator.zero_grad()
                        LOSS_disc.backward(retain_graph=True) #####
                        optimizer_discriminator.step()
                        optimizer_discriminator.zero_grad()
                        ########
                        optimizer_temporaldiscriminator.zero_grad()
                        LOSS_temp_disc_values.backward(retain_graph=True) #########
                        optimizer_temporaldiscriminator.step()
                        optimizer_temporaldiscriminator.zero_grad()

                    logger.log_iter(losses=LOSSES)          


                scheduler_generator.step()
                scheduler_discriminator.step()
                scheduler_kp_detector.step()
                
                logger.log_epoch_vis(epoch,scalenumber, num_temporal, inp=x, out=generated)  
                
                ckp_dict={'generator'+'_'+str(scalenumber): generator,
                          'discriminator'+'_'+str(scalenumber): discriminator,
                          'kp_detector'+'_'+str(scalenumber): kp_detector,
                          'temporaldiscriminator'+'_'+str(scalenumber): temporaldiscriminator,                          
                          'optimizer_generator'+'_'+str(scalenumber): optimizer_generator,
                          'optimizer_discriminator'+'_'+str(scalenumber): optimizer_discriminator,
                          'optimizer_kp_detector'+'_'+str(scalenumber): optimizer_kp_detector,
                          'optimizer_temporaldiscriminator'+'_'+str(scalenumber): optimizer_temporaldiscriminator}

                out_ckp.append(ckp_dict)
            
            out_ckp_dict={}
            for i in out_ckp:
                out_ckp_dict.update(i)            
            logger.log_epoch_ckp(epoch,scaletimes, out_ckp_dict)     
