# Compact Temporal Trajectory Representation for Talking Face Video Compression

## Bolin Chen&dagger;, Zhao Wang&sect;, Binzhe Li&dagger;, Shiqi Wang&dagger; and Yan Ye&sect;

### &dagger; City University of Hong Kong and &sect; Alibaba Group

#### The first two authors (Bolin Chen and Zhao Wang) contributed equally to this work

## Abstract

In this paper, we propose to compactly represent the nonlinear dynamics along the temporal trajectories for talking face video compression. By projecting the frames into a high dimensional space, the temporal trajectories of talking face frames, which are complex, non-linear and difficult to extrapolate, are implicitly modelled in an end-to-end inference framework based upon very compact feature representation. As such, the proposed framework is suitable for ultra-low bandwidth video communication and can guarantee the quality of the reconstructed video in such applications. The proposed compression scheme is also robust against large head-pose motions, due to the delicately designed dynamic reference refresh and temporal stabilization mechanisms. Experimental results demonstrate that compared to the state-of-the-art video coding standard Versatile Video Coding (VVC) as well as the latest generative compression schemes, our proposed scheme is superior in terms of both objective and subjective quality at the same bitrate.

## Quality Comparisons (Similar Bitrate)

### For better quality comparisons, please download the videos (mp4) from the "video" file.

### Example 1

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/222747423-26c459a5-83bc-48cf-999b-a7fb0a3321bd.mp4)](https://user-images.githubusercontent.com/80899378/222747423-26c459a5-83bc-48cf-999b-a7fb0a3321bd.mp4)

### Example 2

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/222747388-7943435b-628a-4d6c-949c-1a595ccdac15.mp4)](https://user-images.githubusercontent.com/80899378/222747388-7943435b-628a-4d6c-949c-1a595ccdac15.mp4)


### Example 3

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/222747398-f9bdea16-1b14-44a4-9ff5-e2b911216f0b.mp4)](https://user-images.githubusercontent.com/80899378/222747398-f9bdea16-1b14-44a4-9ff5-e2b911216f0b.mp4)

### Example 4

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/222747405-550462a3-6348-47d5-a031-f3e484f4ac6d.mp4)](https://user-images.githubusercontent.com/80899378/222747405-550462a3-6348-47d5-a031-f3e484f4ac6d.mp4)

### Example 5

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/226194521-a987e8c7-c03f-4780-b81b-d1696822bdbc.mp4)](https://user-images.githubusercontent.com/80899378/226194521-a987e8c7-c03f-4780-b81b-d1696822bdbc.mp4)

### Example 6

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/226194523-d791d16c-a8a0-474c-8dcc-cb7ea56e0175.mp4)](https://user-images.githubusercontent.com/80899378/226194523-d791d16c-a8a0-474c-8dcc-cb7ea56e0175.mp4)



### Training

To train a model on [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/), please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.

It should be mentioned that the CTTR model can train different 64x-based resolutions (like 256x256, 384x384, 512x512, 640x640 and others), therefore the pre-processed image resolution should include the resolution you would like to inference.

When finishing the downloading and pre-processing the dataset, you can train the model,
```
python run.py
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder. To check the loss values during training see ```log.txt```. You can also check training data reconstructions in the ```train-vis``` subfolder. You can change the training settings in corresponding ```./config/vox-256.yaml``` file.

### Inference

To encode a sequence, please put the provided testing sequence in ```./testing_data/``` file and run
```
python CTTR_Encoder.py
```
After obtaining the bistream, please run
```
python CTTR_Decoder.py
```

In addition, we also provide the dynamic reference strategy for adapting to large-headpose motion. You can run

```
python CTTR_Encoder_dynamic.py  && python CTTR_Decoder_dynamic.py
```

For the testing sequence, it should be in the format of ```RGB:444``` at the different resolutions like 256x256, 384x384, 512x512, 640x640 and others.


The pretrained model can be found under following link: [OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/bolinchen3-c_my_cityu_edu_hk/EVYknkgOCJlPnAHDIMiNdc0Bcn_1gny-XLOR8xCdGYZQJQ?e=hkjSQQ). 


### Evaluate

In ```./evaluate/multiMetric.py``` file, we provide the corresponding quality measures, including DISTS, LPIPS, PSNR and SSIM.



### Additional notes

#### Reference

The training code refers to the CFTE: https://github.com/Berlin0610/CFTE_DCC2022.

The arithmetic-coding refers to https://github.com/nayuki/Reference-arithmetic-coding.


#### Citation:

```
@ARTICLE{10109861,
  author={Chen, Bolin and Wang, Zhao and Li, Binzhe and Wang, Shiqi and Ye, Yan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Compact Temporal Trajectory Representation for Talking Face Video Compression}, 
  year={2023},
  volume={33},
  number={11},
  pages={7009-7023},
  doi={10.1109/TCSVT.2023.3271130}}
```
