# Efficient Self-Supervised Vision Transformers (EsViT)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-self-supervised-vision-transformers/self-supervised-image-classification-on)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=efficient-self-supervised-vision-transformers)

[[Paper]](https://arxiv.org/abs/2106.09785) [[Slides]](http://chunyuan.li/assets/pdf/esvit_talk_chunyl.pdf)

PyTorch implementation for [EsViT](https://arxiv.org/abs/2106.09785), built with two techniques: 

- A multi-stage Transformer architecture. Three multi-stage Transformer variants are implemented under the folder [`models`](./models).
- A non-contrastive region-level matching pre-train task. The region-level matching task is implemented in function `DDINOLoss(nn.Module)`  (Line 648) in [`main_esvit.py`](./main_esvit.py). Please use `--use_dense_prediction True`, otherwise only the view-level task is used.




<div align="center">
  <img width="90%" alt="Efficiency vs accuracy comparison under the linear classification protocol on ImageNet with EsViT" src="./plot/esvit_sota.png">
</div>
Figure: Efficiency vs accuracy comparison under the linear classification protocol on ImageNet. Left: Throughput of all SoTA SSL vision systems, circle sizes indicates model parameter counts; Right: performance over varied parameter counts for models with moderate (throughout/#parameters) ratio. Please refer Section 4.1 for details.



## Pretrained models
You can download the full checkpoint (trained with both view-level and region-level tasks, batch size=512 and ImageNet-1K.), which contains backbone and projection head weights for both student and teacher networks. 

- EsViT (Swin) with network configurations of increased model capacities, **pre-trained with both view-level and region-level tasks**. ResNet-50 trained with both tasks is shown as a reference.

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>tasks</th>
    <th>linear</th>
    <th>k-nn</th>
    <th colspan="1">download</th>
    <th colspan="3">logs</th>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>V+R</td>
    <td>75.7%</td>
    <td>71.3%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/resume_from_ckpt0200/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/resume_from_ckpt0200/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/resume_from_ckpt0200/lincls/epoch_last/lr0.01/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/resume_from_ckpt0200/features/epoch0300/log.txt">knn</a></td>    
  </tr>  
  <tr>
    <td>EsViT (Swin-T, W=7)</td>
    <td>28M</td>
    <td>V+R</td>
    <td>78.0%</td>
    <td>75.7%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint_best.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/lincls/epoch0300/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/features/epoch0280/log.txt">knn</a></td>    
  </tr>
  <tr>
    <td>EsViT (Swin-S, W=7)</td>
    <td>49M</td>
    <td>V+R</td>
    <td>79.5%</td>
    <td>77.7%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint_best.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/lincls/epoch0300/lr_0.003_n_last_blocks4/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/features/epoch0280/log.txt">knn</a></td>   
  </tr>
  <tr>
    <td>EsViT (Swin-B, W=7)</td>
    <td>87M</td>
    <td>V+R</td>
    <td>80.4%</td>
    <td>78.9%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/continued_from0200_dense/checkpoint_best.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/continued_from0200_dense/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/continued_from0200_dense/lincls/epoch0260/lr_0.001_n_last_blocks4/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/continued_from0200_dense/features/epoch0260/log.txt">knn</a></td>       
    
    
    
  <tr>
    <td>EsViT (Swin-T, W=14)</td>
    <td>28M</td>
    <td>V+R</td>
    <td>78.7%</td>
    <td>77.0%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug_window14/continued_from0200_dense/checkpoint_best.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug_window14/continued_from0200_dense/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug_window14/continued_from0200_dense/lincls/epoch_last/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug_window14/continued_from0200_dense/features/epoch0300/log.txt">knn</a></td>       
  </tr>
  

  
  <tr>
    <td>EsViT (Swin-S, W=14)</td>
    <td>49M</td>
    <td>V+R</td>
    <td>80.8%</td>
    <td>79.1%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs16_multicrop_epoch300_dino_aug_window14/continued_from0180_dense/checkpoint_best.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs16_multicrop_epoch300_dino_aug_window14/continued_from0180_dense/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs16_multicrop_epoch300_dino_aug_window14/continued_from0180_dense/lincls/epoch0250/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs16_multicrop_epoch300_dino_aug_window14/continued_from0180_dense/features/epoch0250/log.txt">knn</a></td>      
  </tr>
  <tr>
    <td>EsViT (Swin-B, W=14)</td>
    <td>87M</td>
    <td>V+R</td>
    <td>81.3%</td>
    <td>79.3%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_nodes4_gpu16_bs8_multicrop_epoch300_dino_aug_window14_lv/continued_from_epoch0200_dense_norm_true/checkpoint_best.pth">full ckpt</a></td>  
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_nodes4_gpu16_bs8_multicrop_epoch300_dino_aug_window14_lv/continued_from_epoch0200_dense_norm_true/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_nodes4_gpu16_bs8_multicrop_epoch300_dino_aug_window14_lv/continued_from_epoch0200_dense_norm_true/lincls/epoch0240/lr_0.001_n_last_blocks4/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_nodes4_gpu16_bs8_multicrop_epoch300_dino_aug_window14_lv/continued_from_epoch0200_dense_norm_true/features/epoch0240/log.txt">knn</a></td>       
  </tr>  
</table>


- EsViT with view-level task only


<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>tasks</th>
    <th>linear</th>
    <th>k-nn</th>
    <th colspan="1">download</th>
    <th colspan="3">logs</th>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>V</td>
    <td>75.0%</td>
    <td>69.1%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/lincls/epoch_last/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/resnet/resnet50/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/features/epoch0300/log.txt">knn</a></td>    
  </tr>  
  <tr>
    <td>EsViT (Swin-T, W=7)</td>
    <td>28M</td>
    <td>V</td>
    <td>77.0%</td>
    <td>74.2%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug_window7/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug_window7/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug_window7/lincls/epoch0300/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug_window7/features/epoch0300/log.txt">knn</a></td>    
  </tr>  
  <tr>
    <td>EsViT (Swin-S, W=7)</td>
    <td>49M</td>
    <td>V</td>
    <td>79.2%</td>
    <td>76.9%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_multicrop_epoch300/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_multicrop_epoch300/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_multicrop_epoch300/lincls/epoch0300/lr_0.003_n_last_blocks4/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_small/bl_lr0.0005_gpu16_bs32_multicrop_epoch300/features/epoch0280/log.txt">knn</a></td>   
  </tr>
  <tr>
    <td>EsViT (Swin-B, W=7)</td>
    <td>87M</td>
    <td>V</td>
    <td>79.6%</td>
    <td>77.7%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/lincls/epoch0260/lr_0.001_n_last_blocks4/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_base/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug/continued_from0200_dense/features/epoch0260/log.txt">knn</a></td>       
    
  </tr>  
</table>



- EsViT (Swin-T, W=7) with different pre-train datasets (view-level task only)

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>batch size</th>
    <th>pre-train dataset</th>
    <th>linear</th>
    <th>k-nn</th>
    <th colspan="1">download</th>
    <th colspan="3">logs</th>
  </tr>
  <tr>
    <td>EsViT</td>
    <td>28M</td>
    <td>1024</td>
    <td>ImageNet-1K</td>
    <td>77.1%</td>
    <td>73.7%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300/lincls/epoch0300/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300/features/epoch0280/log.txt">knn</a></td>    
  </tr>
  <tr>
    <td>EsViT</td>
    <td>28M</td>
    <td>1024</td>
    <td>WebVision-v1</td>
    <td>75.4%</td>  
    <td>69.4%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch150_dino_aug_window7_webvision1_debug/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch150_dino_aug_window7_webvision1_debug/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch150_dino_aug_window7_webvision1_debug/lincls/epoch_last/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch150_dino_aug_window7_webvision1_debug/features/epoch0150/log.txt">knn</a></td>   
  </tr>
  
  <tr>
    <td>EsViT</td>
    <td>28M</td>
    <td>1024</td>
    <td>OpenImages-v4</td> 
    <td>69.6%</td>   
    <td>60.3%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch150_dino_aug_window7_openimages_v4_debug/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch150_dino_aug_window7_openimages_v4_debug/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch150_dino_aug_window7_openimages_v4_debug/lincls/epoch_last/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch150_dino_aug_window7_openimages_v4_debug/features/epoch050/log.txt">knn</a></td>   
  </tr>
  
  <tr>
    <td>EsViT</td>
    <td>28M</td>
    <td>1024</td>
    <td>ImageNet-22K</td>
    <td>73.5%</td>    
    <td>66.1%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug_window7_imagenet22k_debug/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug_window7_imagenet22k_debug/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug_window7_imagenet22k_debug/lincls/epoch_last/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/swin/swin_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug_window7_imagenet22k_debug/features/epoch0030/log.txt">knn</a></td>   
  </tr>  
</table>


- EsViT with more multi-stage vision Transformer architectures, pre-trained with **V**iew-level and **R**egion-level tasks.

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>pre-train task</th>
    <th>linear</th>
    <th>k-nn</th>
    <th colspan="1">download</th>
    <th colspan="3">logs</th>
  </tr>

  <tr>
    <td>EsViT (ViL, W=7)</td>
    <td>28M</td>
    <td>V</td>
    <td>77.3%</td>
    <td>73.9%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/vil/vil_2262/bl_lr0.0005_gpu16_bs32_multicrop_epoch300/vil_mode0/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/vil/vil_2262/bl_lr0.0005_gpu16_bs32_multicrop_epoch300/vil_mode0/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/vil/vil_2262/bl_lr0.0005_gpu16_bs64_multicrop_epoch300/lincls/epoch0300/4_last_blocks/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/vil/vil_2262/bl_lr0.0005_gpu16_bs32_multicrop_epoch300/vil_mode0/features/epoch300/log.txt">knn</a></td>       
    
    
    
  <tr>
    <td>EsViT (ViL, W=7)</td>
    <td>28M</td>
    <td>V+R</td>
    <td>77.5%</td>
    <td>74.5%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/vil/vil_2262/bl_lr0.0005_gpu16_bs64_multicrop_epoch300/continued_from0200_dense/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/vil/vil_2262/bl_lr0.0005_gpu16_bs64_multicrop_epoch300/continued_from0200_dense/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/vil/vil_2262/bl_lr0.0005_gpu16_bs64_multicrop_epoch300/continued_from0200_dense/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/vil/vil_2262/bl_lr0.0005_gpu16_bs64_multicrop_epoch300/continued_from0200_dense/features/epoch300/log.txt">knn</a></td>       
  </tr>
  

  
  <tr>
    <td>EsViT (CvT, W=7)</td>
    <td>29M</td>
    <td>V</td>
    <td>77.6%</td>
    <td>74.8%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/cvt/cvt_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/checkpoint.pth">full ckpt</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/cvt/cvt_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/cvt/cvt_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/lincls/epoch0300/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/cvt/cvt_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/features/epoch300/log.txt">knn</a></td>      
  </tr>
  <tr>
    <td>EsViT (CvT, W=7)</td>
    <td>29M</td>
    <td>V+R</td>
    <td>78.5%</td>
    <td>76.7%</td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/cvt/cvt_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/continued_from0200_dense/checkpoint.pth">full ckpt</a></td>  
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/cvt/cvt_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/continued_from0200_dense/log.txt">train</a></td>
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/cvt/cvt_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/continued_from0200_dense/lincls/epoch0300/log.txt">linear</a></td> 
    <td><a href="https://chunyleu.blob.core.windows.net/output/ckpts/esvit/cvt/cvt_tiny/bl_lr0.0005_gpu16_bs64_multicrop_epoch300_dino_aug/continued_from0200_dense/features/epoch300/log.txt">knn</a></td>       
  </tr>  
</table>


## Pre-training

### One-node training
To train on 1 node with 16 GPUs for Swin-T model size:
```
PROJ_PATH=your_esvit_project_path
DATA_PATH=$PROJ_PATH/project/data/imagenet

OUT_PATH=$PROJ_PATH/output/esvit_exp/ssl/swin_tiny_imagenet/
python -m torch.distributed.launch --nproc_per_node=16 main_esvit.py --arch swin_tiny --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 32 --epochs 300 --teacher_temp 0.07 --warmup_epochs 10 --warmup_teacher_temp_epochs 30 --norm_last_layer false --use_dense_prediction True --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml 
```

The main training script is [`main_esvit.py`](./main_esvit.py) and conducts the training loop, taking the following options (among others) as arguments:

- `--use_dense_prediction`: whether or not to use the region matching task in pre-training
- `--arch`: switch between different sparse self-attention in the multi-stage Transformer architecture. Example architecture choices for EsViT training include [`swin_tiny`, `swin_small`, `swin_base`, `swin_large`,`cvt_tiny`, `vil_2262`]. The configuration files should be adjusted accrodingly, we provide example below. One may specify the network configuration by editing the `YAML` file under `experiments/imagenet/*/*.yaml`. The default window size=7; To consider a multi-stage architecture with window size=14, please choose yaml files with `window14` in filenames.


To train on 1 node with 16 GPUs for Convolutional vision Transformer (CvT) models:
```
python -m torch.distributed.launch --nproc_per_node=16 main_evsit.py --arch cvt_tiny --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 32 --epochs 300 --teacher_temp 0.07 --warmup_epochs 10 --warmup_teacher_temp_epochs 30 --norm_last_layer false --use_dense_prediction True --aug-opt dino_aug --cfg experiments/imagenet/cvt_v4/s1.yaml
```

To train on 1 node with 16 GPUs for Vision Longformer (ViL) models:
```
python -m torch.distributed.launch --nproc_per_node=16 main_evsit.py --arch vil_2262 --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 32 --epochs 300 --teacher_temp 0.07 --warmup_epochs 10 --warmup_teacher_temp_epochs 30 --norm_last_layer false --use_dense_prediction True --aug-opt dino_aug --cfg experiments/imagenet/vil/vil_small/base.yaml MODEL.SPEC.MSVIT.ARCH 'l1,h3,d96,n2,s1,g1,p4,f7,a0_l2,h6,d192,n2,s1,g1,p2,f7,a0_l3,h12,d384,n6,s0,g1,p2,f7,a0_l4,h24,d768,n2,s0,g0,p2,f7,a0' MODEL.SPEC.MSVIT.MODE 1 MODEL.SPEC.MSVIT.VIL_MODE_SWITCH 0.75
```


### Multi-node training
To train on 2 nodes with 16 GPUs each (total 32 GPUs) for Swin-Small model size:
```
OUT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_small/bl_lr0.0005_gpu16_bs16_multicrop_epoch300_dino_aug_window14
python main_evsit_mnodes.py --num_nodes 2 --num_gpus_per_node 16 --data_path $DATA_PATH/train --output_dir $OUT_PATH/continued_from0200_dense --batch_size_per_gpu 16 --arch swin_small --zip_mode True --epochs 300 --teacher_temp 0.07 --warmup_epochs 10 --warmup_teacher_temp_epochs 30 --norm_last_layer false --cfg experiments/imagenet/swin/swin_small_patch4_window14_224.yaml --use_dense_prediction True --pretrained_weights_ckpt $OUT_PATH/checkpoint0200.pth
```

## Evaluation: 

### k-NN and Linear classification on ImageNet


To train a supervised linear classifier on frozen weights on a single node with 4 gpus, run `eval_linear.py`. To train a k-NN classifier on frozen weights on a single node with 4 gpus, run `eval_knn.py`. Please specify `--arch`, `--cfg` and `--pretrained_weights` to  choose a pre-trained checkpoint. If you want to evaluate the last checkpoint of EsViT with Swin-T, you can run for example:


```
PROJ_PATH=your_esvit_project_path
DATA_PATH=$PROJ_PATH/project/data/imagenet

OUT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300
CKPT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint.pth

python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --data_path $DATA_PATH --output_dir $OUT_PATH/lincls/epoch0300 --pretrained_weights $CKPT_PATH --checkpoint_key teacher --batch_size_per_gpu 256 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --n_last_blocks 4 --num_labels 1000 MODEL.NUM_CLASSES 0

python -m torch.distributed.launch --nproc_per_node=4 eval_knn.py --data_path $DATA_PATH --dump_features $OUT_PATH/features/epoch0300 --pretrained_weights $CKPT_PATH --checkpoint_key teacher --batch_size_per_gpu 256 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml MODEL.NUM_CLASSES 0
```



## Analysis/Visualization of correspondence and attention maps
You can analyze the learned models by running `python run_analysis.py`. One example to analyze EsViT (Swin-T) is shown.

For an invidiual image (with path `--image_path $IMG_PATH`), we visualize the attention maps and correspondence of the last layer:

```
python run_analysis.py --arch swin_tiny --image_path $IMG_PATH --output_dir $OUT_PATH --pretrained_weights $CKPT_PATH --learning ssl --seed $SEED --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --vis_attention True --vis_correspondence True MODEL.NUM_CLASSES 0 
```

For an image dataset (with path `--data_path $DATA_PATH`), we quantatively measure the correspondence:

```
python run_analysis.py --arch swin_tiny --data_path $DATA_PATH --output_dir $OUT_PATH --pretrained_weights $CKPT_PATH --learning ssl --seed $SEED --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml  --measure_correspondence True MODEL.NUM_CLASSES 0 
```

For more examples, please see `scripts/scripts_local/run_analysis.sh`.

## Citation

If you find this repository useful, please consider giving a star :star:   and citation :beer::

```
@article{li2021esvit,
  title={Efficient Self-supervised Vision Transformers for Representation Learning},
  author={Li, Chunyuan and Yang, Jianwei and Zhang, Pengchuan and Gao, Mei and Xiao, Bin and Dai, Xiyang and Yuan, Lu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2106.09785},
  year={2021}
}
```

#### Related Projects/Codebase

[[Swin Transformers](https://github.com/microsoft/Swin-Transformer)]  [[Vision Longformer](https://github.com/microsoft/vision-longformer)]  [[Convolutional vision Transformers (CvT)](https://github.com/microsoft/CvT)]  [[Focal Transformers](https://github.com/microsoft/Focal-Transformer)]

#### Acknowledgement 
Our implementation is built partly upon packages: [[Dino](https://github.com/facebookresearch/dino)]  [[Timm](https://github.com/rwightman/pytorch-image-models)]




## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
