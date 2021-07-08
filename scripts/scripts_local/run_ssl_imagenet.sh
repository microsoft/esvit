

PROJ_PATH=Anonymous
DATA_PATH=$PROJ_PATH/project/data/imagewoof 

# CKPT_PATH=$PROJ_PATH/project/checkpoints/dino/dino_deitsmall16_pretrain.pth


# Swin-T
OUT_PATH=$PROJ_PATH/output/esvit_exp/ssl/swin_tiny_imagewoof/
python -m torch.distributed.launch --nproc_per_node=2 main_esvit.py --arch swin_tiny --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 8 --use_dense_prediction True --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml 

# ViL-T
OUT_PATH=$PROJ_PATH/output/esvit_exp/ssl/vil_2262_imagewoof/
python -m torch.distributed.launch --nproc_per_node=2 main_esvit.py --arch vil --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 8 --cfg experiments/imagenet/vil/vil_small/base.yaml --use_dense_prediction True MODEL.SPEC.MSVIT.ARCH 'l1,h3,d96,n2,s1,g1,p4,f7,a0_l2,h6,d192,n2,s1,g1,p2,f7,a0_l3,h12,d384,n6,s0,g1,p2,f7,a0_l4,h24,d768,n2,s0,g0,p2,f7,a0' MODEL.SPEC.MSVIT.MODE 1 MODEL.SPEC.MSVIT.VIL_MODE_SWITCH 0.75

# CvT-T
OUT_PATH=$PROJ_PATH/output/esvit_exp/ssl/cvt_tiny_imagewoof/
python -m torch.distributed.launch --nproc_per_node=2 main_esvit.py --arch cvt_tiny --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 8 --use_dense_prediction False --aug-opt dino_aug --cfg experiments/imagenet/cvt_v4/s1.yaml --use_dense_prediction True 

# ResNet50
OUT_PATH=$PROJ_PATH/output/esvit_exp/ssl/resnet50_imagewoof/
python -m torch.distributed.launch --nproc_per_node=2 main_esvit.py --arch resnet50 --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 8 --use_dense_prediction True



