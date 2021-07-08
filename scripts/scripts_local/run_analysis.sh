

DATA_PATH=project/data/imagewoof 
IMG_PATH=$DATA_PATH/train/n02087394/ILSVRC2012_val_00000077.JPEG
OUT_PATH=../output/esvit_exp/ssl/swin_tiny_imagewoof
SEED=8


# EsViT with L_V + L_R
CKPT_PATH=exp_output/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint.pth

python run_analysis.py --arch swin_tiny --image_path $IMG_PATH --output_dir $OUT_PATH/ssl_dense --pretrained_weights $CKPT_PATH --learning ssl --seed $SEED --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --vis_attention True --vis_correspondence False --measure_correspondence False MODEL.NUM_CLASSES 0 

python run_analysis.py --arch swin_tiny --data_path $DATA_PATH --output_dir $OUT_PATH/ssl_dense --pretrained_weights $CKPT_PATH --learning ssl --seed $SEED --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --measure_correspondence True MODEL.NUM_CLASSES 0 


# EsViT with L_V
CKPT_PATH=/exp_output/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_multicrop_epoch300_dino_aug_window7/checkpoint.pth

python run_analysis.py --arch swin_tiny --image_path $IMG_PATH --output_dir $OUT_PATH/ssl_global --pretrained_weights $CKPT_PATH --learning ssl --seed $SEED --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --vis_attention True --vis_correspondence False --measure_correspondence False  MODEL.NUM_CLASSES 0 

python run_analysis.py --arch swin_tiny --data_path $DATA_PATH --output_dir $OUT_PATH/ssl_global --pretrained_weights $CKPT_PATH --learning ssl --seed $SEED --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --measure_correspondence True MODEL.NUM_CLASSES 0 


# DINO with DeiT-Small
CKPT_PATH=project/checkpoints/dino/dino_deitsmall16_pretrain.pth
python run_analysis.py --arch deit_small --data_path $DATA_PATH --output_dir $OUT_PATH/ssl --pretrained_weights $CKPT_PATH --learning ssl --patch_size 16 --seed $SEED --measure_correspondence True MODEL.NUM_CLASSES 0 





# cutest_dog_breeds.jpeg
# cute_dog_1.jpeg
# cute_cat.jpeg
