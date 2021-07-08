import argparse
import os
import shutil
import subprocess
import time

import utils


parser = argparse.ArgumentParser(description="PyTorch Efficient Self-supervised Training")

parser.add_argument('--cfg',
                    help='experiment configure file name',
                    type=str)

# Model parameters
parser.add_argument('--arch', default='deit_small', type=str,
    choices=['swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin', 'vil', 'vil_1281', 'vil_2262', 'vil_14121', 'deit_tiny', 'deit_small', 'vit_base'],
    help="""Name of architecture to train. For quick experiments with ViTs,
    we recommend using deit_tiny or deit_small.""")
parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
    help="""Whether or not to weight normalize the last layer of the DINO head.
    Not normalizing leads to better performance but can make the training unstable.
    In our experiments, we typically set this paramater to False with deit_small and True with vit_base.""")
parser.add_argument('--use_dense_prediction', default=False, type=utils.bool_flag,
    help="Whether to use dense prediction in projection head (Default: False)")
parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
    of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
    starting with the default value of 0.04 and increase this slightly if needed.""")
parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
    help='Number of warmup epochs for the teacher temperature (Default: 30).')
parser.add_argument('--batch_size_per_gpu', default=64, type=int,
    help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
parser.add_argument('--aug-opt', type=str, default='dino_aug', metavar='NAME',
                    help='Use different data augmentation policy. [deit_aug, dino_aug, mocov2_aug, basic_aug] \
                            "(default: dino_aug)')    
parser.add_argument('--zip_mode', type=utils.bool_flag, default=False, help="""Whether or not
    to use zip file.""")
parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
    help='Please specify path to the ImageNet training data.')
parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
parser.add_argument('--pretrained_weights_ckpt', default='.', type=str, help="Path to pretrained weights to evaluate.")

parser.add_argument("--warmup_epochs", default=10, type=int,
    help="Number of epochs for the linear learning-rate warm up.")
# Dataset
parser.add_argument('--dataset', default="imagenet1k", type=str, help='Pre-training dataset.')
parser.add_argument('--tsv_mode', type=utils.bool_flag, default=False, help="""Whether or not to use tsv file.""")
parser.add_argument('--sampler', default="distributed", type=str, help='Sampler for dataloader.')


parser.add_argument('--use_mixup', type=utils.bool_flag, default=False, help="""Whether or not to use mixup/mixcut for self-supervised learning.""")  
parser.add_argument('--num_mixup_views', type=int, default=10, help="""Number of views to apply mixup/mixcut """)
    

# distributed training
parser.add_argument("--num_nodes", default=1, type=int,
                    help="number of nodes for training")               
parser.add_argument("--num_gpus_per_node", default=8, type=int,
                    help="passed as --nproc_per_node parameter")                             
parser.add_argument("--samples_per_gpu", default=1, type=int,
                    help="batch size for training")                                    
parser.add_argument("--node_rank", default=-1, type=int,
                    help="node rank, should be in [0, num_nodes)")

# job meta info
parser.add_argument("--job_name", default="", type=str,
                    help="job name")

args = parser.parse_args()
print(args)

# config_file = args.config_file
# job_name = os.path.basename(args.config_file)[:-5] + "_" + args.job_name

if "OMPI_COMM_WORLD_SIZE" in os.environ:
    if args.num_nodes != int(os.environ["OMPI_COMM_WORLD_SIZE"]):
        args.num_nodes = int(os.environ["OMPI_COMM_WORLD_SIZE"])
else:
    assert args.num_nodes > 0, "number of nodes should be larger than 0!!!"
print("number of nodes: ", args.num_nodes)
imgs_per_batch = args.samples_per_gpu * args.num_nodes * args.num_gpus_per_node
print("batch size: ", imgs_per_batch)

if args.num_nodes > 1:
    args.node_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK')) if 'OMPI_COMM_WORLD_RANK' in os.environ else args.node_rank
    print("node rank: ", args.node_rank)
    # get ip address and port for master process, which the other slave processes will use to communicate
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print("master address-port: {}-{}".format(master_addr, master_port))


    cmd = 'python -m torch.distributed.launch --nproc_per_node={0} --nnodes {1} --node_rank {2} --master_addr {3} --master_port {4} \
            main_esvit.py --data_path {data_path} \
            --output_dir {output_dir} \
            --batch_size_per_gpu {batch_size_per_gpu} \
            --arch {arch} \
            --zip_mode {zip_mode} \
            --epochs {epochs} \
            --teacher_temp {teacher_temp} \
            --warmup_teacher_temp_epochs {warmup_teacher_temp_epochs} \
            --norm_last_layer {norm_last_layer} \
            --cfg {cfg} \
            --use_dense_prediction {use_dense_prediction} \
            --use_mixup {use_mixup} \
            --num_mixup_views {num_mixup_views} \
            --dataset {dataset} \
            --tsv_mode {tsv_mode} \
            --sampler {sampler} \
            --warmup_epochs {warmup_epochs} \
            --pretrained_weights_ckpt {pretrained_weights_ckpt} \
            --aug-opt {aug_opt}'\
            .format(
                args.num_gpus_per_node, args.num_nodes, args.node_rank, master_addr, master_port, 
                data_path=args.data_path, 
                output_dir=args.output_dir, 
                batch_size_per_gpu=args.batch_size_per_gpu, 
                arch=args.arch, 
                zip_mode=args.zip_mode,
                epochs=args.epochs,
                teacher_temp=args.teacher_temp,
                warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
                norm_last_layer=args.norm_last_layer,
                cfg=args.cfg,
                use_dense_prediction=args.use_dense_prediction,
                use_mixup=args.use_mixup,
                num_mixup_views=args.num_mixup_views,        
                dataset=args.dataset,  
                tsv_mode=args.tsv_mode,    
                sampler=args.sampler,
                warmup_epochs=args.warmup_epochs,
                pretrained_weights_ckpt=args.pretrained_weights_ckpt,
                aug_opt=args.aug_opt
            )

else:
    cmd = 'python -m torch.distributed.launch --nproc_per_node={0} --nnodes {1} --node_rank {2} --master_addr {3} --master_port {4} \
            main_esvit.py --data_path {data_path} \
            --output_dir {output_dir} \
            --batch_size_per_gpu {batch_size_per_gpu} \
            --arch {arch} \
            --zip_mode {zip_mode} \
            --epochs {epochs} \
            --teacher_temp {teacher_temp} \
            --warmup_teacher_temp_epochs {warmup_teacher_temp_epochs} \
            --norm_last_layer {norm_last_layer} \
            --cfg {cfg} \
            --use_dense_prediction {use_dense_prediction} \
            --use_mixup {use_mixup} \
            --num_mixup_views {num_mixup_views} \
            --dataset {dataset} \
            --tsv_mode {tsv_mode} \
            --sampler {sampler} \
            --warmup_epochs {warmup_epochs} \
            --pretrained_weights_ckpt {pretrained_weights_ckpt} \
            --aug-opt {aug_opt}'\
            .format(
                args.num_gpus_per_node, args.num_nodes, args.node_rank, master_addr, master_port, 
                data_path=args.data_path, 
                output_dir=args.output_dir, 
                batch_size_per_gpu=args.batch_size_per_gpu, 
                arch=args.arch, 
                zip_mode=args.zip_mode,
                epochs=args.epochs,
                teacher_temp=args.teacher_temp,
                warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
                norm_last_layer=args.norm_last_layer,
                cfg=args.cfg,
                use_dense_prediction=args.use_dense_prediction,
                use_mixup=args.use_mixup,
                num_mixup_views=args.num_mixup_views,        
                dataset=args.dataset,  
                tsv_mode=args.tsv_mode,    
                sampler=args.sampler,
                warmup_epochs=args.warmup_epochs,
                pretrained_weights_ckpt=args.pretrained_weights_ckpt,
                aug_opt=args.aug_opt
            )

subprocess.run(cmd, shell=True, check=True)
