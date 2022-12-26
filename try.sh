# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir /mnt/petrelfs/huyutao/record/lavit_hu1/try
cp ./run1.sh /mnt/petrelfs/huyutao/record/lavit_hu1/try
cp ./*.py /mnt/petrelfs/huyutao/record/lavit_hu1/try
cp -r ./data /mnt/petrelfs/huyutao/record/lavit_hu1/try
cp -r ./lib /mnt/petrelfs/huyutao/record/lavit_hu1/try
srun --partition=Gvadapt  --mpi=pmi2 --gres=gpu:4 -N 1  --job-name=try --kill-on-bad-exit=1 --quotatype=auto --async -o /mnt/petrelfs/huyutao/record/lavit_hu1/try/log.out --pty --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 4 --master_port 18795 train_conloss_try.py --model lavt --dataset refcoco+ \
--model_id try --output-dir /mnt/petrelfs/huyutao/record/lavit_hu1/try --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ~/pretrained_model/swin_base_patch4_window12_384_22k.pth \
--epochs 60 --rootpath /mnt/petrelfs/huyutao/record/lavit_hu1/ --img_size 480 --tar_size 480 2>&1 | tee /mnt/petrelfs/huyutao/record/lavit_hu1/try/output
# -w SH-IDC1-10-198-6-53
#--async -o /mnt/petrelfs/huyutao/record/lavit_hu1/try/log.out
#~/pretrained_model/swinv2_base_patch4_window12_192_22k.pth
#~/pretrained_model/lavt/ref_refcocogoogle_window12.pth
#~/pretrained_model/swin_base_patch4_window12_384_22k.pth