# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir /mnt/petrelfs/huyutao/record/lavit_hu1/h29_1127
cp ./run1.sh /mnt/petrelfs/huyutao/record/lavit_hu1/h29_1127
cp ./*.py /mnt/petrelfs/huyutao/record/lavit_hu1/h29_1127
cp -r ./lib /mnt/petrelfs/huyutao/record/lavit_hu1/h29_1127
srun --partition=Gvadapt  --mpi=pmi2 --gres=gpu:4 -N 1  --job-name=h29_1127 --kill-on-bad-exit=1 --quotatype=auto --async -o /mnt/petrelfs/huyutao/record/lavit_hu1/h29_1127/log.out --pty --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12378 train_flip.py --model lavt --dataset refcoco \
--model_id h29_1127 --output-dir /mnt/petrelfs/huyutao/record/lavit_hu1/h29_1127 --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
--epochs 60 --rootpath /mnt/petrelfs/huyutao/record/lavit_hu1/ --img_size 480 2>&1 | tee /mnt/petrelfs/huyutao/record/lavit_hu1/h29_1127/output
# -w SH-IDC1-10-198-6-53
#--async -o /mnt/petrelfs/huyutao/record/lavit_hu1/h29_1127/log.out
