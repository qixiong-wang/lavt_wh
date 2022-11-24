# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir /mnt/petrelfs/huyutao/record/lavit_hu1/h13_1107
cp ./run1.sh /mnt/petrelfs/huyutao/record/lavit_hu1/h13_1107
cp ./*.py /mnt/petrelfs/huyutao/record/lavit_hu1/h13_1107
cp -r ./lib /mnt/petrelfs/huyutao/record/lavit_hu1/h13_1107
srun --partition=Gvadapt  --mpi=pmi2 --gres=gpu:4 -N 1  --job-name=h13_1107 --kill-on-bad-exit=1 --pty --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12372 train_conloss.py --model lavt --dataset refcoco \
--model_id h13_1107 --output-dir /mnt/petrelfs/huyutao/record/lavit_hu1/h13_1107 --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
--epochs 60 --rootpath /mnt/petrelfs/huyutao/record/lavit_hu1/ --img_size 480 2>&1 | tee /mnt/petrelfs/huyutao/record/lavit_hu1/h13_1107/output
# -w SH-IDC1-10-198-6-53
#--quotatype=spot --async -o /mnt/petrelfs/huyutao/record/lavit_hu1/h13_1107/log.out