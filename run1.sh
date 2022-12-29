# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir /mnt/petrelfs/huyutao/record/lavit_hu1/h68_1228
cp ./run1.sh /mnt/petrelfs/huyutao/record/lavit_hu1/h68_1228
cp ./*.py /mnt/petrelfs/huyutao/record/lavit_hu1/h68_1228
cp -r ./data /mnt/petrelfs/huyutao/record/lavit_hu1/h68_1228
cp -r ./lib /mnt/petrelfs/huyutao/record/lavit_hu1/h68_1228
srun --partition=Gvadapt  --mpi=pmi2 --gres=gpu:4 -N 1  --job-name=h68_1228 --kill-on-bad-exit=1 --quotatype=auto --async -o /mnt/petrelfs/huyutao/record/lavit_hu1/h68_1228/log.out --pty --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12929 train_noun_conloss.py --model lavt --dataset refcocog --splitBy umd \
--model_id h68_1228 --output-dir /mnt/petrelfs/huyutao/record/lavit_hu1/h68_1228 --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ~/pretrained_model/swin_base_patch4_window12_384_22k.pth \
--epochs 60 --rootpath /mnt/petrelfs/huyutao/record/lavit_hu1/ --img_size 480 2>&1 | tee /mnt/petrelfs/huyutao/record/lavit_hu1/h68_1228/output
# -w SH-IDC1-10-198-6-53
# --splitBy unc
#--async -o /mnt/petrelfs/huyutao/record/lavit_hu1/h68_1228/log.out
