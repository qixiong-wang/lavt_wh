# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir /mnt/petrelfs/huyutao/record/lavit_hu1/h89_0110
cp ./run1.sh /mnt/petrelfs/huyutao/record/lavit_hu1/h89_0110
cp ./*.py /mnt/petrelfs/huyutao/record/lavit_hu1/h89_0110
cp -r ./data /mnt/petrelfs/huyutao/record/lavit_hu1/h89_0110
cp -r ./lib /mnt/petrelfs/huyutao/record/lavit_hu1/h89_0110
srun --partition=Gvadapt  --mpi=pmi2 --gres=gpu:4 -N 1  --job-name=h89_0110 --kill-on-bad-exit=1 --quotatype=auto --async -o /mnt/petrelfs/huyutao/record/lavit_hu1/h89_0110/log.out --pty --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 4 --master_port 18910 train_noun_conloss.py --model lavt --dataset refcoco+ --splitBy unc \
--model_id h89_0110 --output-dir /mnt/petrelfs/huyutao/record/lavit_hu1/h89_0110 --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ~/pretrained_model/coco2017pretrain/h14/model_window12_swinseg_29.pth \
--epochs 60 --rootpath /mnt/petrelfs/huyutao/record/lavit_hu1/ --img_size 480 2>&1 | tee /mnt/petrelfs/huyutao/record/lavit_hu1/h89_0110/output
# -w SH-IDC1-10-198-6-53
# --splitBy unc
#--async -o /mnt/petrelfs/huyutao/record/lavit_hu1/h89_0110/log.out
# ~/pretrained_model/swin_base_patch4_window12_384_22k.pth
# ~/pretrained_model/coco2017pretrain/h14/model_window12_swinseg_9.pth
