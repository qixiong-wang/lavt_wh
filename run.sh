# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir /mnt/petrelfs/huyutao/record/lavit_hu1/cocoh06_1015_dynamic_batch8_decodekc
cp ./run.sh /mnt/petrelfs/huyutao/record/lavit_hu1/cocoh06_1015_dynamic_batch8_decodekc
cp ./*.py /mnt/petrelfs/huyutao/record/lavit_hu1/cocoh06_1015_dynamic_batch8_decodekc
cp -r ./lib /mnt/petrelfs/huyutao/record/lavit_hu1/cocoh06_1015_dynamic_batch8_decodekc
srun --partition=Gvadapt  --mpi=pmi2 --gres=gpu:4 -N 1  --job-name=SEG1 --kill-on-bad-exit=1 --quotatype=spot --pty --async --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12358 train_kc.py --model lavt --dataset refcoco \
--model_id cocoh06_1015_dynamic_batch8_decodekc --output-dir /mnt/petrelfs/huyutao/record/lavit_hu1/cocoh06_1015_dynamic_batch8_decodekc --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
--epochs 60 --rootpath /mnt/petrelfs/huyutao/record/lavit_hu1/ --img_size 480 2>&1 | tee /mnt/petrelfs/huyutao/record/lavit_hu1/cocoh06_1015_dynamic_batch8_decodekc/output
# -w SH-IDC1-10-198-6-53
