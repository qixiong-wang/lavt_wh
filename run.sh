# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir /mnt/lustre/huyutao.vendor/record/lavit/h11
cp ./run.sh /mnt/lustre/huyutao.vendor/record/lavit/h11
cp ./*.py /mnt/lustre/huyutao.vendor/record/lavit/h11
cp -r ./lib/ /mnt/lustre/huyutao.vendor/record/lavit/h11
srun --partition=sensedeep  --mpi=pmi2 --gres=gpu:4 -N 1  --job-name=SEG_h11 --kill-on-bad-exit=1 --pty --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12365 train_flip.py --model lavt --dataset refcoco \
--model_id h11 --output-dir /mnt/lustre/huyutao.vendor/record/lavit/h11/ --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
--epochs 60 --img_size 480 2>&1 | tee /mnt/lustre/huyutao.vendor/record/lavit/h11/output
# -w SH-IDC1-10-198-6-53
t
