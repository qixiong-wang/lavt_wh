# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir /mnt/lustre/huyutao.vendor/record/lavit/debug
cp ./run.sh /mnt/lustre/huyutao.vendor/record/lavit/debug
cp ./*.py /mnt/lustre/huyutao.vendor/record/lavit/debug
cp -r ./lib/ /mnt/lustre/huyutao.vendor/record/lavit/debug
srun --partition=sensedeep  --mpi=pmi2 --gres=gpu:1 -N 1  --job-name=SEG_debug --kill-on-bad-exit=1 --pty --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23333 train_flip.py --model lavt --dataset refcoco \
--model_id debug --output-dir /mnt/lustre/huyutao.vendor/record/lavit/debug/ --batch-size 2 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
--epochs 60 --img_size 480 2>&1 | tee /mnt/lustre/huyutao.vendor/record/lavit/debug/output
# -w SH-IDC1-10-198-6-53
t
