# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir /mnt/lustre/huyutao.vendor/record/lavit/test
cp ./run.sh /mnt/lustre/huyutao.vendor/record/lavit/test
cp ./*.py /mnt/lustre/huyutao.vendor/record/lavit/test
cp -r ./lib/ /mnt/lustre/huyutao.vendor/record/lavit/test
srun --partition=sensedeep  --mpi=pmi2 --gres=gpu:1 -N 1  --job-name=test --kill-on-bad-exit=1 --pty --mail-type=ALL \
python test.py --model lavt --swin_type base --dataset refcoco --split val --resume /mnt/lustre/huyutao.vendor/record/lavit/h08/model_best_h08.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
t

