# export CUDA_VISIBLE_DEVICES=0,1,2,3
#mkdir ./models/debug
#cp ./debug.sh ./models/debug
#cp ./*.py ./models/debug
#cp -r ./lib ./models/debug
#srun --partition=Gvadapt   --mpi=pmi2 --gres=gpu:1 -N 1  --job-name=SEG1 --kill-on-bad-exit=1 --quotatype=spot --pty --mail-type=ALL \
#python -m torch.distributed.launch --nproc_per_node 1 --master_port 23333 train.py --model lavt --dataset refcoco \
#--model_id debug --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
#--epochs 40 --img_size 480 2>&1 | tee ./models/debug/output
mkdir /mnt/petrelfs/huyutao/record/lavit_hu1/debug
cp ./debug.sh /mnt/petrelfs/huyutao/record/lavit_hu1/debug
cp ./*.py /mnt/petrelfs/huyutao/record/lavit_hu1/debug
cp -r ./lib /mnt/petrelfs/huyutao/record/lavit_hu1/debug
srun --partition=Gvadapt --mpi=pmi2 --gres=gpu:1 -N 1  --job-name=SEG1 --kill-on-bad-exit=1 --pty --quotatype=spot --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23333 train_conloss.py --model lavt --dataset refcoco \
--model_id debug --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
--epochs 40 --rootpath /mnt/petrelfs/huyutao/record/lavit_hu1/ --img_size 480 2>&1 | tee /mnt/petrelfs/huyutao/record/lavit_hu1/debug/output
# -w SH-IDC1-10-198-6-53
