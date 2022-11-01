I# export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir ./models/coco_1006_dynamic
srun --partition=Gvadapt   --mpi=pmi2 --gres=gpu:1 -N 1  --job-name=SEG1 --kill-on-bad-exit=1 --quotatype=spot --pty --mail-type=ALL \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py --model lavt --dataset refcoco \
--model_id try --batch-size 28 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
--epochs 40 --img_size 480 2>&1 | tee ./models/try/output
# -w SH-IDC1-10-198-6-53
