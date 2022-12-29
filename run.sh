CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py --model lavt --dataset refcoco \
--model_id refcoco --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights /data/huyutao/pretrain_models/swin_base_patch4_window12_384_22k.pth \
--epochs 40 --img_size 480 2>&1 | tee ./models/refcoco/output

