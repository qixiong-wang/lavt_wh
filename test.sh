mkdir /mnt/petrelfs/huyutao/record/lavit_hu1/test
cp ./run1.sh /mnt/petrelfs/huyutao/record/lavit_hu1/test
cp ./*.py /mnt/petrelfs/huyutao/record/lavit_hu1/test
cp -r ./lib /mnt/petrelfs/huyutao/record/lavit_hu1/test
srun --partition=Gvadapt  --mpi=pmi2 --gres=gpu:1 -N 1  --job-name=test --kill-on-bad-exit=1 --pty --quotatype=auto --mail-type=ALL \
python test_ms.py --model lavt --swin_type base --dataset refcoco --split val --model_id test --rootpath /mnt/petrelfs/huyutao/record/lavit_hu1/ \
--resume /mnt/petrelfs/huyutao/record/lavit_hu1/h14_1111/model_best_h14_1111.pth --workers 4 --ddp_trained_weights --window12 --img_size 480