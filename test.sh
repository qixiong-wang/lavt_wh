mkdir /mnt/petrelfs/huyutao/record/lavit_hu1/test
cp ./run1.sh /mnt/petrelfs/huyutao/record/lavit_hu1/test
cp ./*.py /mnt/petrelfs/huyutao/record/lavit_hu1/test
cp -r ./lib /mnt/petrelfs/huyutao/record/lavit_hu1/test
srun --partition=Gvadapt  --mpi=pmi2 --gres=gpu:1 -N 1  --job-name=test --kill-on-bad-exit=1 --pty --mail-type=ALL \
python test.py --model lavt --swin_type base --dataset refcoco --split val --resume /mnt/petrelfs/huyutao/record/lavit_hu1/h14_1111/model_best_h14_1111