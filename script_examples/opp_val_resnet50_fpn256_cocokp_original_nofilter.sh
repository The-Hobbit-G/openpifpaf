#!/bin/bash -l

shopt -s extglob


echo STARTING AT `date`


evalxp='opp_train_resnet50_fpn256_cocokp_original_nofilter'
evalepoch=150


xpdir="/scratch/izar/jiguo/val/cocokp/resnet50_fpn256_original/"

mkdir -p ${xpdir}
# mkdir -p ${xpdir}/code
# tar -czvf ${xpdir}/code/code.tar.gz /home/jiguo/openpifpaf
mkdir -p ${xpdir}/logs


cd ..

mkdir -p ${xpdir}/predictions

for cpepoch in ${evalepoch}
do
  evalfrom="/scratch/izar/jiguo/train/cocokp/resnet50_fpn256_original/${evalxp}/checkpoints/${evalxp}.pt.epoch${cpepoch}"
  echo "Start evaluating ${evalfrom}..."
  python3 -m openpifpaf.eval \
    --write-predictions \
    --output ${xpdir}/predictions/${SLURM_JOB_NAME}_epoch${cpepoch} \
    --dataset=cocokp \
    --coco-eval-long-edge 641 \
    --cocokp-upsample=2 \
    --coco-no-eval-annotation-filter \
    --batch-size 1 \
    --loader-workers 8 \
    --checkpoint ${evalfrom} \
    --decoder cifcaf:0 \
    --base_stride 4 8 16 \
    --seed-threshold 0.2 \
    --force-complete-pose \
    2>&1 | tee ${xpdir}/logs/${SLURM_JOB_NAME}_epoch${cpepoch}.log
  echo "Done evaluating ${evalfrom}"
done

cd -


echo FINISHED at `date`
