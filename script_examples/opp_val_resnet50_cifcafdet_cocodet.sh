#!/bin/bash -l


shopt -s extglob


echo STARTING AT `date`


evalxp='opp_train_resnet50_cifcafdet_cocodet_bg_mask_det_sigma0.8'
evalepoch=120


xpdir="/scratch/izar/jiguo/val/cocodet/cifcafdet/resnet50/${SLURM_JOB_NAME}"

mkdir -p ${xpdir}
# mkdir -p ${xpdir}/code
# tar -czvf ${xpdir}/code/code.tar.gz /home/jiguo/openpifpaf
mkdir -p ${xpdir}/logs


cd ..

mkdir -p ${xpdir}/predictions

for cpepoch in ${evalepoch}
do
  evalfrom="/scratch/izar/jiguo/train/cocodet/cifcafdet_no_fpnfilter/resnet50/${evalxp}/checkpoints/${evalxp}.pt.epoch${cpepoch}"
  echo "Start evaluating ${evalfrom}..."
  python3 -m openpifpaf.eval \
    --write-predictions \
    --output ${xpdir}/predictions/${SLURM_JOB_NAME}_epoch${cpepoch} \
    --dataset=cocodet \
    --coco-eval-long-edge 641 \
    --cocodet-upsample=2 \
    --coco-no-eval-annotation-filter \
    --batch-size 1 \
    --loader-workers 8 \
    --checkpoint ${evalfrom} \
    --decoder=cifcaf:0 \
    --seed-threshold 0.2 \
    --force-complete-pose \
    2>&1 | tee ${xpdir}/logs/${SLURM_JOB_NAME}_epoch${cpepoch}.log
  echo "Done evaluating ${evalfrom}"
done

cd -


echo FINISHED at `date`
