#!/bin/bash -l

shopt -s extglob

export MASTER_ADDR=$(hostname)
export MASTER_PORT=7247
export TORCH_DISTRIBUTED_DEBUG=INFO


echo STARTING AT `date`


xpdir="/scratch/izar/jiguo/train/resnet50_fpn/debug/cocodet"
mkdir -p ${xpdir}
# mkdir -p ${xpdir}/code
# tar -czvf ${xpdir}/code/code.tar.gz <path/to/code/dir>
mkdir -p ${xpdir}/logs


cd ..

mkdir -p ${xpdir}/checkpoints

python3 -m openpifpaf.train --output ${xpdir}/checkpoints/debug.pt \
  --dataset=cocodet \
  --cocodet-square-edge=513 \
  --cocodet-extended-scale \
  --cocodet-orientation-invariant=0.1 \
  --cocodet-upsample=2 \
  --basenet=resnet50 \
  --epochs=150 \
  --batch-size=8 \
  --lr=0.0001 \
  --lr-decay 130 140 \
  --lr-decay-epochs=10 \
  --momentum=0.9 \
  --weight-decay=1e-5 \
  --b-scale=10.0 \
  --clip-grad-value=10.0 \
  2>&1 | tee ${xpdir}/logs/debug.log

cd -


echo FINISHED at `date`