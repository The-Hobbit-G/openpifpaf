#!/bin/bash -l

shopt -s extglob

export MASTER_ADDR=$(hostname)
export MASTER_PORT=7247
export TORCH_DISTRIBUTED_DEBUG=INFO


echo STARTING AT `date`


xpdir="/scratch/izar/jiguo/train/resnet50_fpn/debug/cocodet"
checkpoint="/scratch/izar/jiguo/train/cocodet/resnet50_fpn1024/opp_train_resnet50_fpn1024_cocodet_outstage345/checkpoints/opp_train_resnet50_fpn1024_cocodet_outstage345.pt.epoch117"
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
  --checkpoint=${checkpoint} \
  --necknet=FPN \
  --neck_in 1024 2048 \
  --neck_out=1024 \
  --num_outs=3 \
  --start_level=0 \
  --base_outstage 3 4 5 \
  --head_stride 8 16 32 \
  --add_extra_convs=on_output \
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