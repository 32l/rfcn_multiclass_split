#!/bin/bash

[[ $# -eq 0 ]] && echo "$0 model" && exit 0
model=$1

d=`pwd`
current_dir=`basename $d`
iter=`echo $model |awk -F '.' '{print $1}' |awk -F '_' '{print $NF}'`
save_dir=rfcn_multiclass_split_${current_dir}_iter_$iter
#anno_path=~/data/libraf/benchmark/fovea/image_anno_set/base_6k
anno_path=/home/huyangyang/data/facedet_result_from_caiyang/img_label

#../../../lib/tools/rfcn_test.sh  $anno_path $model 3 --vis
#exit 0

../../../lib/tools/rfcn_test.sh  $anno_path $model 0 --save_dir=$save_dir
echo "begin roc task"
python ~/tools/roc_new.py --anno_file=$anno_path --save_dir=$save_dir --resized_height=104 --ious=0.4 --scale_ranges="(8,1000)"  --show --include_cls="1,2"
