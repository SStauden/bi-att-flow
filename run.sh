#!/usr/bin/env bash
source_path=$1
target_path=$2
inter_dir="inter"
load_path="01"
debug="False"
python3 -m squad.prepro_simple --mode single --single_path $source_path --debug $debug --target_dir $inter_dir --glove_dir .
python3 -m basic.cli --data_dir $inter_dir --nodump_eval --answer_path $target_path --load_path $load_path --draft $debug --eval_num_batches 0 --mode forward --attention --batch_size 8 --use_glove_for_unk --known_if_glove --nofinetune --notraditional