#set GLUE_DIR to /path/to/glue
#export GLUE_DIR=/export/share/akhilesh-gotmare/transformers/glue_data
export GLUE_DIR=/export/home/ProGen/discriminators/fluorescence/protein_tasks
export TASK_NAME=data_adv

## if training on a single GPU, remove
#  -m torch.distributed.launch \
#     --nproc_per_node 8
# from the below command
# results might differ to a slight extent, since the final evaluators will be different due to different effective batchsize used

#final RoBERTa model saved at --output_dir


python get_disc_score.py --model_type bert \
    --model_name_or_path /export/share/bkrause/progen/discriminators/adversarial_top_p_1 \
    --task_name SST-2 \
    --dropout 0.1 \
    --do_train \
    --do_eval \
    --synth_file ../data/samples_code_1.txt\
    --max_seq_length 256 \
    --per_gpu_train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --output_dir disc-bert \
    --overwrite_output_dir \
    --save_steps 500000000 \
    --logging_steps 50 \
