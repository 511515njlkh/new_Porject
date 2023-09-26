



CUDA_VISIBLE_DEVICES=0  python batch_lysozyme_gen.py --code 3 \
              --fname data/samples \
              --num_sample_batches 10 \
              --batch_size 32 \
              --gen_length 256 \
              --top_p 0.5 \
              --top_k 0 \
              --rep_penalty 1.2 \
&

CUDA_VISIBLE_DEVICES=1  python batch_lysozyme_gen.py --code 4 \
              --fname data/samples \
              --num_sample_batches 10 \
              --batch_size 32 \
              --gen_length 256 \
              --top_p 0.5 \
              --top_k 0 \
              --rep_penalty 1.2 \
