#!/bin/bash

# run the training script
python train.py \
  --model_name t5-base \
  --train_file medquad.csv \
  --output_dir rag_model \
  --batch_size 8 \
  --epochs 3 \
  --lr 5e-5 \
  --max_input_length 512 \
  --max_output_length 150 \
  --device mps \
  --logging_steps 10

# run the FAISS index creation script
python create_faiss_index.py

# run the inference script
python inference.py \
  --query "What causes glaucoma?" \
  --model_dir rag_model \
  --csv_file medquad_with_context.csv \
  --index_file context.index \
  --device mps \
  --top_k 3
