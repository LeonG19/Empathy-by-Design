# LLM-Healthcare

## Overview

LLM-Healthcare is a comprehensive framework for evaluating and aligning large language models for medical question answering. The project implements preference-based fine-tuning using Direct Preference Optimization (DPO), evaluates models across semantic, factual, and human-centric dimensions, and integrates RAG-based retrieval and embedding similarity evaluation.

This repository includes all core modules for dataset preprocessing, fine-tuning, embedding evaluation, factual and empathy metrics, and main execution pipelines.

## Environment Setup

Before running the project, install all required dependencies:

```
pip install -r requirements.txt
```

Make sure your OpenAI and Hugging Face API keys are properly configured:

```
export OPENAI_API_KEY="your_openai_key"
export HF_TOKEN="your_huggingface_token"
```

## Project Structure

```
source_code/
│
├── train/
│   ├── dpo_LLM.py              # Fine-tuning script using TRL DPOTrainer
│   ├── sft_LLM.py              # Optional SFT baseline training script
│
├── indexing/
│   ├── create_index.py         # Builds FAISS or Milvus index for RAG pipeline
│   ├── embed_docs.py           # Embedding creation for documents
│
├── llm/
│   ├── llm_base.py        # Class to manage llm deployment and generation
│   ├── init.py     
│
└── main.py                     # Main evaluation entry point
└── config.py                   # Class to manage project configurations
└── metrics.py
```

## Fine-Tuning (DPO Alignment)

To fine-tune a base model using Direct Preference Optimization (DPO):

```
python source_code/train/dpo_LLM.py \
  --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
  --dataset_path "datasets/healthcare_qa_pairs.json" \
  --output_dir "checkpoints/dpo_finetuned" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 3
```

This script uses TRL’s DPOTrainer for preference-based fine-tuning with "chosen/rejected" response pairs.

## Index Creation for RAG Pipeline

To build the vector index for retrieval-augmented evaluation:

```
python source_code/indexing/create_index.py \
  --input_file "datasets/gdpr_chunks_with_metadata.csv" \
  --db_type "milvus" \
  --collection_name "gdpr_index"
```

If you are using FAISS instead of Milvus:

```
python source_code/indexing/create_index.py \
  --input_file "datasets/gdpr_chunks_with_metadata.csv" \
  --db_type "faiss" \
  --output_index "indexes/gdpr_index.faiss"
```

You can also embed documents first:

```
python source_code/indexing/embed_docs.py \
  --input_file "datasets/gdpr_chunks.csv" \
  --model "sentence-transformers/all-mpnet-base-v2"
```

## Running Main Evaluation

Once models are fine-tuned and indexes are built, run the main evaluation pipeline:

```
python source_code/main.py \
  --eval_config "configs/eval_config.yaml" \
  --predictions_file "outputs/predictions.json" \
  --references_file "datasets/reference_answers.json" \
  --metrics "semantic,factual,empathy,readability"
```

This script computes semantic similarity (embedding-based), factual correctness (GEval), and human-centric metrics (empathy and Flesch–Kincaid readability). It outputs detailed metric tables and summaries.

## Outputs

After execution, the following outputs are generated under `outputs/`:

- results_summary.json — consolidated metric scores
- semantic_scores.csv — embedding similarity per sample
- factual_scores.csv — factual correctness
- empathy_scores.csv — empathy evaluation
- readability_scores.csv — Flesch–Kincaid readability results
- benchmark_table.tex — LaTeX-formatted results for inclusion in publications

## Citation

If you use this repository in your research, please cite:

Garza, L., Kotal, A., Piplai, A., Elluri, L., Das, P., & Chadha, A. 
"LLM-Healthcare: Direct Preference Optimization for Medically Aligned Language Models." 
University of Texas at El Paso, 2025.
