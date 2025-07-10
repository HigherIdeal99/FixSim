# Quantized-LLM Bit-Width Sweeper

This repository offers a single-file driver that **explores fixed-point bit settings** for every arithmetic node of a Llama-style model.  
The script runs a loop that

1. picks a candidate bit width for one node,  
2. evaluates language quality with **perplexity** on WikiText,  
3. checks a light **MMLU anatomy quiz** for functional accuracy, and  
4. records the result to a timestamped log.

The loop stops for each node as soon as quality stays inside a user-defined safety margin.  
The final tensor called `analyzer` holds the chosen bit widths and can be re-loaded for deployment.

This fixed-point simulation scripts needs local llm api.
We used official LLaMA repogitory.

---

## Quick start

```bash
python sweep.py run-main \
  --ckpt_dir /path/to/Llama3.2-1B-Instruct \
  --max_seq_len 2048 \
  --max_batch_size 1 \
  --num_samples 16
