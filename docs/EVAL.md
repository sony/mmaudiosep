# Evaluation

## Batch Evaluation Process

To evaluate the model on a dataset, use the `batch_eval.py` script. This script is significantly more efficient for large-scale evaluations compared to `demo.py`, as it supports batched inference, multi-GPU inference, Torch compilation, and the ability to skip video compositions.


Below is an example of how to run this script using four GPUs:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4  batch_eval.py duration_s=10 dataset=vggclean model=large_44k num_workers=8
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4  batch_eval.py duration_s=10 dataset=music model=large_44k num_workers=8
```

Please ensure to update the data paths in `config/eval_data/base.yaml` as necessary.
More configuration options can be found in `config/base_config.yaml` and `config/eval_config.yaml`.

## Obtaining Quantitative Evaluation Metrics


**Note:** Our evaluation code is based on av-benchmark (https://github.com/hkchengrex/av-benchmark).  
The current setup is already functional, and we plan to release modifications soon to better accommodate our specific use case.