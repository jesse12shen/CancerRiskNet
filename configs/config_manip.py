import json
import os.path
file_w = {
  "search_space": {
    "metadata":["data/sample_diag_data.json"],
    "train":[True],
    "dev":[True],
    "test":[False],
    "epochs":[2],
    "num_workers":[4],
    "cuda":[True],
    "hidden_dim":[256],
    "model_name": ["transformer"],
    "optimizer": ["adam"],
    "init_lr":[1e-04],
    "train_batch_size": [4],
    "eval_batch_size": [4],
    "num_heads":[8],
    "max_batches_per_train_epoch": [5],
    "max_batches_per_dev_epoch": [5],
    "max_events_length": [1000],
    "max_eval_indices": [10],
    "pad_size": [200],
    "exclusion_interval": [0,3,6],
    "eval_auroc": [True],
    "eval_auprc": [True],
    "eval_c_index": [True],
    "use_known_risk_factors": [True, False],
    "data_setting_path": ["data/settings_sample_data.yaml"]
  },
  "available_gpus": [1]
}
file_path = '/home/jubal/CancerRiskNet/configs/transformer_only_shortened_search2.json'
with open(file_path, 'w') as json_file:
    json.dump(file_w,json_file)

print('file created')