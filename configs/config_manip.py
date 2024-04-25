import json
import os

# import os.path
file_w = {
  "search_space": {
    "metadata":["data/sample_diag_data.json"],
    # "metadata" : ["data/mimic_data.json"],
    "train":[True],
    "dev":[True],
    "test":[False], #we'll just use a single dataset
    "epochs":[25],
    "num_workers":[4],
    "cuda":[True],
    "hidden_dim":[256],
    "model_name": ["transformer"],
    "optimizer": ["adam"],
    "init_lr":[1e-04],
    "train_batch_size": [4],
    "eval_batch_size": [4],
    "num_heads":[16],  # that was in the SI
    "max_batches_per_train_epoch": [2500], # increasing training size
    "max_batches_per_dev_epoch": [249], # should be all
    "max_events_length": [1000],
    "max_eval_indices": [10],
    "pad_size": [200],
    "exclusion_interval": [0],
    "eval_auroc": [True],
    "eval_auprc": [True],
    "eval_c_index": [False],
    "use_known_risk_factors": [False],
    # "data_setting_path": ["data/settings_MIMIC.yaml"]
    "data_setting_path": ["data/settings_sample_data.yaml"]
  },
  "available_gpus": [1]
}
file_path = '/home/jubal/CancerRiskNet/configs/t_dk_25e.json'
with open(file_path, 'w') as json_file:
    json.dump(file_w,json_file)
print('This machine is capable of: ' + str(os.cpu_count()))
# print('file created')