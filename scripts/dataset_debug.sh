config_file='configs/t_dk_25e.json'
search_name= 'max_data_25_epochs'

python3 scripts/Step1-CheckFiles.py --experiment_config_path "$config_file"
python3 scripts/Step2-ModelTrainScheduler.py --experiment_config_path "$config_file" --search_name "$search_name"