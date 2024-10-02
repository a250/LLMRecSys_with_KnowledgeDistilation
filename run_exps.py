from logging import getLogger
import yaml
from pathlib import PurePath, Path
import argparse
from copy import deepcopy
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.utils import init_logger
import torch

from utils.utils import (
    get_model, 
    process_llm_emb,
    advanced_data_preparation,
    # save_user_item_id,
    # load_user_item_id
)

from utils.tmanager import TrainManager
from utils.journal import ExperimentsJournal

def run_exeriment(config_file_pth):
    
    with open(config_file_pth) as file:
        config_dict = yaml.safe_load(file)  
        
        
    model_class = get_model(config_dict["model"])

    # init_seed is called automatically
    config = Config(model=model_class, config_dict=config_dict)

    if 'seed' in config:
        torch.manual_seed(config['seed'])
        
    init_logger(config)
    logger = getLogger()
    logger.setLevel(config_dict["logger_level"])
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    datasets = advanced_data_preparation(config, deepcopy(dataset))


    expj = ExperimentsJournal(config_file_pth, dataset)
    process_llm_emb(config, dataset)  
    
    tmanager = TrainManager(config, model_class, datasets)
    tmanager.set_journal(expj)
    tmanager.run()
    tmanager.journal.save_results()
    
    return tmanager


def config_getter(exp_pth):
    if not Path(exp_pth).is_dir():
        return [exp_pth]
    
    return sorted([cfg for cfg in Path(exp_pth).iterdir() if cfg.suffix == '.yaml'])


# with open("./experiments/exp_0_NeuMF_ml1m.yaml") as file:
#     config_dict = yaml.safe_load(file)

if __name__ == "__main__":

    start_with = None    
    parser = argparse.ArgumentParser(
        description='Run experiments based on given .yaml config'
        )
    parser.add_argument('config_file', type=PurePath)
    parser.add_argument('--start_with', type=int)

    args = parser.parse_args()
    config_file = args.config_file
    start_with = args.start_with

    exp_config_files = config_getter(config_file)[start_with:]

    for exp_config_file in exp_config_files:
        print(f'Run experiment: {exp_config_file}')
        tm = run_exeriment(exp_config_file)
