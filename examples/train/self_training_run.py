from omegaconf import OmegaConf
from rex.utils.config import ConfigParser
from rex.utils.logging import logger
from rex.utils.initialization import init_all

from src.task.self_training import SelfTrainingTextClassificationTask


def main():
    config = ConfigParser.parse_cmd()
    init_all(config.task_dir, config.random_seed, True, config)
    logger.info(OmegaConf.to_object(config))
    task = SelfTrainingTextClassificationTask(config)
    logger.info(f"task: {type(task)}")
    
    if config.train_type == 'base_train':
        task.train()
    elif config.train_type == 'self_train':
        task.self_training()
    elif config.train_type == 'pst_train':
        task.pst_training()
    else:
        raise ValueError(f'error train_type {config.train_type}')
        

if __name__ == "__main__":
    main()
