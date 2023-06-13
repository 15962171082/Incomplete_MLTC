from omegaconf import OmegaConf
from rex.utils.config import ConfigParser
from rex.utils.logging import logger
from rex.utils.initialization import init_all

from src.task.self_training import HTTNTask


def main():
    config = ConfigParser.parse_cmd()
    init_all(config.task_dir, config.random_seed, True, config)
    logger.info(OmegaConf.to_object(config))
    task = HTTNTask(config)
    logger.info(f"task: {type(task)}")
    
    # task.train_head()
    task.train_transfor()        

if __name__ == "__main__":
    main()
