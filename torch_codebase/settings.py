import os
import getpass
import logging
from git import Repo

base_dir = os.getcwd()
local_repo = Repo(os.path.abspath(os.path.join(base_dir,'..')))
branch_name = local_repo.active_branch.name


def get_logger(log_file_path):
    logger = logging.getLogger('workloglogger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file_path)

    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger



def get_log_dir(task_name):
    username = getpass.getuser()
    npk_cache_dir = os.path.join('/data', 'train_log')
    project_name = base_dir.split('/')[-1]

    # rtld: root train log dir
    npk_rtld = os.path.join(npk_cache_dir, username,
                            project_name, branch_name, task_name)
    os.makedirs(npk_rtld, exist_ok=True)
    print(
        "using data directory for train_log directory: {}".format(npk_rtld))

    return npk_rtld


def get_log_model_dir(task_project_name, task_name):
    username = getpass.getuser()
    project_name = base_dir.split('/')[-1]

    model_path = os.path.join(
        '/data/models/', username, project_name, branch_name, task_project_name, task_name)
    print(
        "Excellent! Model snapshots can be saved to: {}".format(model_path))
    return model_path


# vim: ts=4 sw=4 sts=4 expandtab
