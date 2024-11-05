import os
import yaml

# Global configuration variables
MQ_SERVER_IP = None
FILE_SERVER_IP = None
HIDDEN_PATH = None
MINIO_ACCESS_KEY = None
MINIO_SECRET_KEY = None
MINIO_SERVER_IP = None
MINIO_PORT = None


def load_config(config_file='local.yml'):
    # Get the absolute path of the directory where this script resides
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the config file
    config_path = os.path.join(base_dir, config_file)

    with open(config_path, 'r', encoding='utf-8') as file:
        yaml_content = yaml.safe_load(file)

    config_section = yaml_content['Default']
    print('config_section', config_section)

    config_dict = {
        'MQ_SERVER_IP': config_section.get('MQ_SERVER_IP', '127.0.0.1'),
        'FILE_SERVER_IP': config_section.get('FILE_SERVER_IP', '127.0.0.1'),
        'HIDDEN_PATH': config_section.get('HIDDEN_PATH', '/temp'),
        'MINIO_ACCESS_KEY': config_section.get('MINIO_ACCESS_KEY', 'minioadmin'),
        'MINIO_SECRET_KEY': config_section.get('MINIO_SECRET_KEY', 'minioadmin'),
        'MINIO_SERVER_IP': config_section.get('MINIO_SERVER_IP', '127.0.0.1'),
        'MINIO_PORT': config_section.get('MINIO_PORT', 9000),
    }

    return config_dict
