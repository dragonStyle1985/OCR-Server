# 湖北文理学院
# MQ_SERVER_IP = '172.33.38.218'
# FILE_SERVER_IP = '172.33.38.226'
# HIDDEN_PATH = '/data/work/html/outFile'


import yaml

# 设置配置文件的路径
CONFIG_FILE_PATH = '/usr/src/app/ocr_config.yml'

config = {}

# 尝试加载配置文件
try:
    with open(CONFIG_FILE_PATH, 'r') as file:
        config = yaml.safe_load(file)
        print(f"Configuration loaded from {CONFIG_FILE_PATH}")
except FileNotFoundError:
    print(f"Configuration file not found at {CONFIG_FILE_PATH}. Using default configuration.")

# 从配置中读取参数，如果没有则使用默认值
MQ_SERVER_IP = config.get('MQ_SERVER_IP', '127.0.0.1')
FILE_SERVER_IP = config.get('FILE_SERVER_IP', '127.0.0.1')
HIDDEN_PATH = config.get('HIDDEN_PATH', '/temp')
MINIO_ACCESS_KEY = config.get('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = config.get('MINIO_SECRET_KEY', 'minioadmin')
MINIO_SERVER_IP = config.get('MINIO_SERVER_IP', '127.0.0.1')
MINIO_PORT = config.get('MINIO_PORT', 9000)
