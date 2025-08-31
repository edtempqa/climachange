import yaml

try:
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
except (FileNotFoundError, yaml.YAMLError) as e:
    raise RuntimeError(f"Error loading config.yaml: {e}")

DOWNLOAD_DATA= cfg['data']['DOWNLOAD_DATA']
ENGINE = cfg['data']['ENGINE']
DATA_SOURCE = cfg['data']['DATA_SOURCE']
DATA_FILE_ZIP = cfg['data']['DATA_FILE_ZIP']
DATA_FILE = cfg['data']['DATA_FILE']

TEMPERATURE_FIELD_KELVIN = cfg['variables']['TEMPERATURE_FIELD_KELVIN']
TEMPERATURE_FIELD_CELSIUS = cfg['variables']['TEMPERATURE_FIELD_CELSIUS']
CO2_FIELD = cfg['variables']['CO2_FIELD']
CO2_FIELD_PPM = cfg['variables']['CO2_FIELD_PPM']
TIME_FIELD = cfg['variables']['TIME_FIELD']
TIME_VARIABLE = cfg['variables']['TIME_VARIABLE']
CO2_VARIABLE = cfg['variables']['CO2_VARIABLE']

RETRAIN= cfg['training']['RETRAIN']
MODEL_PATH = cfg['training']['MODEL_PATH']
SCALER_PATH = cfg['training']['SCALER_PATH']
SEQ_LENGTH = int(cfg['training']['SEQ_LENGTH'])
N_PREDICT = int(cfg['training']['N_PREDICT'])
EPOCHS = int(cfg['training']['EPOCHS'])