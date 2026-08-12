# main.py
import yaml
import argparse
from chemembed_single_file import run

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description='Process and predict mass spectrometry data.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    run(config)

if __name__ == "__main__":
    main()
