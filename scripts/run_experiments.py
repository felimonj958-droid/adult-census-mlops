from pathlib import Path
import subprocess
import sys


def main():
    config_dir = Path('configs/experiments')
    configs = sorted(config_dir.glob('*.yaml'))
    if not configs:
        raise FileNotFoundError('No experiment configs found.')

    for config in configs:
        print(f'Running {config}...')
        result = subprocess.run([sys.executable, 'src/train.py', '--config', str(config)], check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
