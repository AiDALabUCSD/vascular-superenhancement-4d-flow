from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

def _load_yaml(name: str = "default") -> dict:
    cfg_file = _CONFIG_DIR / f"{name}.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config file '{cfg_file}' not found")
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)

@dataclass(frozen=True, slots=True)
class PathConfig:
    base_dir: Path
    project_name: str
    dataset: str
    
    @property
    def project_root(self) -> Path:
        return self.base_dir / 'projects' / self.project_name
    
    @property
    def repository_root(self) -> Path:
        return self.base_dir / 'repository' / self.project_name
    
    @property
    def zipped_dir(self) -> Path:
        return self.repository_root / 'zipped_files'
    
    @property
    def unzipped_dir(self) -> Path:
        return self.repository_root / 'unzipped_files'

# global singleton (defaults to config/default.yaml)
path_config = PathConfig(**_load_yaml("default"))