from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _PROJECT_ROOT / "config"

def _load_yaml(name: str = "default") -> dict:
    cfg_file = _CONFIG_DIR / f"{name}.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config file '{cfg_file}' not found")
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)
    
def load_path_config(name: str = "default") -> PathConfig:
    raw = _load_yaml(name)
    return PathConfig(**raw)

@dataclass(frozen=True, slots=True)
class PathConfig:
    base_data_dir: Path
    base_working_dir: Path
    project_name: str
    dataset: str

    def __post_init__(self):
        object.__setattr__(self, "base_data_dir", Path(self.base_data_dir))
        object.__setattr__(self, "base_working_dir", Path(self.base_working_dir))

    @property
    def project_root(self) -> Path:
        return self.base_data_dir / "projects" / self.project_name / self.dataset

    @property
    def repository_root(self) -> Path:
        return self.base_data_dir / "repository" / self.project_name / self.dataset

    @property
    def zipped_dir(self) -> Path:
        return self.repository_root / "zipped_files"

    @property
    def unzipped_dir(self) -> Path:
        return self.repository_root / "unzipped_files"

    @property
    def working_dir(self) -> Path:
        # Working directory is now inside the repository
        return _PROJECT_ROOT / "working_dir" / self.dataset
