from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import yaml

# Try to find the project root - handle both development and installed scenarios
def _find_project_root():
    # Try multiple starting points to find the project root
    search_paths = [
        Path.cwd(),  # Current working directory
        Path(__file__).resolve().parents[3],  # From this file
        Path.home() / "vascular-superenhancement-4d-flow",  # Common location
    ]
    
    for start_path in search_paths:
        for parent in [start_path] + list(start_path.parents):
            if (parent / "pyproject.toml").exists() and (parent / "hydra_configs").exists():
                print(f"Found project root at: {parent}")
                return parent
    
    # If we get here, we couldn't find the project root
    raise FileNotFoundError("Could not find project root with pyproject.toml and hydra_configs")

_PROJECT_ROOT = _find_project_root()
_CONFIG_DIR = _PROJECT_ROOT / "hydra_configs" / "path_config"

def _load_yaml(name: str = "default") -> dict:
    cfg_file = _CONFIG_DIR / f"{name}.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config file '{cfg_file}' not found")
    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)
    
def load_path_config(name: str = "default") -> PathConfig:
    print(f"LISTEN {_CONFIG_DIR}")
    raw = _load_yaml(name)
    raw.pop("path_config_name", None)  # Drop this key if present
    raw.pop("splits_path", None)  # Drop this key if present
    return PathConfig(**raw)

@dataclass(frozen=True, slots=True)
class PathConfig:
    base_data_dir: Path
    base_working_dir: Path
    project_name: str
    dataset: str
    database_file: str

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
    def database_path(self) -> Path:
        return self.repository_root / self.database_file

    @property
    def zipped_dir(self) -> Path:
        return self.repository_root / "zipped_files"

    @property
    def unzipped_dir(self) -> Path:
        return self.repository_root / "unzipped_files"

    @property
    def working_dir(self) -> Path:
        """Return the working directory for this dataset.
        
        The working directory is where all generated files and data will be stored.
        It is located at <base_working_dir>/<project_name>/working_dir/<dataset>.
        """
        return self.base_working_dir / self.project_name / "working_dir" / self.dataset
