"""MatDeplot: paper-aligned curve extraction for materials-science charts."""

from .config import MatDeplotConfig, load_paper_config
from .pipeline import MatDeplotPipeline

__all__ = ["MatDeplotConfig", "MatDeplotPipeline", "load_paper_config"]
__version__ = "0.1.0"
