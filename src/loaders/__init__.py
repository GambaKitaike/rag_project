from .altair_manual_loader import (
    AltairManualLoaderConfig,
    build_hyperlink_graph,
    load_altair_manual_documents,
)

# Backward compatibility with the old module-level config name
AltairLoaderConfig = AltairManualLoaderConfig

__all__ = [
    "AltairLoaderConfig",
    "AltairManualLoaderConfig",
    "build_hyperlink_graph",
    "load_altair_manual_documents",
]
