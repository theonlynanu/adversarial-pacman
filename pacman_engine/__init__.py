import importlib, sys

for _name in ("game", "layout", "util", "graphics_display"):
    sys.modules[_name] = importlib.import_module(f"pacman_engine.{_name}")