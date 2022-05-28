from napari_plugin_engine import napari_hook_implementation
from . import _reader

@napari_hook_implementation
def svsOutMemory(path):
    return _reader.napari_get_reader(path, in_memory = False)
