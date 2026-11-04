from .nodes import JayouSDRAWPython, JayouSDRAWAlignPython
NODE_CLASS_MAPPINGS = {'JayouSDRAWPython': JayouSDRAWPython, 'JayouSDRAWAlignPython': JayouSDRAWAlignPython}
NODE_DISPLAY_NAME_MAPPINGS = {'JayouSDRAWPython': '嘉游SDRAW · Python', 'JayouSDRAWAlignPython': '嘉游SDRAW · 对齐 Python'}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
