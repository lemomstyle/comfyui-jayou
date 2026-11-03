from .nodes import JayouSDRAW
from .align_node import JayouSDRAWAlign
NODE_CLASS_MAPPINGS = {"JayouSDRAW": JayouSDRAW, "JayouSDRAWAlign": JayouSDRAWAlign}
NODE_DISPLAY_NAME_MAPPINGS = {"JayouSDRAW": "嘉游SDRAW", "JayouSDRAWAlign": "嘉游SDRAW · 图像对齐"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
