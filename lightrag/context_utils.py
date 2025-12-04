from contextvars import ContextVar

# 定义全局上下文变量
# default="default" 保证了如果没传 workspace，系统也能正常运行（兼容旧逻辑）
_current_workspace_cv: ContextVar[str] = ContextVar("current_workspace", default="default")

def get_current_workspace() -> str:
    """获取当前协程的 workspace"""
    return _current_workspace_cv.get()

def set_current_workspace(workspace: str):
    """设置当前协程的 workspace，返回 token 用于重置"""
    return _current_workspace_cv.set(workspace)

def reset_current_workspace(token):
    """重置上下文，防止污染后续请求"""
    _current_workspace_cv.reset(token)