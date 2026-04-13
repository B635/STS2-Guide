def handle_api_error(e: Exception) -> str:
    error_type = type(e).__name__
    if "RateLimitError" in error_type:
        return "API 调用频率超限，请稍后再试"
    elif "AuthenticationError" in error_type:
        return "API key 无效，请检查 .env 文件"
    elif "ConnectionError" in error_type:
        return "网络连接失败，请检查网络"
    else:
        return f"发生错误：{e}"


def handle_file_error(e: Exception, filepath: str) -> str:
    if isinstance(e, FileNotFoundError):
        return f"找不到文件：{filepath}"
    elif isinstance(e, PermissionError):
        return f"没有权限读取文件：{filepath}"
    else:
        return f"读取文件失败：{e}"