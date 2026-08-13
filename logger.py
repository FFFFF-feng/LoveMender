#全局日志系统:同时输出到控制台和文件
#所有模块通过From logger import logger 来使用
import logging#导入logging模块,用于日志记录
import sys#导入sys模块,用于获取当前脚本的路径
from pathlib import Path#导入Path类,用于处理文件路径
from datetime import datetime#导入datetime类,用于生成日志文件名

def setup_logger():
    """
    配置全局日志系统
    -控制台:INFO级别,简介格式
    -文件:DEBUG级别,详细格式,包含时间戳,用于排查问题
    -日志文件按日期命名,存在logs/目录下
    :return:
    """
    log_dir = Path("logs")#日志目录
    log_dir.mkdir(parents=True, exist_ok=True)#创建日志目录,如果不存在则创建
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"#日志文件路径
    #创建日志记录器
    logger=logging.getLogger("LoveMender")
    logger.setLevel(logging.DEBUG)#设置日志级别为DEBUG,记录所有日志

    #避免重复添加handler(streamlit重跑的时候会重新import)
    if logger.handlers:
        return logger

    #控制台输出级别handler:INFO级别,简洁格式
    console_handler = logging.StreamHandler(sys.stdout)#创建控制台输出处理器,输出到标准输出流(sys.stdout)
    console_handler.setLevel(logging.INFO)#设置日志级别为INFO,只记录INFO及以上日志
    console_handler.setFormatter(#设置日志格式为:时间-日志级别-日志消息
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",#日志格式:时间-日志级别-日志消息
            datefmt="%H:%M:%S"#设置时间格式为:时:分:秒
        )
    )

    #文件handler:DEBUG级别,详细格式,包含时间戳,用于排查问题
    file_handler=logging.FileHandler(log_file,encoding="utf-8")#创建文件输出处理器,输出到日志文件
    file_handler.setLevel(logging.DEBUG)#设置日志级别为DEBUG,记录所有日志
    file_handler.setFormatter(#设置日志格式为:时间-日志级别-日志消息-日志行号
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"#设置时间格式为:年-月-日 时:分:秒
        )
    )

    logger.addHandler(console_handler)#添加控制台输出处理器到日志记录器
    logger.addHandler(file_handler)#添加文件输出处理器到日志记录器

    logger.info("日志系统配置完成,日志文件_file: %s", log_file)
    return logger

#全局日志记录器,所有模块通过From logger import logger 来使用
logger = setup_logger()

