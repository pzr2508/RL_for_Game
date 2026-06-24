import os
import logging
import datetime
from termcolor import colored
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        colors = {
            logging.DEBUG: 'cyan',
            logging.INFO: 'green',
            logging.WARNING: 'yellow',
            logging.ERROR: 'red',
            logging.CRITICAL: 'magenta'
        }
        return colored(msg, colors.get(record.levelno, 'white'))


class DailyLogger:
    """每天自动创建新的 YYYY-MM-DD.log 文件"""

    def __init__(self, log_dir="logs", backup_days=15):
        self.log_dir = log_dir
        self.backup_days = backup_days
        self.current_date = None
        self.logger = logging.getLogger("DailyLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # 只添加一次控制台 handler
        if not self.logger.handlers:
            console = logging.StreamHandler()
            console.setFormatter(
                ColoredFormatter('[%(levelname)s][%(filename)s:%(lineno)d][%(asctime)s]: %(message)s')
            )
            self.logger.addHandler(console)

        # 初始化第一次文件 handler
        self._update_file_handler_if_needed()

    def _get_today_log_path(self):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        return today, os.path.join(self.log_dir, f"{today}.log")

    def _update_file_handler_if_needed(self):
        today, log_path = self._get_today_log_path()
        if self.current_date == today:
            return

        # 日期变了，更新 handler
        self.current_date = today

        # 删除旧的 file handler
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)
                h.close()

        # 创建目录
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # 创建新的 file handler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            ColoredFormatter('[%(levelname)s][%(filename)s:%(lineno)d][%(asctime)s]: %(message)s')
        )
        self.logger.addHandler(file_handler)

        # 清理旧日志
        self._cleanup_old_logs()

    def _cleanup_old_logs(self):
        threshold_date = datetime.datetime.now() - datetime.timedelta(days=self.backup_days)

        for filename in os.listdir(self.log_dir):
            if filename.endswith(".log"):
                try:
                    date_str = filename.split(".")[0]
                    file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < threshold_date:
                        file_path = os.path.join(self.log_dir, filename)
                        self.logger.info(f"删除日志文件: {file_path}")
                        os.remove(file_path)
                except:
                    pass

    def debug(self, msg):
        self._update_file_handler_if_needed()
        self.logger.debug(msg)

    def info(self, msg):
        self._update_file_handler_if_needed()
        self.logger.info(msg)

    def error(self, msg):
        self._update_file_handler_if_needed()
        self.logger.error(msg)

    def warning(self, msg):
        self._update_file_handler_if_needed()
        self.logger.warning(msg)

    def debug(self, msg):
        self._update_file_handler_if_needed()
        self.logger.debug(msg)

    def critical(self, msg):
        self._update_file_handler_if_needed()
        self.logger.critical(msg)


# 使用示例
logger = DailyLogger("logs")

# my_logger.info("系统启动")
