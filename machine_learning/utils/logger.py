from datetime import datetime
from tqdm import tqdm

OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

now = datetime.now()

class LogHandler:

  def __init__(self, error_file_path='error.log', info_file_path='info.log'):
    self._error_file_path = error_file_path
    self._error_file = open(self._error_file_path, 'w')
    self._info_file_path = info_file_path
    self._info_file = open(self._info_file_path, 'w')

  def log_info_timestamp(self, info_message):
    self.log_info(f'{now.strftime("%d/%m/%Y %H:%M:%S")} - {info_message}')

  def log_error_timestamp(self, error_message):
    self._log_error(f'{now.strftime("%d/%m/%Y %H:%M:%S")} - {error_message}')

  def log_warning_timestamp(self, warning_message):
    self.log_warning(f'{now.strftime("%d/%m/%Y %H:%M:%S")} - {warning_message}')

  def _log_error(self, error_message):
    tqdm.write(f'{FAIL}Logging error: {error_message}{ENDC}')
    self._error_file.write(error_message)
    self._error_file.write('\n')

  def log_info(self, info_message):
    print(f'{OKGREEN}Logging info: {info_message}{ENDC}')
    self._info_file.write(info_message)
    self._info_file.write('\n')

  def log_warning(self, warning_message):
    print(f'{WARNING}Logging warning: {warning_message}{ENDC}')
    self._info_file.write(warning_message)
    self._info_file.write('\n')
    
  def using_loading_bar(self, data, total=None, description="Loading"):
    return tqdm(data, desc=description,
                colour='White',
                ncols=100,
                delay=0.5,
                position=0,
                total=total,
                bar_format="{desc}: {percentage:.1f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}]")
