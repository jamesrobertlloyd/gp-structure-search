import config
import os
import tempfile

def mkstemp_safe(directory, suffix):
    (os_file_handle, file_name) = tempfile.mkstemp(dir=directory, suffix=suffix)
    os.close(os_file_handle)
    return file_name

def create_temp_file(extension):
    return mkstemp_safe(config.TEMP_PATH, extension)

