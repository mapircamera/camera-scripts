# -*- coding: utf-8 -*-
"""
@author: MAPIR, Inc
"""

import subprocess
import os

class ExifUtils:

    @staticmethod
    def copy_simple(inphoto, outphoto, startup_info):
        mod_path = os.path.dirname(os.path.realpath(__file__))
        exifout = subprocess.run(
            [mod_path + os.sep + r'exiftool.exe',
            r'-overwrite_original_in_place', r'-tagsFromFile',
            os.path.abspath(inphoto),
            r'-all:all<all:all',
            os.path.abspath(outphoto)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
            startupinfo=startup_info).stderr.decode("utf-8")

        data = subprocess.run(
                    args=[mod_path + os.sep + r'exiftool.exe', '-m', r'-ifd0:imagewidth', r'-ifd0:imageheight', os.path.abspath(inphoto)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE, startupinfo=startup_info).stderr.decode("utf-8")