import os
import sys

from flask import Flask

# Tambahkan jalur proyek ke sys.path
project_home = '/home/riskalathifah/mysite'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Import aplikasi Flask dan namai 'application' agar sesuai dengan konvensi WSGI
from main import \
    app as application  # Ganti 'main' dengan nama modul aplikasi Flask Anda

if __name__ == "__main__":
    application.run()