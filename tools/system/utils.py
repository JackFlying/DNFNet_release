
import os
from flask import send_file
import zipfile
def process_zip_images():
    zip_filename = 'images.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, cache_dir))
    return send_file(zip_filename, as_attachment=True)