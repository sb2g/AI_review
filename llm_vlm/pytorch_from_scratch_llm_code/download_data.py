"""
Helper code to download datasets
Reference: https://github.com/rasbt/LLMs-from-scratch
"""

# Download a sample text file

import os
import urllib.request
import zipfile
from pathlib import Path
import json

# Sample text file for pretraining
if 0: 
    file_path = "../data/the-verdict.txt"
    if not os.path.exists(file_path):
        url = ("https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt")
        urllib.request.urlretrieve(url, file_path)


# Example dataset for LLM classifier
if 0:

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "../data/sms_spam_collection.zip"
    extracted_path = "../data/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    if not os.path.exists(data_file_path):
        try:
            # Download the file
            with urllib.request.urlopen(url) as response:
                with open(zip_path, "wb") as out_file:
                    out_file.write(response.read())
            # Unzip the file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)
            # Add .tsv file extension
            original_file_path = Path(extracted_path) / "SMSSpamCollection"
            os.rename(original_file_path, data_file_path)
            print(f"File downloaded and saved as {data_file_path}")

        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
            print(f"Primary URL failed: {e}. Try with backup URL {backup_url}")


# Example dataset for instruct LLM
if 0:

    file_path = "../data/instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    # with open(file_path, "r") as file:
    #     data = json.load(file)

