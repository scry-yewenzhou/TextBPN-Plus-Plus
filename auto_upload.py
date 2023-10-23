"""
Automatically upload checkpoints to S3

Usage
- cd ~/TextBPN-Plus-Plus/
- conda activate textbpn++
- specify input_folder and s3 path
- python auto_upload.py
"""
import os
import time
import logging


if __name__ == "__main__":
    input_folder = "./model/TotalText-no-pretrain-reproduce/TD500"
    while True:
        logging.info("prepare to upload")
        models = [x for x in os.listdir(input_folder) if x.endswith(".pth")]
        for model in models:
            command = f"aws s3 mv {os.path.join(input_folder, model)} s3://collatio-nnocr/OCRBenchmark/TextBPNPlusPlus-TotalText-Reproduce-ResNet50-101923/"
            os.system(command)
        time.sleep(120)
