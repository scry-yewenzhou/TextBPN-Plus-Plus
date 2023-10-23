"""
Evaluate trained model on different epochs

Usage
- cd ~/TextBPN-Plus-Plus
- conda activate textbpn++
- specify model name
- python evaluate_epochs.py
"""
import os
import logging
import subprocess
import boto3
from pathlib import Path
import time
import shutil
from tqdm import tqdm


MODELS = {
    "TD500-reproduce-102023": {
        "prefix": "OCRBenchmark/TextBPNPlusPlus-TD500-Reproduce-ResNet50-102323/",
        "s3_path": "s3://collatio-nnocr/OCRBenchmark/TextBPNPlusPlus-TD500-Reproduce-ResNet50-102323",  # no / in the end
        "exp_name": "TD500",
        "save_model_dir": "/root/TextBPN-Plus-Plus/model/TD500-no-pretrain-reproduce",
        "backbone": "resnet50",
        "result_csv": "/root/TextBPN-Plus-Plus/model/TD500-no-pretrain-reproduce/TD500-eval.csv",  # csv of results
        "thresh": 65,  # start evaluating >= thresh epoch
        "eval_path": "TextBPN-Plus-Plus/output/Analysis/TD500_eval.txt",  # path to evaluation txt
    }
}


def find_all_iter_epochs(prefix: str) -> list:
    """Find all iteration epochs, return as a set

    Returns:
        set: looks like set("TextBPN_resnet50_70", "TextBPN_resnet50_75", ...)
    """
    res = set()
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("collatio-nnocr")
    # List objects within a given prefix
    for obj in bucket.objects.filter(Delimiter="/", Prefix=prefix):
        res.add(Path(obj.key).stem)
    res_list = list(res)
    res_list.sort()
    return res_list


def filter_epochs(epochs: list, threshold: int) -> list:
    """Keep all epochs that >= threshold"""
    res = []
    for e in epochs:
        num = e.rsplit("_", maxsplit=1)[1]
        if int(num) >= threshold:
            res.append(e)
    res.sort(key=lambda x: int(x.rsplit("_", maxsplit=1)[1]))
    return res


def download_iter_epoch(s3_path: str, epoch: str, out: str):
    """Download epoch model from s3 to out

    Args:
        s3_path (str): "s3://collatio-nnocr/OCRBenchmark/TextBPNPlusPlus-TD500-Reproduce-ResNet50-102323"
        epoch (str): "TextBPN_resnet50_60"
        out (str): where to save the models
    """
    src = f"{s3_path}/{epoch}.pth"
    os.system(f"aws s3 cp {src} {out}")


def delete_models(model_dir: str):
    """Remove models in model_dir"""
    shutil.rmtree(model_dir)


def set_logging():
    logger = logging.getLogger("Easter Arabic Logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger


def get_epoch_num(epoch: str) -> int:
    """
    Get epoch number from epoch

    Args:
        epoch (str): e.g. TextBPN_resnet50_60

    Returns:
        int: 60
    """
    return int(epoch.rsplit("_", maxsplit=1)[1])


def evaluate(epoch: int, backbone: str, model_save_dir: str):
    """Evaluate epoch

    Args:
        epoch (int): e.g. 20
        backbone (str): e.g. resnet50
        model_save_dir (str): dir where model is saved
    """
    cmd = f"""CUDA_LAUNCH_BLOCKING=1 \
              python3 eval_textBPN.py \
              --net {backbone} \
              --scale 1 \
              --exp_name TD500 \
              --num_workers 8 \
              --save_dir {model_save_dir} \
              --checkepoch {epoch} \
              --test_size 640 960 \
              --dis_threshold 0.35 \
              --cls_threshold 0.875 \
              --gpu 0;
           """
    p = subprocess.run(cmd, shell=True, check=True)
    return p


if __name__ == "__main__":
    # set variables
    model = MODELS["TD500-reproduce-102023"]

    model_prefix = model["prefix"]
    exp_name = model["exp_name"]
    model_s3_path = model["s3_path"]
    save_model_dir = model["save_model_dir"]
    threshold = model["thresh"]
    result_csv = model["result_csv"]
    backbone = model["backbone"]

    logger = set_logging()
    logger.info("Locating all model checkpoints...")
    start = time.time()
    # find all possible epochs
    all_epochs = find_all_iter_epochs(model_prefix)
    filtered_epochs = filter_epochs(all_epochs, threshold)

    for epoch in tqdm(filtered_epochs):
        # download the model
        model_save_to_dir = os.path.join(save_model_dir, exp_name)
        logger.info(f"downloading model {epoch} to {model_save_to_dir}...")
        os.makedirs(model_save_to_dir, exist_ok=True)
        download_iter_epoch(model_s3_path, epoch, model_save_to_dir)

        # evaluate the model
        logger.info(f"Evaluating model {epoch}...")
        epoch_num = get_epoch_num(epoch)
        evaluate(epoch_num, backbone, save_model_dir)

        # delete the model
        logger.info(f"Deleting model {epoch}...")
        delete_models(model_save_to_dir)

        # finish and log
        logger.info(f"Finish {epoch}!!!")

    elapse = time.time() - start
    logger.info(f"It takes {elapse}s to finish the job")
    logger.info(f"It takes {elapse/60}min to finish the job")
