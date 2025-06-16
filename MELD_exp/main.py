import torch
import yaml
import random
import sys
sys.path.append('../')
from dataloader import train_valid_test_meld_dataloader
from torch import nn, optim
import os
from pathlib import Path
import utils
import datetime
from model import Model
import logging
import numpy as np
logger = logging.getLogger('MELD_exp')
logger.setLevel(logging.DEBUG)
from torch.nn import DataParallel
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def set_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open( "config/config.yml", 'r', encoding = "UTF-8" ) as f:
        cfg = yaml.load( f, Loader = yaml.FullLoader )

    set_seed(cfg["exp"]["seed"])
    audio_dir = cfg["exp"]["audio_dir"]
    text_dir = cfg["exp"]["text_dir"]
    meld_csv = {
        "train_csv" : cfg["exp"]["train_csv"],
        "dev_csv" : cfg["exp"]["dev_csv"],
        "test_csv" : cfg["exp"]["test_csv"]
    }



    current_data = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    save_dir = os.path.join(str(Path.cwd()),f"output/{current_data}/{current_time}")

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir,"main.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    model_file_path = '/mnt/cxh10/database/hanlm/emotion2vec/MELD_exp/model.py'
    with open(model_file_path, 'r') as file:
        file_contents = file.read()
        logger.info(file_contents)
    logger.info(cfg)
    logger.info(f"------Now it's begining------")
    batch_size = cfg["train"]["batch_size"]
    train_loader, dev_loader, test_loader = train_valid_test_meld_dataloader(audio_dir,text_dir,batch_size,meld_csv)

    model = Model(cfg)
    model.to( device )
    # 多卡使用
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # 使用所有可见GPU（0和1）
    model.to(device)
    logger.info(model)
    optimizer = optim.RMSprop(model.parameters(), lr=cfg["train"]["optimizer"]["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epoch"])

    test_wa,test_ua = utils.train( logger,save_dir,model, train_loader,dev_loader,test_loader,
                        optimizer, scheduler, device,cfg)


if __name__ == '__main__':
    main()