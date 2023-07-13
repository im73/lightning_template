import os
import argparse
# import wandb
import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from models.LAtrans.LAtrans import LAtrans
from data.vib import Vibration, Vibration_mvg, data_processe
from util.utils import load_yml, visualize_LAtrans, out_segs



parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, help="path to config")
parser.add_argument('--mode', type=str, default='train', help="run mode")
parser.add_argument('--save_path', type=str, default='LAtrans', help="path to save model")
parser.add_argument('--data_path', type=str, default='', help="path to the predict data")

args = parser.parse_args()

config = load_yml(args.config_path)
dataset_dict = {"mel": Vibration, "time": Vibration_mvg}
Dataset_custom = dataset_dict[config.feature_config.type]
model = LAtrans(config.model_config)
trian_dataset =Dataset_custom(config.data_config, config.feature_config, 'train')
val_dataset = Dataset_custom(config.data_config, config.feature_config, 'eval')
test_dataset = Dataset_custom(config.data_config, config.feature_config, 'test')
train_dataloder = DataLoader(dataset=trian_dataset, batch_size=32, shuffle=True, num_workers=24)
val_dataloder = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=24)
test_dataloder = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=24)

# wandb.init(project=configs.project_name, config=configs)
checkpoint_callback = ModelCheckpoint(dirpath=args.save_path, monitor="val_loss", save_top_k=-1)
trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=20, devices=1)
if args.mode == 'train':
    os.system("rm -rf {}".format(args.save_path))
    trainer.fit(model=model, train_dataloaders=train_dataloder, val_dataloaders=val_dataloder)
elif args.mode == 'test':
    root_dir = args.save_path
    obj = os.listdir(root_dir)
    for path in obj:
        if path.endswith('44.ckpt'):
            break
    obj = torch.load(os.path.join(root_dir, path))
    model.load_state_dict(obj["state_dict"])
    out = trainer.predict(model=model, dataloaders=test_dataloder)
    y_preds = []
    y = []
    for result in out:
        y_preds.append(result[0].detach().cpu().numpy())
        y.append(result[1].detach().cpu().numpy())
    anoscores = np.concatenate(y_preds)
    y = np.concatenate(y)
    anoscores = anoscores.reshape(-1)
    y = y.reshape(-1)
    print(model.cal_metric(anoscores, y))
elif args.mode == 'pred' or args.mode == 'none':
    fig_dir = os.path.join(args.save_path, 'fig')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pred_dataset = Dataset_custom(config.data_config, config.feature_config, args.mode)
    pred_dataLoader = DataLoader(dataset=pred_dataset, batch_size=4, shuffle=False, num_workers=24)

    root_dir = args.save_path
    obj = os.listdir(root_dir)
    for path in obj:
        if path.endswith('44.ckpt'):
            break
    obj = torch.load(os.path.join(root_dir, path))
    model.load_state_dict(obj["state_dict"])
    out = trainer.predict(model=model, dataloaders=pred_dataLoader)
# save for use in production environment
    y_preds = []
    y = []
    for result in out:
        y_preds.append(result[0].detach().cpu().numpy())
        y.append(result[1].detach().cpu().numpy())
    anoscores = np.concatenate(y_preds)
    visualize_LAtrans(anoscores, config.pred_config.threshold, pred_dataset, fig_dir)
elif args.mode == 'predict':
    dl = data_processe(config.feature_config)
    data = dl.load_data(args.data_path)
    root_dir = args.save_path
    obj = os.listdir(root_dir)
    for path in obj:
        if path.endswith('44.ckpt'):
            break
    obj = torch.load(os.path.join(root_dir, path))
    model.load_state_dict(obj["state_dict"])
    out = model.predict_step(data, 0)
    out = out.detach().cpu().numpy()
    pred_label = (out > config.pred_config.threshold).astype(int)
    print(out_segs(pred_label))


    


