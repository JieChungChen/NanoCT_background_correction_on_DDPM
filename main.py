import argparse, os, yaml
import torch.amp
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_preprocess import NanoCT_Pair_Dataset
from ddpm.model import Diffusion_UNet
from ddpm.diffusion import DDPM_Trainer
from utils import check_distributed


def get_args_parser():
    parser = argparse.ArgumentParser('diffusion for background correction', add_help=False)
    parser.add_argument('--configs', default='configs/ddpm_pair_v4.yml', type=str)
    return parser


def main(args):
    with open(args.configs, 'r') as f:
        configs = yaml.safe_load(f)
    data_configs = configs['data_settings']
    model_configs = configs['model_settings']
    trn_configs = configs['training_settings']

    # multi-GPU or single-GPU
    rank, local_rank, world_size, is_distributed = check_distributed()
    print(check_distributed())

    if is_distributed:
        torch.cuda.set_device(local_rank)  # set current device
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")  # initialize process group and set the communication backend betweend GPUs
        master_process = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.makedirs(data_configs['model_save_dir'], exist_ok=True)
    model = Diffusion_UNet(model_configs).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if data_configs['pretrained_weight'] is not None:
        model.load_state_dict(torch.load(data_configs['pretrained_weight'], map_location=device), strict=False)
        print("Model weight load down.")

    lr = torch.cuda.device_count() * trn_configs['batch_size'] * trn_configs['lr']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=trn_configs['weight_decay'])
    scaler = torch.amp.GradScaler()
    trainer = DDPM_Trainer(model, configs['ddpm_settings']).to(device)    
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0 
    accumulate_step = trn_configs['accumulate_step']
    for e in range(trn_configs['epoch']):
        # generate new dataset
        if e % data_configs['dataset_init_every_n_epoch'] == 0:
            dataset = NanoCT_Pair_Dataset(data_dir=data_configs['train_data_dir'], img_size=data_configs['img_size'])
            if is_distributed:
                train_sampler = DistributedSampler(dataset, shuffle=True)
                dataloader = DataLoader(dataset, batch_size=trn_configs['batch_size'], num_workers=16, drop_last=True, pin_memory=True, sampler=train_sampler)
                dataloader.sampler.set_epoch(e)
            else:
                dataloader = DataLoader(dataset, batch_size=trn_configs['batch_size'], num_workers=16, drop_last=True, pin_memory=True, shuffle=True)
        # training
        with tqdm(dataloader, dynamic_ncols=False, miniters=10, maxinterval=200, disable=not master_process) as tqdmDataLoader:
            for obj_ref, ref in tqdmDataLoader:

                if trn_configs['use_mix_precision']:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        condit, x_0 = obj_ref.to(device, non_blocking=True), ref.to(device, non_blocking=True) 
                        loss = trainer(condit, x_0).mean()
                    scaler.scale(loss).backward()
                    step+=1
                    if step % accumulate_step == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), trn_configs['grad_clip'])
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                
                else: # normal fp32
                    condit, x_0 = obj_ref.to(device, non_blocking=True), ref.to(device, non_blocking=True) 
                    loss = trainer(condit, x_0).mean()
                    loss.backward()   
                    step+=1
                    if step % accumulate_step == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), trn_configs['grad_clip'])
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                # log
                if (step-1)%10==0:
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": e,
                        "loss: ": loss.item(),
                        "img shape: ": x_0.shape,
                    })

                if (step/accumulate_step) % 5000 == 0:
                    torch.save(model.state_dict(), '%s/ckpt_step=%d.pt'%(data_configs['model_save_dir'], 0+step//accumulate_step))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)