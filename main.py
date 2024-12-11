import argparse
import os
import torch.amp
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_preprocess import NanoCT_Dataset
from ddpm.model import Diffusion_UNet
from ddpm.diffusion import GaussianDiffusionTrainer
from utils import check_distributed, model_eval, model_eval_for_val, val_inference, test_inference



def get_args_parser():
    parser = argparse.ArgumentParser('diffusion for background correction', add_help=False)
    parser.add_argument('--train', default=False, type=bool)
    # Data Settings
    parser.add_argument('--data_dir', default='./training_data_n', type=str)
    parser.add_argument('--model_save_dir', default='./checkpoints', type=str)
    parser.add_argument('--load_weight', default=True, type=bool)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--checkpoint', default='ckpt_step=48437.pt', type=str)   
    # Training Settings
    parser.add_argument('--use_mix_precision', default=True, type=bool)
    parser.add_argument('--uncon_ratio', default=0.5, type=float)
    parser.add_argument('--use_torch_attn', default=False, type=bool)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--lr', default=4.5e-6, type=float)
    parser.add_argument('--grad_clip', default=1., type=float)  
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accumulate_step', default=8, type=int)
    parser.add_argument('--epoch', default=500, type=int)        
    # DDPM Settings
    parser.add_argument('--T', default=1000, type=float)
    parser.add_argument('--beta_sche', default='quad', type=str)
    parser.add_argument('--beta_1', default=1e-4, type=float)
    parser.add_argument('--beta_T', default=2e-2, type=float)
    return parser


def main(args):
    # multi-GPU settings
    rank, local_rank, world_size, is_distributed = check_distributed()
    print(check_distributed())
    if is_distributed:
        torch.cuda.set_device(local_rank)  # set current device
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")  # initialize process group and set the communication backend betweend GPUs
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # data settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.model_save_dir, exist_ok=True)
    model = Diffusion_UNet(use_torch_attn=args.use_torch_attn).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # model settings
    if args.load_weight:
        model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=device), strict=False)
        print("Model weight load down.")

    lr = torch.cuda.device_count() * args.batch_size * args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler()
    trainer = GaussianDiffusionTrainer(model, args.beta_1, args.beta_T, args.T, args.uncon_ratio).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0 
    for e in range(args.epoch):
        # generate new dataset
        if e % 5 == 0:
            dataset = NanoCT_Dataset(data_dir='./training_data_n', img_size=args.img_size)
            if is_distributed:
                train_sampler = DistributedSampler(dataset, shuffle=True)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, drop_last=True, pin_memory=True, sampler=train_sampler)
                dataloader.sampler.set_epoch(e)
            else:
                dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, drop_last=True, pin_memory=True, shuffle=True)
        # training
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for obj_ref, ref in tqdmDataLoader:
                # for mix precision
                if args.use_mix_precision:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        condit, x_0 = obj_ref.to(device, non_blocking=True), ref.to(device, non_blocking=True) 
                        loss = trainer(condit, x_0).mean()
                    scaler.scale(loss).backward()
                    step+=1
                    if step % args.accumulate_step == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                # normal fp32
                else:
                    condit, x_0 = obj_ref.to(device, non_blocking=True), ref.to(device, non_blocking=True) 
                    loss = trainer(condit, x_0).mean()
                    loss.backward()   
                    step+=1
                    if step % args.accumulate_step == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                # log
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        if (e+1)%10==0:
            torch.save(model.state_dict(), '%s/ckpt_step=%d.pt'%(args.model_save_dir, step//args.accumulate_step))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.train:
        main(args)
    else:
        model = Diffusion_UNet(use_torch_attn=args.use_torch_attn).to(args.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_save_dir+'/'+args.checkpoint, map_location=args.device), strict=False)
        model.eval()
        print("Model weight load down.")
        test_inference(args, model=model)