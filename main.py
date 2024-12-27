import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

from data_preprocess import NanoCT_Pair_Dataset
from ddpm.model import Diffusion_UNet
from ddpm.diffusion import DDPM_Trainer
from utils import check_distributed
from configs.pair_ddpm_base import get_args_parser


def main(args):
    # multi-GPU settings
    rank, local_rank, world_size, is_distributed = check_distributed()
    print(check_distributed())

    if is_distributed:
        torch.cuda.set_device(local_rank)  
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")  
        master_process = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        master_process = True
    
    # model settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.model_save_dir, exist_ok=True)
    model = Diffusion_UNet(use_torch_attn=args.use_torch_attn, size=args.img_size, input_ch=3).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if args.pretrained_weight is not None:
        model.load_state_dict(torch.load(args.pretrained_weight, map_location=device), strict=False)
        print("Model weight load down.")

    lr = torch.cuda.device_count() * args.batch_size * args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler()
    trainer = DDPM_Trainer(model, args.beta_1, args.beta_T, args.beta_scdl, args.T, args.uncon_ratio).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = 0 
    for e in range(args.epoch):

        if e % args.dataset_init_every_n_epoch == 0:
            dataset = NanoCT_Pair_Dataset(data_dir=args.train_data_dir, img_size=args.img_size)
            if is_distributed:
                train_sampler = DistributedSampler(dataset, shuffle=True)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, drop_last=True, pin_memory=True, sampler=train_sampler)
                dataloader.sampler.set_epoch(e)
            else:
                dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, drop_last=True, pin_memory=True, shuffle=True)

        # training
        with tqdm(dataloader, dynamic_ncols=False, miniters=10, maxinterval=200, disable=not master_process) as tqdmDataLoader:
            for obj_ref, ref in tqdmDataLoader:

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
                
                else: # normal fp32
                    condit, x_0 = obj_ref.to(device, non_blocking=True), ref.to(device, non_blocking=True) 
                    loss = trainer(condit, x_0).mean()
                    loss.backward()   
                    step+=1
                    if step % args.accumulate_step == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                # log
                if (step-1)%10==0:
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": e,
                        "loss: ": loss.item(),
                        "img shape: ": x_0.shape,
                    })

        if (e+1) % args.model_save_every_n_epoch == 0:
            torch.save(model.state_dict(), '%s/ckpt_step=%d.pt'%(args.model_save_dir, step//args.accumulate_step))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)