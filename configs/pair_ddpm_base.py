import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('DDPM For Background Correction', add_help=False)
    
    # IO Settings
    parser.add_argument('--train_data_dir', default='./training_data_n', type=str)
    parser.add_argument('--model_save_dir', default='./checkpoints', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--dataset_init_every_n_epoch', default=5, type=int)
    parser.add_argument('--model_save_every_n_epoch', default=10, type=int)
    parser.add_argument('--pretrained_weight', default='./ckpts_pair/ddpm_pair_130K.pt', type=str)   

    # Training Settings
    parser.add_argument('--use_mix_precision', default=True, type=bool)
    parser.add_argument('--uncon_ratio', default=0.5, type=float)
    parser.add_argument('--use_torch_attn', default=False, type=bool)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr', default=4.5e-6, type=float)
    parser.add_argument('--grad_clip', default=1., type=float)  
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accumulate_step', default=8, type=int)
    parser.add_argument('--epoch', default=1000, type=int)        

    # DDPM Settings
    parser.add_argument('--T', default=1000, type=float)
    parser.add_argument('--beta_scdl', default='linear', type=str)
    parser.add_argument('--beta_1', default=1e-4, type=float)
    parser.add_argument('--beta_T', default=2e-2, type=float)
    return parser