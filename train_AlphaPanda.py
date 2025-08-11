import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from AlphaPanda.datasets import get_dataset
from AlphaPanda.models import get_model
from AlphaPanda.utils.misc import *
from AlphaPanda.utils.data import *
from AlphaPanda.utils.train import *

import pdb
import numpy as np
import subprocess as ps

#huyue
#import esm
#from esm import Alphabet
cpu_num=5
torch.set_num_threads(cpu_num)
from AlphaPanda.datasets.sabdab import AA_tensor_to_sequence
from AlphaPanda.utils.utils_trx import *
import residue_constants
from AlphaPanda.utils.protein.constants import BBHeavyAtom, AA
from einops import rearrange




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    args = parser.parse_args()

    if args.device:
        print('Device:', args.device)
        torch.set_default_device(args.device)

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)
    train_generator = torch.Generator(device=args.device).manual_seed(config.train.seed)
    train_iterator = inf_iterator(DataLoader(
        train_dataset, 
        batch_size=config.train.batch_size, 
        collate_fn=PaddingCollate(), 
        shuffle=True,
        num_workers=args.num_workers,
        generator=train_generator
    ))
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model, args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))
  


    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info('Resuming from checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])

    # Train
    def train(it):
        time_start = current_milli_time()
        model.train()
        # Prepare data
        batch = recursive_to(next(train_iterator), args.device)

		
		#esm
		
        #huyue

		
		
		
		
		
		
		
		
		
		
		
		
        # Forward
        # if args.debug: torch.set_anomaly_enabled(True)
        loss_dict = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        time_forward_end = current_milli_time()

        # Backward
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        log_losses(loss_dict, it, 'train', logger, writer, others={
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })

        if not torch.isfinite(loss):
            logger.error('NaN or Inf detected.')
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': it,
                'batch': recursive_to(batch, 'cpu'),
            }, os.path.join(log_dir, 'checkpoint_nan_%d.pt' % it))
            raise KeyboardInterrupt()

    # Validate
    def validate(it):
        start_time = time.perf_counter()
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            gpu_memory = np.round(torch.cuda.memory_allocated() / 2**20).astype(int)
            max_gpu_memory = np.round(torch.cuda.memory_allocated() / 2**20).astype(int)
            print(f"CUDA memory allocated: {gpu_memory} MB")
            print(f"CUDA Max memory allocated: {max_gpu_memory} MB")
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)
                # Forward
				
				
				
				#forward
                #loss_dict = model(batch,EsmToken_res_feat,EsmToken_pair_feat)
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss

                loss_tape.update(loss_dict, 1)

        avg_loss = loss_tape.log(it, logger, writer, 'val')
        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1e3 # convert to ms
        print(f"Val Elapsed Time: {elapsed_ms:.2f} ms")
        
        return avg_loss

    try:
        for it in range(it_first, config.train.max_iters + 1):
            start_time = time.perf_counter()
            gpu_memory = np.round(torch.cuda.memory_allocated() / 2**20).astype(int)
            max_gpu_memory = np.round(torch.cuda.memory_allocated() / 2**20).astype(int)
            print(f"CUDA memory allocated: {gpu_memory} MB")
            print(f"CUDA Max memory allocated: {max_gpu_memory} MB")
            train(it)

            # save checkpoint every 25 or val_freq iterations
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                    }, ckpt_path)
            
            # remove old iterations checkpoints
            if it % 1000 == 0 and it > 5000:
                ckpt_exn = os.path.join(ckpt_dir, r'{%d..%d..%d}.pt' % (it-6000+config.train.val_freq, it-5000, config.train.val_freq))
                cmd = 'rm %s' % ckpt_exn
                bash = ['bash', '-c', cmd]
                try:
                    run = ps.run(bash, capture_output=True, check=True, shell=False, timeout=5)
                    stdout, stderr = run.stdout, run.stderr
                    print(run.stdout)
                    print('Cleaned checkpoints %d thru %d' % (it-6000+config.train.val_freq, it-5000))
                except ps.CalledProcessError as e:
                    print(f'Rm files failed with error: {e}')
                except ps.TimeoutExpired as e:
                    pass


            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1e3  # convert to ms
            
            print(f"Elapsed Time: {elapsed_ms:.2f} ms")
            
    except KeyboardInterrupt:
        logger.info('Terminating...')
