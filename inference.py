# coding: utf-8
import os
import time
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from src.util import ExeDataset,write_pred
from src.model import MalConv
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    # Load config file for experiment
    parser.add_argument('--config_path', default='config/example.yaml')
    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()
    print(args)
    return args

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def inference(args, malconv, validloader):
    malconv.eval()

    if args.channels_last:
        try:
            malconv = malconv.to(memory_format=torch.channels_last)
            print("---- Use NHWC format")
        except RuntimeError as e:
            print("---- Use normal format")
            print("failed to enable NHWC: ", e)
    if args.nv_fuser:
       fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    total_time = 0.0
    total_sample = 0
    profile_len = min(len(validloader), args.num_iter + args.num_warmup) // 2

    if args.profile and args.device == "xpu":    
        for i, val_batch_data in enumerate(validloader):
            if i >= args.num_iter:
                break

            cur_batch_size = val_batch_data[0].size(0)
        
            exe_input = val_batch_data[0].to(args.device)
            exe_input = Variable(exe_input.long(),requires_grad=False)

            if args.channels_last:
                exe_input = exe_input.to(memory_format=torch.channels_last) if len(exe_input.shape) == 4 else exe_input
            if args.jit and i == 0:
                try:
                    malconv = torch.jit.trace(malconv, exe_input, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)
   
            elapsed = time.time()
            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                pred = malconv(exe_input)
            torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
            if args.profile and i == profile_len:
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i, val_batch_data in enumerate(validloader):
                if i >= args.num_iter:
                    break

                cur_batch_size = val_batch_data[0].size(0)
            
                exe_input = val_batch_data[0].to(args.device)
                exe_input = Variable(exe_input.long(),requires_grad=False)

                if args.channels_last:
                    exe_input = exe_input.to(memory_format=torch.channels_last) if len(exe_input.shape) == 4 else exe_input
                if args.jit and i == 0:
                    try:
                        malconv = torch.jit.trace(malconv, exe_input, check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
   
                elapsed = time.time()
                with torch.jit.fuser(fuser_mode):
                    pred = malconv(exe_input)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i, val_batch_data in enumerate(validloader):
                if i >= args.num_iter:
                    break

                cur_batch_size = val_batch_data[0].size(0)
            
                exe_input = val_batch_data[0].to(args.device)
                exe_input = Variable(exe_input.long(),requires_grad=False)

                if args.channels_last:
                    exe_input = exe_input.to(memory_format=torch.channels_last) if len(exe_input.shape) == 4 else exe_input
                if args.jit and i == 0:
                    try:
                        malconv = torch.jit.trace(malconv, exe_input, check_trace=False, strict=False)
                        print("---- JIT trace enable.")
                    except (RuntimeError, TypeError) as e:
                        print("---- JIT trace disable.")
                        print("failed to use PyTorch jit mode due to: ", e)
   
                elapsed = time.time()
                pred = malconv(exe_input)
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif not args.profile and args.device == "cuda":
        for i, val_batch_data in enumerate(validloader):
            if i >= args.num_iter:
                break

            cur_batch_size = val_batch_data[0].size(0)
        
            exe_input = val_batch_data[0].to(args.device)
            exe_input = Variable(exe_input.long(),requires_grad=False)

            if args.channels_last:
                exe_input = exe_input.to(memory_format=torch.channels_last) if len(exe_input.shape) == 4 else exe_input
            if args.jit and i == 0:
                try:
                    malconv = torch.jit.trace(malconv, exe_input, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)
   
            elapsed = time.time()
            with torch.jit.fuser(fuser_mode):
                pred = malconv(exe_input)
            torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
    else:
        for i, val_batch_data in enumerate(validloader):
            if i >= args.num_iter:
                break

            cur_batch_size = val_batch_data[0].size(0)
        
            exe_input = val_batch_data[0].to(args.device)
            exe_input = Variable(exe_input.long(),requires_grad=False)

            if args.channels_last:
                exe_input = exe_input.to(memory_format=torch.channels_last) if len(exe_input.shape) == 4 else exe_input
            if args.jit and i == 0:
                try:
                    malconv = torch.jit.trace(malconv, exe_input, check_trace=False, strict=False)
                    print("---- JIT trace enable.")
                except (RuntimeError, TypeError) as e:
                    print("---- JIT trace disable.")
                    print("failed to use PyTorch jit mode due to: ", e)
   
            elapsed = time.time()
            pred = malconv(exe_input)
            if args.device == "xpu":
                torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

def main():
    args = parse_args()
    seed = args.seed

    conf = yaml.load(open(args.config_path,'r'))

    exp_name = conf['exp_name']+'_sd_'+str(seed)
    print('Experiment:')
    print('\t',exp_name)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_data_path = conf['train_data_path']
    train_label_path = conf['train_label_path']
    
    valid_data_path = conf['valid_data_path']
    valid_label_path = conf['valid_label_path']
    
    log_dir = conf['log_dir']
    pred_dir = conf['pred_dir']
    checkpoint_dir = conf['checkpoint_dir']
    
    
    log_file_path = log_dir+exp_name+'.log'
    chkpt_acc_path = checkpoint_dir+exp_name+'.model'
    pred_path = pred_dir+exp_name+'.pred'
    
    # Parameters
    use_gpu = conf['use_gpu']
    use_cpu = conf['use_cpu']
    learning_rate = conf['learning_rate']
    max_step = conf['max_step']
    test_step = conf['test_step']
    batch_size = conf['batch_size']
    first_n_byte = conf['first_n_byte']
    window_size = conf['window_size']
    display_step = conf['display_step']
    sample_cnt = conf['sample_cnt']

    # OOB params
    use_gpu = True if args.device == "cuda" else False
    batch_size = args.batch_size
    
    # Load Ground Truth.
    tr_label_table = pd.read_csv(train_label_path,header=None,index_col=0)
    tr_label_table.index=tr_label_table.index.str.upper()
    tr_label_table = tr_label_table.rename(columns={1:'ground_truth'})
    val_label_table = pd.read_csv(valid_label_path,header=None,index_col=0)
    val_label_table.index=val_label_table.index.str.upper()
    val_label_table = val_label_table.rename(columns={1:'ground_truth'})
    
    # Merge Tables and remove duplicate
    tr_table = tr_label_table.groupby(level=0).last()
    del tr_label_table
    val_table = val_label_table.groupby(level=0).last()
    del val_label_table
    tr_table = tr_table.drop(val_table.index.join(tr_table.index, how='inner'))
    
    print('Training Set:')
    print('\tTotal',len(tr_table),'files')
    print('\tMalware Count :',tr_table['ground_truth'].value_counts()[1])
    print('\tGoodware Count:',tr_table['ground_truth'].value_counts()[0])
    
    
    print('Validation Set:')
    print('\tTotal',len(val_table),'files')
    print('\tMalware Count :',val_table['ground_truth'].value_counts()[1])
    print('\tGoodware Count:',val_table['ground_truth'].value_counts()[0])
    
    if sample_cnt != 1:
        tr_table = tr_table.sample(n=sample_cnt,random_state=seed)
    
    dataloader = DataLoader(ExeDataset(list(tr_table.index), train_data_path, list(tr_table.ground_truth),first_n_byte),
                                batch_size=batch_size, shuffle=True, num_workers=use_cpu)
    validloader = DataLoader(ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth),first_n_byte),
                            batch_size=batch_size, shuffle=False, num_workers=use_cpu)
    
    valid_idx = list(val_table.index)
    del tr_table
    del val_table
    
    
    malconv = MalConv(input_length=first_n_byte,window_size=window_size)
    
    malconv = malconv.to(args.device)

    with torch.inference_mode():
        if args.precision == "float16" and args.device == "cuda":
            print("---- Use autocast fp16 cuda")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                inference(args, malconv, validloader)
        elif args.precision == "float16" and args.device == "xpu":
            print("---- Use autocast fp16 xpu")
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                inference(args, malconv, validloader)
        elif args.precision == "bfloat16" and args.device == "cpu":
            print("---- Use autocast bf16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                inference(args, malconv, validloader)
        elif args.precision == "bfloat16" and args.device == "xpu":
            print("---- Use autocast bf16 xpu")
            with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                inference(args, malconv, validloader)
        else:
            print("---- no autocast")
            inference(args, malconv, validloader)


if __name__ == "__main__":
    main()
