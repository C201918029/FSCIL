
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import optuna
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from argparse import Namespace
import utils


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None, prototype=None):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Task:{task_id+1} Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output,_,_,_ = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        output,d_prompt,f_prompt,pos_embed= model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']
#修改
        total_cosine_loss = 0
        for i in range(3):
            f_prompt_slice = f_prompt[i]
            # 计算d_prompt和f_prompt的余弦相似度，假定最后的维度用于通道（特征）比较
            cosine_sim = torch.nn.functional.cosine_similarity(d_prompt, f_prompt_slice, dim=-1)
            # 将相似度转成损失
            cosine_loss = (1 - cosine_sim).mean()  # 平均化作为损失
            total_cosine_loss += cosine_loss

        contrast_loss = utils.contrastive_loss(logits, target)
        center_reg_loss = utils.center_regularization(logits, target, prototype)

        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask) 
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        ce_loss = criterion(logits, target)
        alpha = 0.01  # 你可以根据实验调整此权重
        beat = 0.01
        gammar = 0.05
        total_loss = ce_loss + alpha * contrast_loss + beat * center_reg_loss + gammar * total_cosine_loss
        # total_loss = ce_loss + alpha * contrast_loss + beat * center_reg_loss
        if args.pull_constraint and 'reduce_sim' in output:
            total_loss = total_loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=total_loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))  
    prototype = utils.ClassPrototypes(num_class=args.nb_classes,dim=args.nb_classes,device=device) 
    # prototype = utils.ClassPrototypesModule(num_class=args.nb_classes,num_prototypes=3,dim=args.nb_classes,device=device) # 原型可变

    for task_id in range(args.num_tasks): 
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool and args.use_f_prompt:
            if task_id > 0: 
                prev_start = (task_id - 1) * args.top_k 
                prev_end = task_id * args.top_k 

                cur_start = prev_end 
                cur_end = (task_id + 1) * args.top_k 

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_f_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_f_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad(): 
                        if args.distributed:
                            model.module.f_prompt.prompt.grad.zero_()
                            model.module.f_prompt.prompt[cur_idx] = model.module.f_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.f_prompt.prompt.grad.zero_()
                            model.f_prompt.prompt[cur_idx] = model.f_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key and args.use_f_prompt:
            if task_id > 0: 
                with torch.no_grad():
                    if args.distributed:
                        model.module.f_prompt.prompt_key.grad.zero_()
                        model.module.f_prompt.prompt_key[cur_idx] = model.module.f_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.f_prompt.prompt_key.grad.zero_()
                        model.f_prompt.prompt_key[cur_idx] = model.f_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer: 
            optimizer = create_optimizer(args, model) 

        if task_id > 0:
            try: 
                args.epochs = args.inc_epochs
            except:
                pass

        for epoch in range(args.epochs):            
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args, prototype=prototype)

            if lr_scheduler:
                lr_scheduler.step(epoch)               


        prototype = utils.prototype_update_with_losses(prototype=prototype,model=model,original_model=original_model,
                                data_loader=data_loader[task_id]['train'],task_id=task_id,device=device)
        acc_matrix = utils.prototype_evaluate_till_now(prototype=prototype, model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)


    column_mean_values = []
    for i in range(args.num_tasks):
        selected_elements = acc_matrix[:i+1, i]
        mean_value = np.mean(selected_elements)
        column_mean_values.append(mean_value)

    print(acc_matrix)
    print(column_mean_values)
    
    return acc_matrix