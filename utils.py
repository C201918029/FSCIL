
import io
import os
import time
import math
from collections import defaultdict, deque
import datetime
from typing import Iterable
import torch.nn.functional as F
import torch
import torch.distributed as dist

def adjust_position_embeddings(pos_embed, new_length):
     pos_embed = pos_embed.unsqueeze(1)
     new_pos_embed = F.interpolate(pos_embed, size=(new_length, pos_embed.size(-1)), mode='bilinear', align_corners=False)
     new_pos_embed = new_pos_embed.squeeze(1)
     return new_pos_embed
def attention_weights(logits,pos_embed):
    attention_logits = torch.mean(logits, dim=1)  
    attention_weights = F.softmax(attention_logits, dim=0)  
    
    pos_embed = pos_embed.squeeze(0)
    attention_scores = torch.matmul(pos_embed, pos_embed.transpose(0, 1))
    position_attention_weights = F.softmax(attention_scores, dim=-1)

    combined_attention_weights = (attention_weights + position_attention_weights) / 2
    
    return combined_attention_weights
def compute_attention_weights_with_position(logits,pos_embed):
    adjusted_pos_embed = pos_embed.repeat(logits.size(0), 1, 1)
    attention_weights = torch.mean(adjusted_pos_embed, dim=2)
    pos_attention_weights = torch.mean(attention_weights, dim=1)

    attention_logits = torch.mean(logits, dim=1)  
    global_attention_weights = F.softmax(attention_logits, dim=0)  
    
    combined_attention_weights = (global_attention_weights + pos_attention_weights) / 2
    
    return combined_attention_weights

#修改
class ClassPrototypes(object):
    def __init__(self, num_class, dim, device):
        self.prototype = torch.zeros((num_class, dim), device=device)
        self.counts = torch.zeros(num_class, device=device)

    def update(self, labels, outputs):
        for label, output in zip(labels, outputs):
            proto_current = self.prototype[label]
            count_current = self.counts[label]

            self.prototype[label] = (proto_current * count_current + output) / (count_current + 1)
            self.counts[label] += 1

    def __getitem__(self, index):
        return self.prototype[index]
    
    def atten_pro(self, labels, outputs,attention_weights):
        for i in range(len(labels)):
            label = labels[i]
            output = outputs[i]
            attention_weight = attention_weights[i] 
            self.prototype[label] = (self.prototype[label] * self.counts[label] + output * attention_weight) / (self.counts[label] + attention_weight)
            self.counts[label] += attention_weight

def contrastive_loss(features, labels, margin=1.0):
    pairwise_distances = torch.cdist(features, features)
    labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)
    pos_pair_loss = pairwise_distances[labels_equal].mean()
    neg_pair_loss = F.relu(margin - pairwise_distances[~labels_equal]).mean()
    return pos_pair_loss + neg_pair_loss

def center_regularization(features, labels, prototypes, lambda_c=1.0):
    protos_for_labels = prototypes[labels]
    distances = torch.norm(features - protos_for_labels, dim=1)
    return lambda_c * distances.mean()

def prototype_update_with_losses(prototype: ClassPrototypes, model: torch.nn.Module, device: torch.device,original_model:torch.nn.Module,data_loader:Iterable, task_id=-1):
    model.eval()
    original_model.eval()
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.no_grad():
            if original_model is not None:
                output,_,_,_ = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

            output,_,_,pos_embed= model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']
            attention_weights = compute_attention_weights_with_position(logits,pos_embed)  
        prototype.atten_pro(target,logits,attention_weights)

    return prototype

def prototype_evaluate_with_losses(prototype: ClassPrototypes, model: torch.nn.Module, device: torch.device,original_model:torch.nn.Module,data_loader:Iterable, task_id=-1,test_id=-1):
    model.eval()
    total, correct = 0, 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            if original_model is not None:
                output,_,_,_ = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output,_,_,_  = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits'] 

            distances = torch.cdist(logits, prototype.prototype)  
            predictions = torch.argmin(distances, dim=1)

            total += target.size(0)
            correct += (predictions == target).sum().item()
    acc = 100.0 * correct / total
    print('Task',task_id+1,'Test',test_id,' ACC:',acc)
    return acc


def prototype_updata(prototype:ClassPrototypes, model:torch.nn.Module, device:torch.device,
                    original_model:torch.nn.Module, data_loader:Iterable, task_id=-1):
    model.eval()
    original_model.eval()
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']
        prototype.pro(target,logits)

    return prototype


def prototype_evaluate(prototype:ClassPrototypes, model: torch.nn.Module, original_model: torch.nn.Module, 
                       data_loader:Iterable, device=None, task_id=-1, test_id=-1):
    model.eval()
    original_model.eval()
    total = 0
    correct = 0
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits'] 

            distances = torch.cdist(logits, prototype.prototype)

            predictions = torch.argmin(distances, dim=1)
            total += target.size(0)
            correct += (predictions == target).sum().item()
    acc = correct*100/total
    print('Task',task_id+1,'Test',test_id,' ACC:',acc)
    return acc



@torch.no_grad()
def prototype_evaluate_till_now(prototype:ClassPrototypes,model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    for i in range(task_id+1):
        acc = prototype_evaluate_with_losses(prototype=prototype,model=model,original_model=original_model,
                                data_loader=data_loader[i]['val'],task_id=task_id,device=device,test_id = i)
        acc_matrix[i, task_id] = acc
    
    return acc_matrix

class ClassPrototypesModule(torch.nn.Module):
    def __init__(self, num_class, num_prototypes, dim, device):
        super(ClassPrototypesModule, self).__init__()
        self.prototype = torch.nn.Parameter(torch.randn(num_class, num_prototypes, dim))
        self.attention = torch.nn.Parameter(torch.randn(num_class, num_prototypes))
        self.to(device)


    def forward(self, logits):
        attention_weights = F.softmax(self.attention, dim=1)
        weighted_prototypes = torch.einsum('ijk,ij->ik', self.prototype, attention_weights)

        # Normalize logits and weighted_prototypes
        logits_normalized = F.normalize(logits, p=2, dim=1)
        weighted_prototypes_normalized = F.normalize(weighted_prototypes, p=2, dim=1)

        # Compute cosine similarity
        cosine_similarity_matrix = torch.matmul(logits_normalized, weighted_prototypes_normalized.T)

        # Compute cosine distances
        cosine_distances = 1 - cosine_similarity_matrix

        return cosine_distances


def prototype_update_module(prototype: ClassPrototypesModule, model: torch.nn.Module, device: torch.device,
                     original_model: torch.nn.Module, data_loader: Iterable, task_id=-1):
    optimizer = torch.optim.Adam(prototype.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    original_model.eval()
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
                # cls_features = output
            else:
                cls_features = None

            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

        optimizer.zero_grad()
        min_distance = prototype(logits)
        loss = criterion(min_distance, target)
        loss.backward()
        optimizer.step()

    return prototype

def prototype_evaluate_module(prototype: ClassPrototypesModule, model: torch.nn.Module, original_model: torch.nn.Module,
                       data_loader: Iterable, device=None, task_id=-1, test_id=-1):
    model.eval()
    original_model.eval()
    total = 0
    correct = 0
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            cosine_distances = prototype(logits)
            total += target.size(0)

            predicted_indices = torch.argmin(cosine_distances, dim=1)
            correct_predictions = (predicted_indices == target).sum().item()

    acc = correct_predictions / total * 100
    print('Task', task_id + 1, 'Test', test_id, ' ACC:', acc)
    return acc

@torch.no_grad()
def prototype_evaluate_till_now_module(prototype:ClassPrototypes,model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    for i in range(task_id+1):
        acc = prototype_evaluate_module(prototype=prototype,model=model,original_model=original_model,
                                data_loader=data_loader[i]['val'],task_id=task_id,device=device,test_id = i)
        acc_matrix[i, task_id] = acc

    return acc_matrix

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}'
            # 'time: {time}',
            # 'data: {data}'
        ]
        # if torch.cuda.is_available():
        #     log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0) 
