from datasets import load_dataset
import torch
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import os
import shutil
import time


def get_pile(tokenizer, train_size, val_size, seed, seqlen):
    print("get_pile")
    traindata = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    traindata = traindata.shuffle(seed=seed) 

    random.seed(seed)
    val_sample_ratio = 0.9
    trainloader = []
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata)*val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader



def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, valloader


def get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_c4")
    try:
        # set local path for faster loading
        traindata = load_dataset("arrow",
                    data_files={
                        "train": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-6fbe877195f42de5/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/json-train-00000-of-00002.arrow",
                        "validation": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-efc3d4f4606f44bd/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-validation.arrow",
                    },split='train'
                    )
        valdata = load_dataset("arrow",
                    data_files={
                        "validation": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-efc3d4f4606f44bd/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-validation.arrow",
                    },split='validation'
                    )
    except:
        traindata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        valdata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    if test_only:
        return valenc 

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata)*val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))



    return trainloader, valloader 

def get_redpajama(tokenizer, train_size, val_size, seed, seqlen):
    print("get_redpajama")
    try:
        loacal_dataset = "/cpfs01/user/chenmengzhao/huggingface/datasets/togethercomputer___red_pajama-data-1_t-sample"
        traindata = load_dataset(loacal_dataset,split='train')   
    except:
        traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample",split='train')   
    random.seed(seed)
    traindata = traindata.shuffle(seed=seed) 
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata)*val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader



def get_loaders(
    name, tokenizer, train_size=128, val_size=64,seed=0, seqlen=2048, test_only=False
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'c4' in name:
        return get_c4(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'redpajama' in name:
        return get_redpajama(tokenizer,train_size,val_size,seed,seqlen)
    elif 'pile' in name:
        return get_pile(tokenizer,train_size,val_size,seed,seqlen)
    else:
        raise NotImplementedError



@torch.no_grad()
def test_ppl(args, model, tokenizer,prefixed_key_values=None, datasets=['wikitext2']):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=args.ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = args.ppl_seqlen
        nsamples = testenc.numel() // seqlen
        model.eval()
        nlls = []
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)]
            labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)]
            batch = batch.to(model.device)
            labels = labels.to(model.device)
            outputs = model(batch,labels=labels, past_key_values=prefixed_key_values)
            neg_log_likelihood = outputs.loss * seqlen
            nlls.append(neg_log_likelihood)

        
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        results[dataset] = ppl.item()
        print(f'{dataset}:{ppl}')
    return results

class BlockTrainDataset(Dataset):
    def __init__(self, size, seqlen, hidden_size, batch_size, dtype, cache_path='./cache/', off_load_to_disk=False):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        assert size%batch_size == 0
         
        if self.off_load_to_disk:
            self.cache_path = self.get_cache_path(cache_path)
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            self.data = torch.zeros((self.size//self.batch_size, self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)

    def _initialize_data_on_disk(self):
        for idx in range(self.size//self.batch_size):
            tensor = torch.zeros((self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx):
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        return self.size//self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            tensor = self.data[idx]
        return tensor

    def update_data(self, idx, new_data):
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            torch.save(new_data.to(self.dtype), filepath)
        else:
            self.data[idx] = new_data

    def get_subset(self,num):
        data = []
        iter = num//self.batch_size
        for idx in range(iter):
            if self.off_load_to_disk:
                filepath = self._get_file_path(idx)
                tensor = torch.load(filepath)
            else:
                tensor = self.data[idx]
            data.append(tensor)
        return torch.cat(data,dim=0)
    
    def delete_cache(self):
        if os.path.exists(self.cache_path):
            shutil.rmtree(self.cache_path)
            
    def get_cache_path(self, parent_dir):
        new_cache_path = None
        while new_cache_path is None or os.path.exists(new_cache_path):
            flag = time.time()
            new_cache_path = os.path.join(parent_dir, flag)
            time.sleep(1) # avoid same flag        

def replace_last_directory_level(original_path, new_directory):
    parts = original_path.split(os.sep)
    parts[-1] = new_directory
    new_path = os.path.join(*parts)
    return new_path

def copy_block_dataset(dataset:BlockTrainDataset, cache_dir=None):
    new_cache_path = None
    if dataset.off_load_to_disk:
        old_cache_path = dataset.cache_path
        while new_cache_path is None or os.path.exists(new_cache_path):
            flag = time.time()
            new_cache_path = replace_last_directory_level(old_cache_path, flag)
            time.sleep(1) # avoid same flag
        shutil.copytree(old_cache_path, new_cache_path)
    new_dataset = BlockTrainDataset(dataset.size, dataset.seqlen,
                                    dataset.hidden_size, dataset.batch_size,
                                    dataset.dtype, new_cache_path,
                                    dataset.off_load_to_disk)
    if not dataset.off_load_to_disk:
        for index, data in enumerate(dataset):
            new_dataset.update_data(index, data)
    return new_dataset