# prepare_data.py - download and preprocess Amazon datasets
import os,json,gzip,sys,argparse,random
from collections import defaultdict,Counter
import torch
import urllib.request

DATA_URLS={'beauty':'https://jmcauley.ucsd.edu/data/amazon_v2/reviewFiles/Beauty.json.gz','sports':'https://jmcauley.ucsd.edu/data/amazon_v2/reviewFiles/Sports_and_Outdoors.json.gz'}
META_URLS={'beauty':'https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/Beauty.json.gz'}
def parse_reviews(path,min_user=5,min_item=5):
    user_items=defaultdict(list)
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            uid=r['reviewerID']
            iid=r['asin']
            user_items[uid].append(iid)
    item_cnt=Counter()
    for seq in user_items.values():
        for iid in set(seq): item_cnt[iid]+=1
    keep={iid for iid,cnt in item_cnt.items() if cnt>=min_item}
    res={}
    for uid,seq in user_items.items():
        seq=[i for i in seq if i in keep]
        if len(set(seq))>=min_user: res[uid]=seq
    return res,len(keep)
def download(url,dest):
    os.makedirs(os.path.dirname(dest),exist_ok=True)
    if not os.path.exists(dest):
        print('downloading '+url)
        urllib.request.urlretrieve(url,dest)
    return dest
def split_sequences(seqs,val_pct=0.1,test_pct=0.1):
    users=list(seqs.keys())
    random.shuffle(users)
    nv=int(len(users)*val_pct)
    nt=int(len(users)*test_pct)
    val_u=set(users[:nv])
    test_u=set(users[nv:nv+nt])
    train={u:seqs[u] for u in seqs if u not in val_u and u not in test_u}
    val={u:seqs[u] for u in seqs if u in val_u}
    test={u:seqs[u] for u in seqs if u in test_u}
    return train,val,test
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['beauty','sports'],required=True)
    parser.add_argument('--data-dir',default='./data')
    parser.add_argument('--min-user',type=int,default=5)
    parser.add_argument('--min-item',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    random.seed(args.seed)
    name=args.dataset
    out=os.path.join(args.data_dir,name)
    os.makedirs(out,exist_ok=True)
    url=DATA_URLS[name]
    rpath=download(url,os.path.join(out,name+chr(46)+chr(106)+chr(115)+chr(111)+chr(110)+chr(46)+chr(103)+chr(122)))
    seqs,num_items=parse_reviews(rpath,args.min_user,args.min_item)
    item2idx={}
    idx=1
    for seq in seqs.values():
        for i in seq:
            if i not in item2idx: item2idx[i]=idx; idx+=1
    remapped={u:[item2idx[i] for i in seq] for u,seq in seqs.items()}
    train,val,test=split_sequences(remapped)
    torch.save(train,os.path.join(out,'train.pt'))
    torch.save(val,os.path.join(out,'val.pt'))
    torch.save(test,os.path.join(out,'test.pt'))
    torch.save(num_items,os.path.join(out,'num_items.pt'))
    print(str(len(train))+' train, '+str(len(val))+' val, '+str(len(test))+' test, '+str(num_items)+' items')
if __name__=='__main__':main()