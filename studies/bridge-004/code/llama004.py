"""Native float32 TinyLlama interventions; query-output heads preserve GQA semantics."""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import torch
from common004 import ROOT,SEED
from transport_bridge.engine import Engine,outcomes,result_hooks,state_hook

class OutputWeights:
    def __init__(self,model): self.model=model
    def __getitem__(self,key):
        layer,head=key
        return self.model.model.layers[layer].self_attn.o_proj.weight.detach().T[head*64:(head+1)*64]

class LlamaEngine(Engine):
    def __init__(self,source=None):
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(4);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.device=torch.device('cuda');self.site=16;self.layers=list(range(16,22));self.heads=32
        self.source=[tuple(x) for x in (source or [(0,0)])]
        self.config={'head_chunk':4,'threads':4,'batch_size':4}
        self.spec={'width':2048,'site':16,'heads':32,'layers':self.layers}
        self.model=AutoModelForCausalLM.from_pretrained(str(ROOT/'models/tinyllama-1.1b-base'),dtype=torch.float32,attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval().to(self.device)
        self.weights=OutputWeights(self.model)

    @contextmanager
    def native_hooks(self,edits=(),cache=None):
        handles=[];mapping={}
        for name,fn in edits: mapping.setdefault(name,[]).append(fn)
        def apply(name,value):
            for fn in mapping.get(name,[]): value=fn(value,SimpleNamespace(name=name))
            if cache is not None:cache[name]=value.detach().clone()
            return value
        try:
            for layer,block in enumerate(self.model.model.layers):
                rn=f'blocks.{layer}.hook_resid_pre';zn=f'blocks.{layer}.attn.hook_z';an=f'blocks.{layer}.hook_attn_out'
                if cache is not None or rn in mapping:
                    def residual(module,args,name=rn):return (apply(name,args[0]),)+args[1:]
                    handles.append(block.register_forward_pre_hook(residual))
                if cache is not None or zn in mapping:
                    def zhook(module,args,name=zn):
                        value=args[0];out=apply(name,value.reshape(*value.shape[:-1],32,64))
                        return (out.reshape_as(value),)+args[1:]
                    handles.append(block.self_attn.o_proj.register_forward_pre_hook(zhook))
                if an in mapping:
                    def attnout(module,args,out,name=an):return apply(name,out)
                    handles.append(block.self_attn.o_proj.register_forward_hook(attnout))
            yield
        finally:
            for handle in handles:handle.remove()

    def logits(self,rows,edits=(),cache=None):
        tokens=torch.tensor([r['token_ids'] for r in rows],device=self.device)
        with self.native_hooks(edits,cache):return self.model(tokens,use_cache=False,logits_to_keep=1).logits

    def forward(self,rows,hooks=()):
        cache={};logits=self.logits(rows,hooks,cache)
        return outcomes(logits,rows),cache

    def suffix(self,hidden,start,hooks=()):
        n=hidden.shape[1];pos=torch.arange(n,device=hidden.device)[None]
        mask=torch.full((1,1,n,n),torch.finfo(hidden.dtype).min,device=hidden.device).triu(1)
        rotary=self.model.model.rotary_emb(hidden,pos)
        with self.native_hooks(hooks):
            for block in self.model.model.layers[start:]:
                hidden=block(hidden,attention_mask=mask,position_ids=pos,position_embeddings=rotary,use_cache=False)
        return self.model.lm_head(self.model.model.norm(hidden[:,-1:]))

    def validate(self,rows):
        errors={};direct=self.logits(rows);base,cache=self.forward(rows)
        errors['native_cached_logits']=float((self.logits(rows,cache={})-direct).abs().max())
        for layer in [0,16,21]:errors[f'suffix_{layer}']=float((self.suffix(cache[f'blocks.{layer}.hook_resid_pre'],layer)-direct).abs().max())
        pos=torch.tensor([r['s2'] for r in rows],device=self.device)
        zero=torch.zeros(len(rows),2048,device=self.device)
        errors['zero_edit']=float((self.logits(rows,[state_hook(self.site,pos,zero)])-direct).abs().max())
        dz=torch.randn(len(rows),64,device=self.device)*.03;layer,head=self.source[0]
        dr=(dz@self.weights[layer,head])[:,None]
        projected=self.logits(rows,result_hooks([(layer,head)],dr,pos))
        def add_z(value,hook):
            v=value.clone();v[torch.arange(len(v),device=v.device),pos,head]+=dz;return v
        explicit=self.logits(rows,[(f'blocks.{layer}.attn.hook_z',add_z)])
        errors['projected_vs_individual_head']=float((projected-explicit).abs().max())
        bank=torch.stack([cache[f'blocks.{l}.attn.hook_z'][:,-1].mean(0).cpu() for l in self.layers])
        raw,behavior=self.signatures(rows,bank)
        for bi in range(2):
            for layer,head in [(16,0),(21,31)]:
                replacement=torch.zeros(64,device=self.device) if bi==0 else bank[self.layers.index(layer),head].to(self.device)
                def ablate(value,hook,replacement=replacement,head=head):
                    v=value.clone();v[:,-1,head]=replacement;return v
                singleton=outcomes(self.logits(rows,[(f'blocks.{layer}.attn.hook_z',ablate)]),rows)
                expected=(base[:,:2]-singleton[:,:2]).cpu().numpy()
                errors[f'expanded_b{bi}_l{layer}_h{head}']=float(np.max(np.abs(raw[bi,:,self.layers.index(layer)*32+head]-expected)))
        corrupt=[dict(r,token_ids=r['corrupt_tokens']) for r in rows]
        replay,reb=self.signatures(corrupt,bank,boundary=cache['blocks.16.hook_resid_pre'].cpu())
        errors['whole_boundary_behavior']=float(np.max(np.abs(reb-behavior)))
        errors['whole_boundary_all_signatures']=float(np.max(np.abs(replay-raw)))
        result={'errors':errors,'maximum':max(errors.values()),'tolerance':.002,'native_tolerance':.0005,'passed':max(errors.values())<.002 and errors['native_cached_logits']<.0005,'query_output_heads':32,'kv_heads':4,'dtype':'float32','attention':'eager'}
        return result
