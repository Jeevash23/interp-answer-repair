import numpy as np
from transformers import AutoTokenizer
from common004 import HERE,ROOT,PREV,MODELS,SEED,read,write,freeze
from build003 import NAMES,FAMILIES,tokenize

EXTRA='Aaron Abel Abraham Adam Adrian Albert Alberto Alexander Alfred Alicia Allen Alvin Amber Andrew Anthony Antonio April Audrey Austin Barry Benjamin Bernard Beth Billy Blake Bobby Brad Bradley Brandon Bruce Bryan Calvin Carl Carlos Casey Cecil Chad Chester Chris Christian Christopher Clarence Clifford Clyde Cody Colin Connie Curtis Daniel David Dean Dolores Doris Dorothy Duane Duncan Edgar Edith Edward Edwin Eleanor Elizabeth Emily Emma Evelyn Felix Frank Fred Frederick George Grace Guy Henry Holly Hugh Ian Irene Isaac Isabel Jack James Jason Jennifer Jessica Jim Jordan Joseph Josh Julia Julian Justin Katherine Kathy Kevin Laura Lauren Leonard Leonard Louise Malcolm Manuel Marcus Mark Martin Marvin Mason Michael Mike Mildred Milton Miriam Nancy Neil Nelson Oliver Pamela Patrick Paul Paula Peggy Perry Ralph Rebecca Renee Richard Rick Robert Robin Roger Rose Ruby Ruth Ryan Sean Scott Seth Simon Sophia Stephen Susan Sylvia Taylor Ted Thomas Tim Timothy Victor Victoria Wade William Zachary'.split()
BLOCKS={
 'calibration':[('museum','ticket')],
 'screen':[('library','letter'),('market','basket')],
 'discovery':[('station','parcel'),('school','pencil'),('theater','program'),('hotel','key')],
 'test':[('airport','passport'),('beach','towel'),('clinic','bottle'),('gallery','picture'),('office','folder'),('park','ball'),('restaurant','napkin'),('shop','bag')],
}

def main():
    assert not (HERE/'data/design.json').exists()
    old=read(PREV/'data/design.json')
    excluded=set(sum(old['names'].values(),[]))|set(old['excluded_prior_names'])
    toks={m:AutoTokenizer.from_pretrained(str(ROOT / 'models'/m if m=='gpt2-medium' else ROOT/'models'/m),local_files_only=True) for m in MODELS}
    pool=sorted({n for n in NAMES+EXTRA if n not in excluded and all(len(t.encode(' '+n,add_special_tokens=False))==1 for t in toks.values())})
    assert len(pool)>=48,len(pool)
    np.random.default_rng(SEED).shuffle(pool)
    banks={'calibration':pool[:6],'screen':pool[6:18],'discovery':pool[18:30],'test':pool[30:48]}
    old_blocks={tuple(x) for values in old['blocks'].values() for x in values}
    assert not old_blocks.intersection({x for v in BLOCKS.values() for x in v})
    write(HERE/'data/design.json',dict(seed=SEED,names=banks,blocks=BLOCKS,families=FAMILIES,excluded_names=sorted(excluded),tokenization_pool=len(pool)))
    for model,tok in toks.items():
        rows=[]
        for split,names in banks.items():
            for ti,(place,obj) in enumerate(BLOCKS[split]):
                for pi in range(len(names)//3):
                    tr=names[pi*3:pi*3+3]
                    for role in range(2):
                        io,s,alt=tr[role],tr[1-role],tr[2]
                        for fi,template in enumerate(FAMILIES):
                            for order in range(2):
                                a,b=(io,s) if order==0 else (s,io)
                                text,ids,pos=tokenize(template,dict(a=a,b=b,s=s,place=place,obj=obj),tok)
                                corrupt=list(ids);corrupt[pos['s']]=tok.encode(' '+alt,add_special_tokens=False)[0]
                                rows.append(dict(id=f'b004-{split}-t{ti}-p{pi}-r{role}-f{fi}-o{order}',split=split,template=ti,pair=pi,role=role,family=fi,wording=fi,order=order,io=io,subject=s,alternative=alt,text=text,token_ids=ids,corrupt_tokens=corrupt,s2=pos['s'],io_position=pos['a' if order==0 else 'b'],subject_position=pos['b' if order==0 else 'a'],io_id=tok.encode(' '+io,add_special_tokens=False)[0],subject_id=tok.encode(' '+s,add_special_tokens=False)[0],alt_id=corrupt[pos['s']]))
        write(HERE/'data'/f'{model}-candidates.json',rows)
    print('Built disjoint banks', {k:len(v) for k,v in banks.items()},'eligible names',len(pool),flush=True)
if __name__=='__main__':main()
