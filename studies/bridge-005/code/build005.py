"""Tokenization-only fresh data construction, before any new scientific outcomes."""
import itertools,re
from transformers import AutoTokenizer
from common005 import *
from build003 import NAMES,FAMILIES as OLD_FAMILIES,tokenize
from build004 import EXTRA

EXTRA_NEW='''Abigail Adele Agnes Aidan Aiden Alana Alec Alejandro Alexis Alfonso Ali Alison Alistair Allan Alma Amos Anders Andrea Angelina Anita Anja Annette Annie Arabella Ariana Arnold Ashton Athena Aubrey Aurora Ava Avery Axel Bailey Beatrice Becky Bella Benedict Ben Benson Bethany Beth Bianca Bonnie Boris Brent Brett Bridget Brittany Brock Brooke Byron Caitlin Cameron Candace Carla Carmen Carter Cassandra Cassie Catalina Cecilia Celeste Chadwick Chandler Charlene Chelsea Cheryl Chip Chloe Chris Connor Connie Conrad Cooper Corbin Corey Crystal Cynthia Dakota Damian Damon Dana Danielle Daphne Darrell Darren Dawn Deborah Delia Denise Derek Derrick Desmond Devon Diego Dominic Don Donovan Dora Dorian Drew Dustin Earl Eden Edmund Eduardo Elaine Elena Elias Elijah Elise Eliza Elliott Ellis Elsa Emerson Emmett Enrique Enzo Erin Esther Evan Everett Fabian Faith Fernando Fiona Floyd Forrest Francesca Francisco Gabriela Gabriel Gail Gareth Gavin Geoffrey Geraldine Gina Glen Glenn Gordon Grant Gretchen Griffin Gus Gwen Hailey Haley Hank Harriet Harrison Harvey Hazel Hector Heidi Herbert Herman Hope Horace Hudson Hugo Hunter Ida Ignacio Imogen Ingrid Iris Irving Ivan Jacqueline Jade Jake Jamie Jared Jasmine Jasper Javier Jay Jayden Jeanette Jed Jenna Jenny Jerome Jill Joanna Joanne Jocelyn Jonah Jonas Jorge Josie Joy Judith Jules June Kai Kara Karl Kate Kathleen Katie Kay Kayla Kelly Ken Kendrick Kent Kerry Kieran Klaus Kristen Kristin Kristina Kurt Kyle Lana Lance Lara Latoya Leah Lena Lennox Leon Leroy Lester Levi Lila Lily Lionel Liz Lloyd Lois Lorenzo Lorraine Luca Lucas Lucia Lucille Luis Luther Lydia Mack Madison Mae Maggie Malcolm Mandy Marc Marcel Marco Margaret Mariah Marianne Marilyn Marisa Marjorie Marshall Marta Matilda Maureen Max Maxwell Maya Mel Melanie Melody Mia Miguel Miles Milo Mindy Miranda Mitchell Molly Morgan Morris Myra Nadia Naomi Natalia Natalie Natasha Ned Nell Nelly Nestor Neville Nick Nigel Noel Nolan Nora Octavia Omar Opal Otto Owen Pablo Paige Parker Pascal Pat Pearl Pedro Penny Percy Petra Phoebe Pierce Pierre Preston Quentin Quincy Quinn Rafael Raj Ramon Randall Randy Raul Regina Reginald Remy Rex Rhonda Ricardo Riley Rob Roberta Roberto Rochelle Rodney Roland Roman Romeo Rosa Rosalind Rosemary Ross Rowan Roxanne Rudy Rufus Sabrina Sadie Sal Salvador Sam Samson Sandy Santiago Saul Savannah Selena Serena Sergio Shannon Shaun Shawn Sheila Shelby Sid Sidney Silas Silvia Simone Sonia Sonya Spencer Stacy Stella Sterling Stuart Summer Suzanne Sydney Tamara Tanya Tara Terrence Thelma Theo Theodore Tobias Toby Tonya Tracey Tracy Trent Trevor Tricia Trudy Tucker Val Valerie Vanessa Vera Vernon Vicki Vince Violet Vivian Wallace Wanda Wesley Whitney Wilbur Winston Wyatt Xavier Yasmin Yolanda Yvonne Zach Zack Zoe Zora'''.split()
FAMILIES=OLD_FAMILIES+[
 'After {a} and {b} arrived at the {place}, {s} offered a {obj} to',
 'When {a} and {b} entered the {place}, {s} returned a {obj} to',
]
BLOCKS={
 'calibration':[('abbey','candle'),('attic','blanket')],
 'screen':[('balcony','cupcake'),('barracks','badge'),('bistro','sandwich')],
 'fit':[('boathouse','paddle'),('boutique','brooch'),('brewery','glass'),('bunker','helmet'),('cabin','quilt'),('cathedral','booklet')],
 'selection':[('cellar','jug'),('chapel','ribbon'),('chateau','bracelet'),('clubhouse','bat'),('conservatory','bouquet'),('cottage','cushion')],
 'qualification':[('courtyard','flute'),('deli','sausage'),('depot','tool'),('dormitory','notebook'),('embassy','document'),('fortress','sword')],
 'test':[('gazebo','pillow'),('gymnasium','racket'),('hangar','wrench'),('hostel','suitcase'),('lighthouse','lantern'),('lodge','mug'),('mansion','pendant'),('monastery','scroll'),('orchard','peach'),('pavilion','fan'),('plaza','pamphlet'),('tavern','goblet')],
}
SIZES={'calibration':6,'screen':12,'fit':18,'selection':18,'qualification':18,'test':36}
def main():
    assert not (HERE/'data/design.json').exists()
    old=read(PREV/'data/design.json');exclude=set(old['excluded_names'])|set(sum(old['names'].values(),[]));sources={}
    # Include named actors/corruptions in both original projects, not only bridge banks.
    def collect(x):
        if isinstance(x,list):
            for v in x:collect(v)
        elif isinstance(x,dict):
            for k,v in x.items():
                if k in ['io','subject','alternative','io_name','subject_name','corrupt_name','name_a','name_b','name_c'] and isinstance(v,str):exclude.add(v)
                else:collect(v)
    for directory in [ROOT / 'studies/transport/data',ROOT / 'studies/routing/data']:
        for p in sorted(directory.iterdir()):
            if p.suffix=='.jsonl' or (p.suffix=='.json' and any(k in p.name for k in ['prompts','recipients'])):
                sources[os.path.relpath(p,HERE)]=digest(p)
                if p.suffix=='.jsonl':
                    for line in p.read_text().splitlines():
                        if line.strip():collect(json.loads(line))
                else:collect(read(p))
    tok=AutoTokenizer.from_pretrained(str(ROOT / 'models/gpt2-medium'),local_files_only=True)
    pool=sorted({n for n in NAMES+EXTRA+EXTRA_NEW if n not in exclude and len(tok.encode(' '+n,add_special_tokens=False))==1})
    assert len(pool)>=sum(SIZES.values()),(len(pool),sum(SIZES.values()))
    np.random.default_rng(SEED).shuffle(pool);banks={};offset=0
    for split,n in SIZES.items():banks[split]=pool[offset:offset+n];offset+=n
    prior_blocks={tuple(x) for d in [old,read(ROOT/'studies/bridge-003/data/design.json')] for v in d['blocks'].values() for x in v}
    assert not prior_blocks.intersection(x for v in BLOCKS.values() for x in v)
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
                            rows.append(dict(id=f'b005-{split}-t{ti}-p{pi}-r{role}-f{fi}-o{order}',split=split,template=ti,pair=pi,role=role,family=fi,wording=fi,order=order,io=io,subject=s,alternative=alt,text=text,token_ids=ids,corrupt_tokens=corrupt,s2=pos['s'],io_position=pos['a' if order==0 else 'b'],subject_position=pos['b' if order==0 else 'a'],io_id=tok.encode(' '+io,add_special_tokens=False)[0],subject_id=tok.encode(' '+s,add_special_tokens=False)[0],alt_id=corrupt[pos['s']]))
    groups={}
    for r in rows:groups.setdefault((r['split'],r['template'],r['pair'],r['role'],r['order']),{})[r['family']]=r
    eligible=[]
    for a,b in itertools.combinations(range(len(FAMILIES)),2):
        if all(len(g[a]['token_ids'])==len(g[b]['token_ids']) and g[a]['s2']==g[b]['s2'] for g in groups.values()):eligible.append([a,b])
    write(HERE/'data/design.json',dict(seed=SEED,names=banks,blocks=BLOCKS,families=FAMILIES,excluded_names=sorted(exclude),tokenization_pool=len(pool),alignment_eligible_pairs=eligible,prior_name_sources=sources,scope='Eight inherited grammar skeletons plus two new candidates; fresh actor banks and context pairs; conceptual replication if wording pair changes'))
    write(HERE/'data/candidates.json',rows)
    print('Built tokenization-only data',dict(pool=len(pool),excluded=len(exclude),banks={k:len(v) for k,v in banks.items()},rows=len(rows),aligned_pairs=len(eligible)),flush=True)
if __name__=='__main__':main()
