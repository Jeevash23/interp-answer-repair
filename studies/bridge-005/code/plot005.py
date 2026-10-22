"""Descriptive plots of audited outcomes; no new analysis or selection rules."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bridge005-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common005 import *

def main():
    assert read(HERE/'results/verification.json')['status']=='passed'
    status=read(HERE/'results/status.json');out=HERE/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    files=[]
    if (HERE/'data/selection.json').exists():
        sel=read(HERE/'data/selection.json')
        scores=sorted(sel['scores'].values(),key=lambda x:(-x['worst'],-x['average'],x['families']))
        y=np.arange(len(scores));height=max(5,0.32*len(scores)+1.8)
        fig,ax=plt.subplots(figsize=(10,height))
        fit=np.array([min_separation(x['fit_separation']) for x in scores])*100
        valid=np.array([x['worst'] for x in scores])*100
        ax.hlines(y,np.minimum(fit,valid),np.maximum(fit,valid),color='#ccd3d9',linewidth=2)
        ax.scatter(fit,y,marker='o',facecolors='white',edgecolors='#6b7280',s=48,label='Fit bank: worst cell',zorder=3)
        ax.scatter(valid,y,color='#33658a',marker='s',s=40,label='Fresh selection bank: worst cell',zorder=4)
        ax.axvline(90,color='#b14d2a',linestyle='--',linewidth=1.5,label='Selection gate: every cell ≥90%')
        ax.set(yticks=y,yticklabels=['F%d / F%d%s'%(*x['families'],'  • selected' if x['families']==sel['families'] else '') for x in scores],xlabel='Lowest classification accuracy across 8 cells (%)',xlim=(max(0,min(fit.min(),valid.min())-5),103))
        ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
        fig.suptitle('Bridge 005 · Clean dependency qualification',x=.15,ha='left',fontweight='bold',fontsize=17)
        ax.set_title('All aligned, behaviorally screened pairs; each point is the minimum over wording, order and reference.',loc='left',fontsize=10,pad=15)
        ax.legend(loc='lower left',bbox_to_anchor=(0,-.19),frameon=False,ncol=1,fontsize=10)
        fig.subplots_adjust(left=.15,right=.97,top=.88,bottom=.2)
        for ext in ['png','svg']:
            p=out/('selection-qualification.'+ext);fig.savefig(p,bbox_inches='tight');files.append(p)
        plt.close(fig)
    if status['test_measured']:
        s=read(HERE/'results/analysis/summary.json');fig,axes=plt.subplots(1,3,figsize=(13,4.4));cells=[f'w{w}_o{o}' for w in range(2) for o in range(2)];labels=[f'F{s["families"][w]} / order {o}' for w in range(2) for o in range(2)]
        for ax,key,title in zip(axes,['recovery','zero','fit_mean'],['Behavioral recovery (logits)','Paired margin · zero reference','Paired margin · fit-mean reference']):
            data=s['primary']['recovery'] if key=='recovery' else s['primary']['margin'][key]
            for i,c in enumerate(cells):
                r=data[c]
                if r is not None:
                    lo,hi=r['adjusted_ci'];ax.plot([lo,hi],[i,i],color='#33658a',linewidth=2);ax.scatter(r['mean'],i,color='#33658a',s=38,zorder=3)
            ax.axvline(0,color='#9b4c34',linestyle='--');ax.set(yticks=range(4),yticklabels=labels,title=title,ylim=(3.5,-.5));ax.grid(axis='x',alpha=.15)
        fig.suptitle('Bridge 005 · Frozen primary comparisons',x=.03,ha='left',fontweight='bold',fontsize=17)
        fig.text(.03,.02,'Intervals: 12-quantity Bonferroni family, 50,000 crossed-bootstrap draws. Negative margins favor the clean recipient.',fontsize=10)
        fig.tight_layout(rect=[0,.07,1,.9])
        for ext in ['png','svg']:
            p=out/('primary-outcomes.'+ext);fig.savefig(p,bbox_inches='tight');files.append(p)
        plt.close(fig)
    write(out/'figure-provenance.json',dict(utc=now(),source=digest(HERE/'data/selection.json') if (HERE/'data/selection.json').exists() else None,summary=digest(HERE/'results/analysis/summary.json') if status['test_measured'] else None,files={p.name:digest(p) for p in files},scope='Descriptive plots of saved audited quantities; no additional inference'))
    print('Saved figures',len(files),flush=True)
if __name__=='__main__':main()
