"""Reproduce manuscript tables, conditional subject bootstrap, and vector figures."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'07_paper'
SNAP=ROOT/'06_results/hf_final_snapshot'
TEX=PAPER/'latex'
GEN=TEX/'generated'
FIG=TEX/'figures'
LABELS={'fpn':'FPN','unet':'U-Net','linknet':'LinkNet','multireslinknet':'MultiResLinkNet',
        'L2_loss_only':'L2: composite loss','L3_c1_only':'L3: eight inputs',
        'L4_c1_c5':'L4: inputs + loss','L5_no_wavelet':'L5: no lifting',
        'L6_no_ssm':'L6: no SSM','L7_singletask':'L7: single-task',
        'L8_no_film':'L8: no FiLM','L9_full':'L9: full S4D model',
        'L10_transformer':'L10: Transformer'}
ORDER=list(LABELS)

def texnum(v,d=2):
    return f'{v:.{d}f}' if np.isfinite(v) else '--'

def rowsfile(name,rows):
    headers={
        'primary_rows.tex':('lrrrrr',r'Model & MAE & MSE & $\cct$ & $\ccs$ & Params (M)'),
        'statistics_rows.tex':('lrrrr',r'Comparator & Median $\Delta$ & Mean $\Delta$ [95\% interval] & $p$ & $p_{\mathrm{Holm}}$'),
        'clinical_rows.tex':('lrrrrr',r'Model & F1 & Precision & Recall & HR MAE & RMSSD MAE \\ & & & & (nominal bpm) & (nominal ms)'),
    }
    cols,header=headers[name]
    body='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)
    (GEN/name).write_text(r'\begin{tabular}{'+cols+r'}\toprule'+'\n'+header+r'\\\midrule'+'\n'+body+'\n'+r'\bottomrule\end{tabular}'+'\n',encoding='utf-8')

def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight',dpi=220)
    plt.close(fig)

def main():
    GEN.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    summary=[]; windows=[]; clinical=[]
    for group in ('baselines','cardiomamba'):
        for path in sorted((SNAP/'provenance'/group/'runs').glob('B_rva__*/summary.json')):
            s=json.loads(path.read_text(encoding='utf-8')); variant=s.get('variant',s.get('model'))
            if variant not in LABELS: continue
            m=s['metrics'].copy()
            if group=='cardiomamba': m['MAE']/=2; m['MSE']/=4
            summary.append(dict(variant=variant,fold=s['fold'],params=s['params'],**m))
            p=PAPER/'evidence'/group/'runs'/path.parent.name/'metrics_windows.parquet'
            w=pd.read_parquet(p); w['variant']=variant; w['fold']=s['fold']; windows.append(w)
            p=p.with_name('metrics_subjects.parquet')
            if p.exists():
                c=pd.read_parquet(p); c['variant']=variant; clinical.append(c)
    R=pd.DataFrame(summary); W=pd.concat(windows,ignore_index=True); C=pd.concat(clinical,ignore_index=True)
    assert len(R)==65 and W['subject'].nunique()==30
    S=W.groupby(['subject','variant'])['CC_temporal'].mean().unstack()
    assert S.notna().all().all()
    rng=np.random.default_rng(20260907)
    ix=rng.integers(0,len(S),(20000,len(S)))
    comparisons=['multireslinknet']+ORDER[4:]
    comparisons.remove('L9_full')
    stats=[]
    for v in comparisons:
        delta=(S['L9_full']-S[v]).to_numpy()
        lo,hi=np.percentile(delta[ix].mean(axis=1),[2.5,97.5])
        stats.append(dict(comparator=v,n=len(delta),p=float(wilcoxon(delta).pvalue),
                          median_delta=float(np.median(delta)),mean_delta=float(delta.mean()),
                          ci_low=float(lo),ci_high=float(hi),subjects_better=int((delta>0).sum())))
    stats=sorted(stats,key=lambda x:x['p']); prev=0
    for i,s in enumerate(stats):
        s['p_holm']=min(1,max(prev,(len(stats)-i)*s['p']));prev=s['p_holm']
    old=pd.read_csv(SNAP/'evaluation/tables/table7_significance.csv').sort_values('p')
    assert np.allclose(old.p.to_numpy(),[s['p'] for s in stats],atol=1e-12)
    assert np.allclose(old.p_holm.to_numpy(),[s['p_holm'] for s in stats],atol=1e-12)
    pd.DataFrame(stats).to_csv(GEN/'subject_effects.csv',index=False)
    S.to_csv(GEN/'subject_correlations.csv')
    rows=[]
    for v in ORDER:
        d=R[R.variant==v]
        rows.append([LABELS[v],texnum(d.MAE.mean(),4),texnum(d.MSE.mean(),4),
                     f'{d.CC_temporal.mean():.2f} $\\pm$ {d.CC_temporal.std():.2f}',
                     f'{d.CC_spectral.mean():.2f} $\\pm$ {d.CC_spectral.std():.2f}',
                     texnum(d.params.mean()/1e6,3)])
    rowsfile('primary_rows.tex',rows)
    rowsfile('statistics_rows.tex',[[LABELS[s['comparator']],f"{s['median_delta']:+.2f}",
            f"{s['mean_delta']:+.2f} [{s['ci_low']:.2f}, {s['ci_high']:.2f}]",
            f"{s['p']:.4f}",f"{s['p_holm']:.4f}"] for s in stats])
    rowsfile('clinical_rows.tex',[[LABELS[v],texnum(d.F1.mean(),3),texnum(d.precision.mean(),3),
             texnum(d.recall.mean(),3),texnum((d.gt_mean_hr_bpm-d.pr_mean_hr_bpm).abs().mean()),
             texnum((d.gt_rmssd_ms-d.pr_rmssd_ms).abs().mean())]
             for v in ['multireslinknet','L9_full','L7_singletask','L8_no_film']
             for d in [C[C.variant==v]]])
    inv=pd.read_csv(PAPER/'evidence/inventory/inventory.csv')
    rec=pd.read_csv(SNAP/'provenance/data/recordings.csv')
    inv['rec_id']=inv['subject']+'__'+inv['scenario']
    merged=rec.merge(inv[['rec_id','n_radar','fs_radar','fs_ecg']],on='rec_id',validate='one_to_one')
    merged['expected_125_n']=np.ceil(merged.n_radar/16).astype(int)
    rate_matches=int((merged.n==merged.expected_125_n).sum())
    assert len(merged)==134 and rate_matches==134
    merged[['rec_id','n_radar','n','expected_125_n','fs_radar','fs_ecg']].to_csv(GEN/'sampling_rate_audit.csv',index=False)
    # Archived implementation must agree with each run's recorded module hashes.
    modules=[]
    for group in ('cardiomamba','baselines'):
        for p in (PAPER/'evidence'/group).glob('crvs_*.py'):
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            cfgs=list((SNAP/'provenance'/group/'runs').glob('B_rva__*/run_config.json'))
            matches=sum(json.loads(c.read_text())['library_sha256'].get(p.name)==digest for c in cfgs)
            modules.append(dict(group=group,file=p.name,sha256=digest,matching_runs=matches,total_runs=len(cfgs)))
    report=dict(canonical_primary_runs=len(R),primary_test_windows_per_model=W.groupby('variant').size().to_dict(),
                subjects=len(S),nb05_statistics_reproduced=True,bootstrap_replicates=20000,bootstrap_seed=20260907,
                sampling_rate_audit_records=len(merged),sampling_rate_125_matches=rate_matches,
                archived_module_hashes=modules,comparisons=stats)
    (GEN/'analysis_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.labelsize':8,'legend.fontsize':7,'axes.titlesize':9})
    # Fold performance is descriptive; each dot is a held-out fold, never an independent repeat.
    fig,ax=plt.subplots(figsize=(7,4.3))
    for i,v in enumerate(ORDER):
        d=R[R.variant==v].sort_values('fold').CC_temporal.to_numpy()
        color='#136b74' if v.startswith('L') else '#747e8a'
        ax.scatter(d,np.full(len(d),i)+np.linspace(-.13,.13,len(d)),s=17,color=color,alpha=.75)
        ax.plot([d.mean()],[i],'D',ms=4,color='#172b4d')
    ax.set_yticks(range(len(ORDER)),[LABELS[v] for v in ORDER]);ax.invert_yaxis()
    ax.set_xlabel('Temporal correlation (%)');ax.set_xlim(0,82);ax.grid(axis='x',alpha=.2)
    ax.set_title('Primary RVA experiment: five fold means and their arithmetic mean',loc='left')
    save(fig,'fold_performance')

    fig,axs=plt.subplots(1,2,figsize=(7,3.1))
    for ax,v,label in zip(axs,['multireslinknet','L7_singletask'],['Full vs MultiResLinkNet','Full vs single-task']):
        x=S[v];y=S['L9_full'];ax.scatter(x,y,s=22,c='#136b74',edgecolors='white',linewidth=.4)
        ax.plot([0,90],[0,90],ls='--',c='#a45d4f',lw=1);ax.set(xlim=(0,90),ylim=(0,90),
                 xlabel=LABELS[v]+' temporal CC (%)',ylabel='Full model temporal CC (%)',title=label)
        ax.grid(alpha=.16)
    fig.tight_layout();save(fig,'subject_pairs')

    rob=pd.read_csv(SNAP/'evaluation/tables/table9_robustness.csv')
    fig,axs=plt.subplots(1,2,figsize=(7,2.8))
    for v,col in [('multireslinknet','#7b8492'),('L9_full','#136b74')]:
        d=rob[(rob.variant==v)&(rob.corruption=='awgn')].sort_values('level')
        axs[0].plot(d.level,d.CC_temporal,'o-',ms=4,c=col,label=LABELS[v])
        d=rob[(rob.variant==v)&(rob.corruption=='motion_drift')].sort_values('level')
        axs[1].plot(d.level,d.CC_temporal,'o-',ms=4,c=col,label=LABELS[v])
    axs[0].set(xlabel='Added feature-space noise (dB SNR)',ylabel='Temporal CC (%)',title='Independent feature noise')
    axs[1].set(xlabel='Shared sinusoidal drift amplitude',ylabel='Temporal CC (%)',title='0.30 Hz nominal drift')
    for ax in axs: ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();save(fig,'robustness')

    fig,axs=plt.subplots(2,2,figsize=(7,5.2))
    ba=[]
    for i,v in enumerate(['multireslinknet','L9_full']):
        for j,(key,unit) in enumerate([('mean_hr_bpm','bpm'),('rmssd_ms','nominal ms')]):
            d=C[C.variant==v];a=d['gt_'+key].to_numpy();b=d['pr_'+key].to_numpy()
            mean=(a+b)/2;diff=b-a;bias=diff.mean();sd=diff.std(ddof=0)
            lo,hi=bias-1.96*sd,bias+1.96*sd
            ax=axs[j,i];ax.scatter(mean,diff,s=17,c='#136b74');ax.axhline(bias,c='#172b4d',lw=1)
            for line in (lo,hi):ax.axhline(line,c='#a45d4f',ls='--',lw=.8)
            ax.set(title=LABELS[v]+(' | HR' if j==0 else ' | RMSSD'),
                   xlabel='Reference/prediction mean ('+unit+')',ylabel='Prediction - reference ('+unit+')')
            ax.text(.03,.95,f'Bias {bias:.2f}; limits [{lo:.1f}, {hi:.1f}]',transform=ax.transAxes,va='top',fontsize=7)
            ax.grid(alpha=.12);ba.append(dict(variant=v,metric=key,bias=bias,loa_low=lo,loa_high=hi,n=len(d)))
    pd.DataFrame(ba).to_csv(GEN/'agreement.csv',index=False)
    fig.tight_layout();save(fig,'agreement')
    samples={}
    for v in ['multireslinknet','L9_full','L8_no_film']:
        group='baselines' if v=='multireslinknet' else 'cardiomamba'
        with np.load(PAPER/'evidence'/group/'runs'/f'B_rva__{v}__f0'/'preds_sample.npz',allow_pickle=False) as z:
            y=z['y'].copy();p=z['p'].copy();subjects=z['subject'].copy()
        if group=='cardiomamba': y=(y+1)/2;p=(p+1)/2
        samples[v]=(y,p,subjects)
    truth=samples['L9_full'][0]
    assert all(np.allclose(a[0],truth,atol=2e-7) and np.array_equal(a[2],samples['L9_full'][2]) for a in samples.values())
    def cc(y,p):return 100*np.corrcoef(y,p)[0,1]
    ranks=np.argsort([cc(y,p) for y,p in zip(truth,samples['L9_full'][1])])
    ids=[int(ranks[int(q*(len(ranks)-1))]) for q in (.25,.5,.75)]
    fig,axs=plt.subplots(3,3,figsize=(7.1,5.0),sharex=True,sharey=True)
    selection=[]
    for row,ix in enumerate(ids):
        for col,(v,(y,p,sub)) in enumerate(samples.items()):
            ax=axs[row,col];t=np.arange(y.shape[1])/125
            ax.plot(t,y[ix],c='#20272f',lw=.7,label='Reference')
            ax.plot(t,p[ix],c='#b34f40',lw=.8,alpha=.9,label='Prediction')
            ax.text(.02,.96,f'CC {cc(y[ix],p[ix]):.1f}%',transform=ax.transAxes,va='top',fontsize=7)
            if row==0:ax.set_title(LABELS[v],fontsize=8)
            if col==0:ax.set_ylabel(f'Q{[25,50,75][row]} | {sub[ix]}\nNormalised ECG',fontsize=7)
            if row==2:ax.set_xlabel('Physical time (s)',fontsize=7)
            ax.set(xlim=(0,8.192),ylim=(-.08,1.08));ax.grid(alpha=.12)
            selection.append(dict(sample_index=ix,quantile=[25,50,75][row],subject=str(sub[ix]),variant=v,cc=cc(y[ix],p[ix])))
    handles,legend_labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',ncol=2,fontsize=7,frameon=False)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'qualitative')
    pd.DataFrame(selection).to_csv(GEN/'qualitative_selection.csv',index=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('archived_module_hashes','comparisons')},indent=2))
    print(pd.DataFrame(stats).to_string(index=False))
    print('module hashes:',modules)

if __name__=='__main__':main()
