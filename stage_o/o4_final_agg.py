import json, numpy as np
from collections import defaultdict

d = json.load(open('/home/ubuntu/.hermes/cache/scratch/o4_test_eval.json'))
per = d['per_tag']
bh = d['bh']
print(f"tags={len(per)}  bh={ {k: round(v*100,2) for k,v in bh.items()} }")

combos = {}
_k = 0
for el in (2, 3):
    for dm in (64, 128, 256):
        for lr in (1e-6, 5e-6, 1e-5):
            _k += 1
            combos[_k] = (el, dm, lr)

def combo_of(tag):
    i = int(tag.split('_c')[1].split('_')[0])
    el,dm,lr = combos[i]
    return f"el{el}/dm{dm}/lr{lr:g}"

groups = defaultdict(list)
for tag, r in per.items():
    w = r['window']
    i = int(tag.split('_c')[1].split('_')[0])
    seed = 42 if tag.startswith('O4A') else int(tag.split('_')[-1])
    groups[(w,i)].append((seed, tag, r))

print(f"\n{'窗':<4}{'组合':<20}{'seeds':<18}{'中位ret%':>9}{'min':>8}{'max':>8}{'中位E%/笔':>10}{'正E':>4}{'中位CI95'}")
for (w,i) in sorted(groups, key=lambda x:(x[0],x[1])):
    rows = groups[(w,i)]
    def _g(r,k,d=0.0): 
        v=r.get(k); return d if v is None else v
    rets = [_g(r,'ret_total') for _,_,r in rows]
    Es = [_g(r,'breakeven_E') for _,_,r in rows]
    cis = [r.get('boot_ci') or [0.0,0.0] for _,_,r in rows]
    npos = sum(1 for e in Es if e>0)
    se = sorted(s for s,_,_ in rows)
    lo = float(np.median([c[0] for c in cis])); hi = float(np.median([c[1] for c in cis]))
    print(f"{w:<5}{combo_of(rows[0][1]):<20}{str(se):<18}"
          f"{np.median(rets)*100:+9.2f}{min(rets)*100:+8.2f}{max(rets)*100:+8.2f}"
          f"{np.median(Es)*100:+10.3f}{npos:>4}/{len(rows)}"
          f"  [{lo*100:+.2f}%, {hi*100:+.2f}%]")

print("\n=== 每窗口 best combo (按 4-seed 中位 E) ===")
for w in ['W1','W2','W3','W4']:
    best = None
    for (ww,i),rows in groups.items():
        if ww!=w: continue
        Es=[r.get('breakeven_E') or 0.0 for _,_,r in rows]
        medE = float(np.median(Es))
        rets=[r.get('ret_total') or 0.0 for _,_,r in rows]
        npos = sum(1 for e in Es if e>0)
        if best is None or medE>best[1]:
            best=(i,medE,float(np.median(rets)),npos,len(rows),rows[0][1])
    if best:
        i,medE,medR,npos,n,tag0 = best
        print(f"{w}: c{i:02d} {combo_of(tag0)}  中位E={medE*100:+.3f}%/笔  中位ret={medR*100:+.2f}%  正E={npos}/{n}  (B&H={bh[w]*100:+.2f}%)")

# 全 32 模型汇总
allE = [(r.get('breakeven_E') or 0.0) for r in per.values()]
allR = [(r.get('ret_total') or 0.0) for r in per.values()]
print(f"\n全部 {len(per)} 模型(4-seed 展开): 中位 E={np.median(allE)*100:+.3f}%/笔, 中位 ret={np.median(allR)*100:+.2f}%, 正E占比={sum(1 for e in allE if e>0)}/{len(allE)}")
