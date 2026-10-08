# -*- coding: utf-8 -*-
"""보조 분석: 비명 '매우 낮음' 창이 이벤트 겹침·길이·군중 동반과 관련 있는지. 외부(real31 scream 74) + AudioSet 학습분."""
import os, sys, csv, re
import numpy as np
sys.path[:0] = [os.path.expanduser('~/max78000-proj/tools'), os.path.expanduser('~/max78000-proj/scripts')]
import label_audit as LA
import eval_external as E
names = LA.label_names()
M = LA.ext_meta()
ext = LA.load('ext'); se = LA.class_scores(ext['P'], names)
y = E.labels(M, 'real31')
rows = []
for i, r in enumerate(M):
    if y[i] != 2:
        continue
    ws = float(r['win_start_s']); we = ws + 1.024
    es, ee = float(r['event_start']), float(r['event_end'])
    ov = max(0.0, min(we, ee) - max(ws, es))
    rows.append((se['scream'][i], ov, float(r['event_dur']), r['scream_sub'], r['labels_all'], r['speaker'], r['hardneg_like']))
s = np.array([x[0] for x in rows]); ov = np.array([x[1] for x in rows]); dur = np.array([x[2] for x in rows])
vl = s < 0.02
print('외부 scream(real31) n', len(s), ' 매우낮음', vl.sum())
print(' 겹침 s  매우낮음 중앙 %.2f  나머지 중앙 %.2f' % (np.median(ov[vl]), np.median(ov[~vl])))
print(' 이벤트 길이 s  매우낮음 중앙 %.2f  나머지 중앙 %.2f' % (np.median(dur[vl]), np.median(dur[~vl])))
print(' 겹침<0.5s 비율  매우낮음 %.2f  나머지 %.2f' % ((ov[vl] < 0.5).mean(), (ov[~vl] < 0.5).mean()))
print(' 겹침 전체(>=1.0)  매우낮음 %.2f  나머지 %.2f' % ((ov[vl] >= 1.0).mean(), (ov[~vl] >= 1.0).mean()))
from collections import Counter
print(' scream_sub 매우낮음', Counter(x[3] for x, v in zip(rows, vl) if v).most_common(6))
print(' scream_sub 나머지 ', Counter(x[3] for x, v in zip(rows, vl) if not v).most_common(6))
crowd = np.array([bool(re.search('Crowd|Cheering|Children|Chatter|Hubbub', x[4])) for x in rows])
print(' 군중류 라벨 동반  매우낮음 %.2f  나머지 %.2f' % (crowd[vl].mean(), crowd[~vl].mean()))
# 상위 PANNs 라벨 (매우낮음 창)
P = ext['P'].astype(np.float32)
idx = [i for i, r in enumerate(M) if y[i] == 2]
top = Counter()
for k, i in enumerate(idx):
    if vl[k]:
        top[names[int(P[i].argmax())]] += 1
print(' 매우낮음 창의 PANNs 최상위 라벨', top.most_common(10))
# AudioSet 학습분 scream: ev_overlap vs 점수
z = LA.load('as_train'); sa = LA.class_scores(z['P'], names)
sel = z['y'] == 2
ovs, sc, in31 = [], [], []
for n, v, in_ in zip(z['notes'][sel], sa['scream'][sel], [nn.endswith('in_v31=1') for nn in z['notes'][sel]]):
    m = re.search(r'ev_overlap=([0-9.]+)', n)
    ovs.append(float(m.group(1)) if m else np.nan); sc.append(v); in31.append(in_)
ovs, sc, in31 = np.array(ovs), np.array(sc), np.array(in31)
vl2 = sc < 0.02
print('AudioSet 학습 scream n', len(sc), ' ev_overlap 단위 예:', z['notes'][sel][0])
for lab, m_ in (('Screaming 보유', in31), ('Yell·Shout 만', ~in31)):
    a = vl2 & m_; b = ~vl2 & m_
    print(f' {lab}: 매우낮음 {a.sum()} 겹침중앙 {np.nanmedian(ovs[a]):.2f} / 나머지 {b.sum()} 겹침중앙 {np.nanmedian(ovs[b]):.2f}')
top2 = Counter()
P2 = z['P'].astype(np.float32)
for i in np.flatnonzero(sel):
    if sa['scream'][i] < 0.02 and z['notes'][i].endswith('in_v31=1'):
        top2[names[int(P2[i].argmax())]] += 1
print(' AudioSet Screaming보유 매우낮음 창의 PANNs 최상위 라벨', top2.most_common(10))
# FSD50K 시험 scream 매우낮음의 최상위 라벨
zt = np.load(LA.TEST_CACHE, allow_pickle=True); st = LA.class_scores(zt['P'], names)
top3 = Counter()
for i in np.flatnonzero(zt['y'] == 2):
    if st['scream'][i] < 0.02:
        top3[names[int(zt['P'][i].astype(np.float32).argmax())]] += 1
print(' FSD50K 시험 scream 매우낮음 창의 최상위 라벨', top3.most_common(8))
# PANNs 가 Yell/Shout 를 얼마나 내놓는가: 전 출처 양성에서 shout 최대
print(' shout 점수 최대: FSD train %.3f  AS train %.3f  ext %.3f' % (LA.class_scores(LA.load('v1_train')['P'], names)['shout'].max(), sa['shout'].max(), se['shout'].max()))
