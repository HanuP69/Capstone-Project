import matplotlib; matplotlib.use("Agg")
from IPython.display import display
import random, re, math, zlib, datetime as dt, warnings
from collections import Counter, defaultdict
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

warnings.filterwarnings('ignore')
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 40); pd.set_option('display.precision', 3)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})

SEED = 7

# ---- ASSUMPTIONS of the SIMULATED generator (not measured on a real LLM) --------------------
PLAIN_ACC = {   # P(claim correct) for an LLM with NO retrieval, by category
    'exam_dates': .70, 'fee_deadlines': .70, 'hostel_rules': .78, 'circulars': .62, 'syllabus': .82,   # institution A
    'admissions': .68, 'fees': .66, 'hostel': .76, 'exams': .70, 'library': .80}                     # institution B
P_CORRECT_IF_GOLD = .93   # RAG: P(correct) when the gold passage is in the top-k context
P_USE_STALE       = .60   # RAG: P(copies the superseded value) when a stale passage outranks the current one
P_FABRICATE       = .25   # among wrong answers: share that invent a non-existent rule (-> 'unverifiable')
TOPK_GEN, TOPK_GROUND = 3, 3
QUERIES_PER_FACT, SIDE_CLAIMS = 4, 2   # each query -> 1 main claim + 2 side claims in the answer

# ---------- typed values: date / time / amount / int ----------
def fmt_date(d): return f"{d.day} {d.strftime('%B')} {d.year}"
def fmt_time(m): h, mi = divmod(m, 60); return f"{(h % 12) or 12}:{mi:02d} {'AM' if h < 12 else 'PM'}"
def fmt(v):
    t, x = v
    return {'date': fmt_date, 'time': fmt_time, 'amount': lambda a: f"Rs {a}", 'int': str, 'text': str}[t](x)

def parse_value(s):
    s = s.strip()
    if re.fullmatch(r'\d{1,2} [A-Za-z]+ \d{4}', s):   return ('date', dt.datetime.strptime(s, '%d %B %Y').date())
    m = re.fullmatch(r'(\d{1,2}):(\d{2}) (AM|PM)', s)
    if m: return ('time', (int(m[1]) % 12 + (12 if m[3] == 'PM' else 0)) * 60 + int(m[2]))
    m = re.fullmatch(r'Rs (\d+)', s)
    if m: return ('amount', int(m[1]))
    if re.fullmatch(r'\d+', s): return ('int', int(s))
    return ('text', s)

def perturb(v, rng):
    """plausible-but-wrong value of the same type"""
    t, x = v
    if t == 'date':   return (t, x + dt.timedelta(days=rng.choice([-1, 1]) * rng.randint(1, 21)))
    if t == 'time':
        n = x + rng.choice([-1, 1]) * rng.choice([30, 60, 90]); n = max(0, min(23 * 60 + 30, n))
        return (t, n if n != x else x + 30)
    if t == 'amount':
        n = int(round(x * rng.choice([.5, .8, 1.25, 1.5, 2.0]) / 50) * 50)
        return (t, n if n != x else x + 50)
    n = max(1, x + rng.choice([-2, -1, 1, 2])); return (t, n if n != x else x + 1)

def fake_value(vtype, rng):
    return ('int', rng.randint(2, 12)) if vtype == 'int' else ('amount', rng.randrange(100, 2001, 50))

tok  = lambda s: re.findall(r'[a-z0-9]+', s.lower())
norm = lambda s: ' '.join(tok(s))
STOP = {'the', 'for', 'of', 'is', 'a', 'an', 'and', 'to', 'in', 'on'}

def contains_seq(hay, needle):
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))

# ---------- student-style query noise (abbreviations / synonyms / dropped words) ----------
SYN = {'last date': ['deadline'], 'fee': ['charges'], 'semester': ['sem'], 'maximum': ['max'], 'minimum': ['min'],
       'payment': ['pay'], 'exam': ['examination'], 'time': ['timing'], 'hours': ['hrs'], 'number of': ['no of'],
       'registration': ['reg']}
def noisy_query(field, entity, rng, p):
    q = f"{field} {entity}".lower()
    if rng.random() < p:
        for k, v in SYN.items():
            if k in q and rng.random() < .7: q = q.replace(k, rng.choice(v))
    if rng.random() < p * .5:
        t = q.split()
        if len(t) > 3: t.pop(rng.randrange(len(t))); q = ' '.join(t)
    return q

class Rule:
    def __init__(self, name, field, entity, check): self.name, self.field, self.entity, self.check = name, norm(field), (norm(entity) if entity else None), check
    def applies(self, e, f): return self.field == f and (self.entity is None or self.entity == e)

def win(lo, hi):  return lambda e, v: v[0] == 'date' and lo <= v[1] <= hi
def rng_amt(lo, hi): return lambda e, v: v[0] in ('amount', 'int') and lo <= v[1] <= hi
def eq_to(truth): return lambda e, v: v == truth
def rtime(rng, lo, hi, step=30): return ('time', rng.choice(range(lo, hi + 1, step)))

class Inst: pass

def build_institution(name, spec, rng, stale_cats, record_cats, prefixes, fakes, doc_fmt, abbr, n_filler, rules_fn):
    I = Inst(); I.name = name
    facts = []
    for i, f in enumerate(spec):
        f = dict(f, fid=f"{name}{i:03d}", stale=None)
        if f['cat'] in stale_cats and rng.random() < .5: f['stale'] = perturb(f['truth'], rng)
        facts.append(f)
    for cat in sorted({f['cat'] for f in facts}):                # dev/test split by FACT (no leakage), stratified by category
        fs = [f for f in facts if f['cat'] == cat]; rng.shuffle(fs)
        for j, f in enumerate(fs): f['split'] = 'dev' if j % 2 == 0 else 'test'
    passages, n = [], 0
    def ref(): nonlocal n; n += 1; return f"{abbr}-2026-{n:03d}"
    for f in facts:
        pre = prefixes[f['cat']]
        if f['stale'] is not None:
            old_d = dt.date(2026, 3, 1) + dt.timedelta(days=rng.randint(0, 60))
            new_d = dt.date(2026, 7, 1) + dt.timedelta(days=rng.randint(0, 90))
            po = dict(pid=f"{f['fid']}-old", fid=f['fid'], text=doc_fmt(pre, ref(), f['entity'], f['field'], fmt(f['stale']), ''), issued=old_d, supersedes=None)
            note = f" This notice supersedes the earlier notice dated {old_d.day:02d}{old_d.strftime('%b%Y')}."
            pn = dict(pid=f"{f['fid']}-new", fid=f['fid'], text=doc_fmt(pre, ref(), f['entity'], f['field'], fmt(f['truth']), note), issued=new_d, supersedes=po['pid'])
            passages += [po, pn]; f['gold_pid'], f['stale_pid'] = pn['pid'], po['pid']
        else:
            new_d = dt.date(2026, 7, 1) + dt.timedelta(days=rng.randint(0, 90))
            p = dict(pid=f"{f['fid']}-cur", fid=f['fid'], text=doc_fmt(pre, ref(), f['entity'], f['field'], fmt(f['truth']), ''), issued=new_d, supersedes=None)
            passages.append(p); f['gold_pid'], f['stale_pid'] = p['pid'], None
    pairs = {(norm(f['entity']), norm(f['field'])) for f in facts}   # lexical distractors: entity of one fact + field of another
    for k in range(n_filler):
        while True:
            e, fl = rng.choice(facts)['entity'], rng.choice(facts)['field']
            if (norm(e), norm(fl)) not in pairs: break
        passages.append(dict(pid=f"{name}-fill{k}", fid=None, text=f"General notice [{ref()}]: students are advised to read the {fl} guidelines for {e} on the portal.", issued=dt.date(2026, 6, 1), supersedes=None))
    for p in passages: p['toks'] = tok(p['text'])
    I.facts, I.passages = facts, passages
    I.pby = {p['pid']: p for p in passages}
    I.records = {(norm(f['entity']), norm(f['field'])): f['truth'] for f in facts if f['cat'] in record_cats}
    I.rules = rules_fn(facts)
    I.fakes, I.abbr = fakes, abbr
    return I

# ------------------------------------------------ institution A (NITJ-like) ------------------------------------------------
COURSES = [("CS301","Data Structures"),("CS302","Operating Systems"),("CS303","Computer Networks"),("CS304","Database Management Systems"),
           ("CS305","Theory of Computation"),("CS306","Compiler Design"),("CS307","Machine Learning"),("EC301","Digital Logic Design"),
           ("EC302","Signals and Systems"),("ME301","Thermodynamics"),("ME302","Fluid Mechanics"),("MA301","Engineering Mathematics"),
           ("MA302","Linear Algebra"),("PH301","Quantum Physics"),("EE301","Control Systems"),("EC303","VLSI Design"),
           ("CS308","Software Engineering"),("CS309","Computer Architecture"),("MA303","Discrete Mathematics"),("MA304","Probability and Statistics")]
SEM_BASE = {'Odd Semester 2025': dt.date(2025,7,20), 'Even Semester 2026': dt.date(2026,1,10), 'Odd Semester 2026': dt.date(2026,7,20),
            'Even Semester 2027': dt.date(2027,1,10), 'Summer Term 2026': dt.date(2026,5,15), 'Summer Term 2027': dt.date(2027,5,15)}
FEE_DATE_FIELDS = ['tuition fee payment last date', 'hostel fee payment last date', 'backlog exam form last date', 're-registration last date']
HOSTEL_ENT = ["Boys' hostels", "Girls' hostels", "International students' hostel"]
HOSTEL_FIELDS = [('gate closing time','time',(21*60,23*60+30)), ('maximum guest stay (nights)','int',(1,5)), ('mess rebate minimum absence (days)','int',(3,10)),
                 ('caution deposit (Rs)','amount',(2000,10000)), ('late entry fine (Rs)','amount',(50,500)), ('visiting hours end','time',(17*60,20*60)),
                 ('laundry days per week','int',(1,4)), ('room change window (days)','int',(5,20)), ('maximum visitors per day','int',(1,4)), ('quiet hours begin','time',(21*60,23*60))]
HOSTEL_POLICY_FIELDS = {f for f, _, _ in HOSTEL_FIELDS[:6]}   # these 6 have an explicit policy clause (exact rule); other 4 do not
ACTIVITIES = ['Grade improvement application','Summer term registration','Scholarship renewal','Migration certificate request','Transcript request',
              'Provisional degree certificate','Project submission','Internship NOC','Fee concession application','Bonafide certificate',
              'Hostel allotment','Library clearance','No-dues certificate','Convocation registration','Re-evaluation request']

def spec_A(rng):
    S = []
    for code_, nm in COURSES:
        S.append(dict(cat='exam_dates', entity=f"{code_} {nm}", field='mid-semester exam date', truth=('date', dt.date(2026,10,5) + dt.timedelta(days=rng.randint(0,12)))))
        S.append(dict(cat='exam_dates', entity=f"{code_} {nm}", field='end-semester exam date', truth=('date', dt.date(2026,12,1) + dt.timedelta(days=rng.randint(0,18)))))
    for sem, base in SEM_BASE.items():
        for fld in FEE_DATE_FIELDS: S.append(dict(cat='fee_deadlines', entity=sem, field=fld, truth=('date', base + dt.timedelta(days=rng.randint(4,26)))))
        S.append(dict(cat='fee_deadlines', entity=sem, field='late fee per day (Rs)', truth=('amount', rng.randrange(50,201,50))))
    for ent in HOSTEL_ENT:
        for fld, vt, (lo, hi) in HOSTEL_FIELDS:
            v = rtime(rng, lo, hi) if vt == 'time' else ('int', rng.randint(lo, hi)) if vt == 'int' else ('amount', rng.randrange(lo, hi + 1, 500 if lo >= 2000 else 50))
            S.append(dict(cat='hostel_rules', entity=ent, field=fld, truth=v))
    for act in ACTIVITIES:
        S.append(dict(cat='circulars', entity=act, field='last date', truth=('date', dt.date(2026,11,15) + dt.timedelta(days=rng.randint(0,40)))))
        S.append(dict(cat='circulars', entity=act, field='processing fee (Rs)', truth=('amount', rng.randrange(50,501,50))))
    for _, nm in COURSES[:15]:
        S.append(dict(cat='syllabus', entity=nm, field='credits', truth=('int', rng.randint(2,4))))
        S.append(dict(cat='syllabus', entity=nm, field='number of units', truth=('int', rng.randint(4,6))))
    return S

def rules_A(facts):
    R = [Rule('mid-sem exam window', 'mid-semester exam date', None, win(dt.date(2026,10,5), dt.date(2026,10,17))),
         Rule('end-sem exam window', 'end-semester exam date', None, win(dt.date(2026,12,1), dt.date(2026,12,19))),
         Rule('late fee cap', 'late fee per day (Rs)', None, rng_amt(50, 250)),
         Rule('circular deadline window', 'last date', None, win(dt.date(2026,11,15), dt.date(2026,12,31))),
         Rule('circular fee cap', 'processing fee (Rs)', None, rng_amt(50, 1000))]
    for sem, base in SEM_BASE.items():
        for fld in FEE_DATE_FIELDS: R.append(Rule(f'{sem} fee window', fld, sem, win(base, base + dt.timedelta(days=30))))
    for f in facts:
        if f['cat'] == 'hostel_rules' and f['field'] in HOSTEL_POLICY_FIELDS: R.append(Rule('hostel clause', f['field'], f['entity'], eq_to(f['truth'])))
    return R

PREF_A = {'exam_dates':'Examination Cell timetable','fee_deadlines':'Accounts Section notice','hostel_rules':'Hostel Handbook clause','circulars':'Academic circular','syllabus':'Syllabus booklet'}
FAKE_A = {'exam_dates':[('re-evaluation fee waiver (Rs)','amount'),('grace period (days)','int')], 'fee_deadlines':[('installment facility (months)','int'),('early payment discount (Rs)','amount')],
          'hostel_rules':[('weekend leave quota (days)','int'),('mess guest charge (Rs)','amount')], 'circulars':[('appeal window (days)','int'),('express processing fee (Rs)','amount')],
          'syllabus':[('prerequisite credits','int'),('minimum attendance for lab (days)','int')]}
docA = lambda pre, ref, e, fld, v, note: f"{pre} ({ref}): the {fld} for {e} is {v}.{note}"

# ------------------------------------------------ institution B (public-university-like) ------------------------------------------------
PROGS = ['MSc Data Science','MBA','BA Economics','BSc Physics','MA English','PhD Computer Science']
def spec_B(rng):
    S = []
    for p in PROGS:
        S += [dict(cat='admissions', entity=p, field='application last date', truth=('date', dt.date(2027,1,10) + dt.timedelta(days=rng.randint(0,60)))),
              dict(cat='admissions', entity=p, field='application fee (Rs)', truth=('amount', rng.randrange(500,1501,50))),
              dict(cat='admissions', entity=p, field='entrance test date', truth=('date', dt.date(2027,2,15) + dt.timedelta(days=rng.randint(0,45)))),
              dict(cat='fees', entity=p, field='annual tuition (Rs)', truth=('amount', rng.randrange(40000,150001,5000))),
              dict(cat='fees', entity=p, field='late payment fine per week (Rs)', truth=('amount', rng.randrange(100,501,50))),
              dict(cat='fees', entity=p, field='refund window (days)', truth=('int', rng.randint(7,30)))]
    for blk in ['Block North','Block South','Block East','Block West']:
        S += [dict(cat='hostel', entity=blk, field='curfew time', truth=rtime(rng, 21*60, 23*60+30)),
              dict(cat='hostel', entity=blk, field='guest policy max nights', truth=('int', rng.randint(1,4))),
              dict(cat='hostel', entity=blk, field='security deposit (Rs)', truth=('amount', rng.randrange(3000,8001,500))),
              dict(cat='hostel', entity=blk, field='mess monthly charge (Rs)', truth=('amount', rng.randrange(3000,5001,500))),
              dict(cat='hostel', entity=blk, field='visitor hours end', truth=rtime(rng, 17*60, 20*60))]
    for _, nm in COURSES[:8]:
        S += [dict(cat='exams', entity=nm, field='exam date', truth=('date', dt.date(2026,12,1) + dt.timedelta(days=rng.randint(0,19)))),
              dict(cat='exams', entity=nm, field='re-exam application last date', truth=('date', dt.date(2026,12,22) + dt.timedelta(days=rng.randint(0,19))))]
    for mem in ['UG student','PG student','Research scholar','Faculty']:
        S += [dict(cat='library', entity=mem, field='max books allowed', truth=('int', rng.randint(2,10))),
              dict(cat='library', entity=mem, field='loan period (days)', truth=('int', rng.randint(7,30))),
              dict(cat='library', entity=mem, field='fine per day (Rs)', truth=('int', rng.randint(2,20))),
              dict(cat='library', entity=mem, field='renewals allowed', truth=('int', rng.randint(1,3)))]
    return S

def rules_B(facts):
    R = [Rule('exam window', 'exam date', None, win(dt.date(2026,12,1), dt.date(2026,12,20))),
         Rule('refund window cap', 'refund window (days)', None, rng_amt(7, 45)),
         Rule('loan period cap', 'loan period (days)', None, rng_amt(7, 60)),
         Rule('books cap', 'max books allowed', None, rng_amt(1, 20)),
         Rule('fine cap', 'fine per day (Rs)', None, rng_amt(1, 20))]
    for f in facts:
        if f['cat'] == 'hostel' and f['field'] in {'curfew time', 'guest policy max nights', 'security deposit (Rs)'}: R.append(Rule('hostel clause', f['field'], f['entity'], eq_to(f['truth'])))
    return R

PREF_B = {'admissions':'Admissions Office','fees':'Finance Office','hostel':'Residence Life','exams':'Controller of Examinations','library':'Library'}
FAKE_B = {'admissions':[('scholarship quota (seats)','int'),('waitlist fee (Rs)','amount')], 'fees':[('sibling discount (Rs)','amount'),('installment count','int')],
          'hostel':[('weekend leave quota (days)','int'),('guest meal charge (Rs)','amount')], 'exams':[('revaluation fee (Rs)','amount'),('grace marks limit','int')],
          'library':[('inter-library loan limit','int'),('late return waiver (days)','int')]}
docB = lambda pre, ref, e, fld, v, note: f"{pre} FAQ [{ref}] - {e}: {fld} is {v}.{note}"

rng0 = random.Random(SEED)
A = build_institution('A', spec_A(rng0), rng0, stale_cats={'circulars'}, record_cats={'exam_dates','fee_deadlines','circulars'},
                      prefixes=PREF_A, fakes=FAKE_A, doc_fmt=docA, abbr='NITJ', n_filler=150, rules_fn=rules_A)
B = build_institution('B', spec_B(rng0), rng0, stale_cats={'admissions'}, record_cats=set(),
                      prefixes=PREF_B, fakes=FAKE_B, doc_fmt=docB, abbr='PUX', n_filler=100, rules_fn=rules_B)

for I in (A, B):
    d = pd.DataFrame(I.facts)
    print(f"Institution {I.name}: {len(I.facts)} facts, {len(I.passages)} passages, {len(I.records)} records, {len(I.rules)} rules, {int(d.stale.notna().sum())} facts with a superseded notice")
    print(d.groupby('cat').agg(facts=('fid','count'), stale=('stale', lambda s: s.notna().sum()), dev=('split', lambda s: (s=='dev').sum()), test=('split', lambda s: (s=='test').sum())).T.to_string(), '\n')
print('Example passage (A):', A.passages[A.facts[0]['gold_pid'] and 0]['text'])
print('Example superseded pair (A):'); sp_ = next(f for f in A.facts if f['stale']); print('  OLD:', A.pby[sp_['stale_pid']]['text']); print('  NEW:', A.pby[sp_['gold_pid']]['text'])

class BM25:
    def __init__(self, passages, k1=1.5, b=0.75):
        docs = [p['toks'] for p in passages]; N = len(docs); self.vocab = {}
        for d in docs:
            for t in d: self.vocab.setdefault(t, len(self.vocab))
        df = Counter(t for d in docs for t in set(d)); L = np.array([len(d) for d in docs]); avg = L.mean()
        self.idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items(): self.idf[i] = math.log(1 + (N - df[t] + .5) / (df[t] + .5))
        rows, cols, vals = [], [], []
        for di, d in enumerate(docs):
            for t, f in Counter(d).items():
                rows.append(di); cols.append(self.vocab[t]); vals.append(self.idf[self.vocab[t]] * f * (k1 + 1) / (f + k1 * (1 - b + b * L[di] / avg)))
        self.W = sp.csc_matrix((vals, (rows, cols)), shape=(N, len(self.vocab)))
    def scores(self, q):
        idx = sorted({self.vocab[t] for t in tok(q) if t in self.vocab and t not in STOP})
        return np.asarray(self.W[:, idx].sum(axis=1)).ravel() if idx else np.zeros(self.W.shape[0])

class Retriever:
    def __init__(self, passages, mode='bm25', supersession=False, w_dense=.5, rrf_k=60):
        self.P, self.mode, self.sup, self.w, self.k = passages, mode, supersession, w_dense, rrf_k
        self.bm25 = BM25(passages)
        self.tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True).fit([p['text'] for p in passages])
        self.M = self.tf.transform([p['text'] for p in passages])
        self.superseded = {p['supersedes'] for p in passages if p['supersedes']}
        self.cache = {}
    def search(self, q, k=3):
        key = (q, k)
        if key in self.cache: return self.cache[key]
        sb = self.bm25.scores(q); ob = [i for i in np.argsort(-sb)[:30] if sb[i] > 0]
        if self.mode == 'bm25': ranked = ob
        else:
            sd = (self.M @ self.tf.transform([q]).T).toarray().ravel(); od = [i for i in np.argsort(-sd)[:30] if sd[i] > 0]
            fused = defaultdict(float)
            for r, i in enumerate(ob, 1): fused[i] += (1 - self.w) / (self.k + r)
            for r, i in enumerate(od, 1): fused[i] += self.w / (self.k + r)
            ranked = sorted(fused, key=fused.get, reverse=True)
        pids = [self.P[i]['pid'] for i in ranked]
        if self.sup: pids = [p for p in pids if p not in self.superseded]
        self.cache[key] = pids[:k]; return self.cache[key]

def make_retrievers(I):
    I.R = {'bm25': Retriever(I.passages, 'bm25'), 'bm25sup': Retriever(I.passages, 'bm25', True),
           'hyb':  Retriever(I.passages, 'hybrid'), 'hybsup': Retriever(I.passages, 'hybrid', True)}
make_retrievers(A); make_retrievers(B)

# ---- Experiment 0: retrieval quality on student-style queries (4 noisy queries per fact) ----
def retrieval_quality(I, names, p=.6):
    rows = []
    for name in names:
        h1 = h3 = stale_first = n = 0
        for f in I.facts:
            for q in range(QUERIES_PER_FACT):
                qq = noisy_query(f['field'], f['entity'], random.Random(f"rq-{f['fid']}-{q}"), p)
                pids = I.R[name].search(qq, 3); n += 1
                h1 += bool(pids) and pids[0] == f['gold_pid']; h3 += f['gold_pid'] in pids
                if f['stale_pid'] and f['stale_pid'] in pids and (f['gold_pid'] not in pids or pids.index(f['stale_pid']) < pids.index(f['gold_pid'])): stale_first += 1
        n_stale = sum(1 for f in I.facts if f['stale_pid']) * QUERIES_PER_FACT
        rows.append(dict(retriever=name, **{'gold@1': h1 / n, 'gold@3': h3 / n, 'stale-outranks-gold (% of stale facts)': stale_first / n_stale if n_stale else np.nan}))
    return pd.DataFrame(rows).set_index('retriever')

names = ['bm25', 'hyb', 'hybsup']
rqA, rqB = retrieval_quality(A, names), retrieval_quality(B, names)
print('Institution A'); display(rqA); print('Institution B'); display(rqB)
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for a, (t, d) in zip(ax, [('A', rqA), ('B', rqB)]):
    d[['gold@1', 'gold@3']].rename(index={'bm25': 'BM25', 'hyb': 'Hybrid (RRF)', 'hybsup': 'Hybrid + supersession'}).plot.bar(ax=a, rot=15, ylim=(0, 1.02)); a.set_title(f'Retrieval: institution {t}'); a.set_xlabel('')
plt.tight_layout(); plt.close("all")

TPL = {'A': lambda e, f, v: f"The {f} for {e} is {v}.", 'B': lambda e, f, v: f"For {e}, the {f} is {v}."}
MODE_ID = {'plain': 1, 'rag_v0': 2, 'rag_v1': 3}

def decide_kind(I, f, mode, retr, rng):
    wrong = lambda: 'fabricated' if rng.random() < P_FABRICATE else 'perturbed'
    if mode == 'plain': return 'correct' if rng.random() < PLAIN_ACC[f['cat']] else wrong()
    pids = retr.search(noisy_query(f['field'], f['entity'], rng, .6), TOPK_GEN); pos = {p: i for i, p in enumerate(pids)}
    g, s = f['gold_pid'] in pos, bool(f['stale_pid']) and f['stale_pid'] in pos
    if s and (not g or pos[f['stale_pid']] < pos[f['gold_pid']]) and rng.random() < P_USE_STALE: return 'stale'
    if g: return 'correct' if rng.random() < P_CORRECT_IF_GOLD else wrong()
    return 'correct' if rng.random() < PLAIN_ACC[f['cat']] else wrong()

def make_claim(I, f, kind, rng, cid):
    if kind == 'correct':   ent, fld, v, lab = f['entity'], f['field'], f['truth'], 'true'
    elif kind == 'perturbed': ent, fld, v, lab = f['entity'], f['field'], perturb(f['truth'], rng), 'false'
    elif kind == 'stale':   ent, fld, v, lab = f['entity'], f['field'], f['stale'], 'false'
    else:
        fk, vt = rng.choice(I.fakes[f['cat']]); ent, fld, v, lab = f['entity'], fk, fake_value(vt, rng), 'unverifiable'
    return dict(cid=cid, fid=f['fid'], split=f['split'], cat=f['cat'], entity=ent, field=fld, value=v, label3=lab, kind=kind, text=TPL[I.name](ent, fld, fmt(v)))

def build_claims(I, mode, retr, seed):
    rng = random.Random(seed * 100 + MODE_ID[mode]); out = []
    for q in range(QUERIES_PER_FACT):
        for f in I.facts:
            for g in [f] + rng.sample(I.facts, SIDE_CLAIMS):
                out.append(make_claim(I, g, decide_kind(I, g, mode, retr, rng), rng, f"{I.name}-{mode}-{len(out)}"))
    return out

claims = {m: build_claims(A, m, {'plain': None, 'rag_v0': A.R['bm25'], 'rag_v1': A.R['hybsup']}[m], SEED) for m in MODE_ID}
raw = pd.DataFrame([{**{k: c[k] for k in ('cat', 'label3', 'kind', 'split')}, 'mode': m} for m, cs in claims.items() for c in cs])
raw['hall'] = (raw.label3 != 'true').astype(int)
print('claims per generator:', {m: len(c) for m, c in claims.items()})
tab = raw.pivot_table(index='cat', columns='mode', values='hall', aggfunc='mean')[['plain', 'rag_v0', 'rag_v1']]
tab.loc['ALL'] = raw.groupby('mode').hall.mean()[['plain', 'rag_v0', 'rag_v1']]
print('\nRaw hallucination rate (claim is false or unverifiable), institution A, before any verification'); display(tab)
print('Error composition of RAG v0 claims:'); display(raw[(raw['mode'] == 'rag_v0') & (raw.hall == 1)].kind.value_counts(normalize=True).to_frame('share of errors').T)

RE1 = re.compile(r"^The (?P<field>.+?) for (?P<entity>.+?) is (?P<value>.+?)\.$")
RE2 = re.compile(r"^For (?P<entity>.+?), the (?P<field>.+?) is (?P<value>.+?)\.$")
def extract(answer):
    """STUB claim extractor: sentence split + regex slot filling. Real system: LLM decomposition into atomic claims."""
    out = []
    for s in re.split(r'(?<=\.)\s+(?=[A-Z])', answer.strip()):
        m = RE1.match(s) or RE2.match(s)
        if m: out.append(dict(entity=m['entity'], field=m['field'], value=parse_value(m['value']), text=s))
    return out

def agent_F(I, c):
    k = (norm(c['entity']), norm(c['field']))
    return float('nan') if k not in I.records else float(I.records[k] == c['value'])

def agent_P(I, c):
    e, f = norm(c['entity']), norm(c['field']); app = [r for r in I.rules if r.applies(e, f)]
    if not app: return float('nan')
    def ok(r):
        try: return bool(r.check(c['entity'], c['value']))
        except Exception: return False
    return float(all(ok(r) for r in app))

def agent_D(I, c, cid, retr_name):
    q = noisy_query(c['field'], c['entity'], random.Random(f"g-{cid}"), .25)
    ent, fld = set(tok(c['entity'])) - STOP, set(tok(c['field'])) - STOP; vt = tok(fmt(c['value'])); sup = con = 0.
    for pid in I.R[retr_name].search(q, TOPK_GROUND):
        pt = I.pby[pid]['toks']; ps = set(pt)
        if len(ent & ps) / len(ent) * len(fld & ps) / len(fld) < .99: continue     # passage must be about this entity AND this field
        if contains_seq(pt, vt): sup = 1.
        else: con = 1.
    return max(0., sup - .5 * con)

def featurize(I, claims, d_variants):
    rows, fails = [], 0
    for c in claims:
        ex = extract(c['text'])
        if not ex: fails += 1; continue
        e = ex[0]
        r = dict(cid=c['cid'], fid=c['fid'], split=c['split'], cat=c['cat'], label3=c['label3'], kind=c['kind'], hall=int(c['label3'] != 'true'), F=agent_F(I, e), P=agent_P(I, e))
        for v in d_variants: r['D_' + v] = agent_D(I, e, c['cid'], v)
        rows.append(r)
    df = pd.DataFrame(rows); df.attrs['extract_fail'] = fails; return df

V = ['bm25', 'bm25sup', 'hyb', 'hybsup']
feat_v0 = featurize(A, claims['rag_v0'], V)      # claims produced by the BASELINE generator, scored by every verifier variant (controlled comparison)
feat_v1 = featurize(A, claims['rag_v1'], ['hybsup'])
print('extraction failures:', feat_v0.attrs['extract_fail'], feat_v1.attrs['extract_fail'])
cov = feat_v0.assign(F_app=feat_v0.F.notna(), P_app=feat_v0.P.notna()).groupby('cat')[['F_app', 'P_app']].mean()
cov.loc['ALL'] = feat_v0[['F', 'P']].notna().mean().values
print('\nSignal applicability (share of claims for which the signal can be computed), institution A'); display(cov.rename(columns={'F_app': 'F applicable', 'P_app': 'P applicable'}))

GRID = [(a / 10, b / 10, (10 - a - b) / 10) for a in range(11) for b in range(11 - a)]

def G_masked(F, P, D, w):
    mF, mP, mD = ~np.isnan(F), ~np.isnan(P), ~np.isnan(D)
    num = w[0] * np.where(mF, F, 0) + w[1] * np.where(mP, P, 0) + w[2] * np.where(mD, D, 0)
    den = w[0] * mF + w[1] * mP + w[2] * mD
    return np.where(den > 0, num / np.where(den > 0, den, 1), .5)
def G_naive(F, P, D, w): return w[0] * np.nan_to_num(F) + w[1] * np.nan_to_num(P) + w[2] * np.nan_to_num(D)

def arrays(df, dcol, drop=()):
    F, P, D = df.F.values.copy(), df.P.values.copy(), df[dcol].values.copy() if dcol else np.full(len(df), np.nan)
    if 'F' in drop: F[:] = np.nan
    if 'P' in drop: P[:] = np.nan
    return F, P, D

def fit_weights(dev, dcol, masked=True, drop=()):
    F, P, D = arrays(dev, dcol, drop); y = dev.hall.values; fn = G_masked if masked else G_naive; res = []
    for w in GRID:
        g = fn(F, P, D, w)
        if np.ptp(g) > 0: res.append((roc_auc_score(y, 1 - g), w))
    best = max(a for a, _ in res)
    return min((w for a, w in res if a >= best - .002), key=lambda w: sum((x - 1 / 3) ** 2 for x in w))

def tune_gate(g, y, pass_prec=.97, block_prec=.90, min_n=10):
    cand = np.unique(g); t_pass = cand.max() + 1e-9
    for t in cand:
        s = g >= t
        if s.sum() >= min_n and 1 - y[s].mean() >= pass_prec: t_pass = t; break
    t_block = cand.min()
    for t in cand[::-1]:
        if t > t_pass: continue
        s = g < t
        if s.sum() >= min_n and y[s].mean() >= block_prec: t_block = t; break
    return t_pass, min(t_block, t_pass)

def gate_report(df, g, t_pass, t_block):
    y = df.hall.values; d = np.where(g >= t_pass, 'PASS', np.where(g < t_block, 'BLOCK', 'FLAG')); P_ = d == 'PASS'
    return dict(n=len(df), raw_halluc=y.mean(), pass_halluc=y[P_].mean() if P_.any() else np.nan, pass_share=P_.mean(), flag_share=(d == 'FLAG').mean(), block_share=(d == 'BLOCK').mean(),
                true_retained=P_[y == 0].mean(), halluc_leak=P_[y == 1].mean(), decisions=d)

def boot_auc_ci(y, s, fids, n=300, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(fids); idx = {f: np.where(fids == f)[0] for f in u}; out = []
    for _ in range(n):
        ii = np.concatenate([idx[f] for f in rng.choice(u, len(u))])
        if y[ii].min() != y[ii].max(): out.append(roc_auc_score(y[ii], s[ii]))
    return np.percentile(out, [2.5, 97.5])

def evaluate(feat, dcol, masked=True, drop=(), w=None, thr=None):
    """fit on dev (unless w/thr given), report on test"""
    dev, test = feat[feat.split == 'dev'], feat[feat.split == 'test']
    fn = G_masked if masked else G_naive
    w = w or fit_weights(dev, dcol, masked, drop)
    gd = fn(*arrays(dev, dcol, drop), w); gt = fn(*arrays(test, dcol, drop), w)
    thr = thr or tune_gate(gd, dev.hall.values)
    y = test.hall.values; s = 1 - gt
    out = dict(w=w, thr=thr, g_test=gt, test=test, auc=roc_auc_score(y, s), ap=average_precision_score(y, s), auc_ci=boot_auc_ci(y, s, test.fid.values), **gate_report(test, gt, *thr))
    return out

S0 = evaluate(feat_v0, 'D_bm25',  masked=True, drop=('F', 'P'))                         # Arch-0 verifier: grounding only, BM25 evidence
S1 = evaluate(feat_v0, 'D_hybsup', masked=True, drop=('F', 'P'))                        # + hybrid retrieval & supersession (still D only)
S2 = evaluate(feat_v0, 'D_hybsup', masked=True)                                         # Arch-1 verifier: F + P + D, masked, fitted weights
sys_ = {'Arch-0: D only (BM25)': S0, '+ hybrid & supersession (D only)': S1, 'Arch-1: F+P+D masked': S2}
rows = [dict(system=k, AUC=v['auc'], AUC_95CI=f"[{v['auc_ci'][0]:.3f}, {v['auc_ci'][1]:.3f}]", AP=v['ap'], weights_FPD=v['w'] if k.startswith('Arch-1') else '-') for k, v in sys_.items()]
print(f"test claims: {S2['n']}  | raw hallucination rate: {S2['raw_halluc']:.3f}  | CI = cluster bootstrap over facts")
display(pd.DataFrame(rows).set_index('system'))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); col = ['#8896ab', '#e0872b', '#1f4e8c']
for (k, v), c in zip(sys_.items(), col):
    fpr, tpr, _ = roc_curve(v['test'].hall.values, 1 - v['g_test']); ax[0].plot(fpr, tpr, color=c, label=f"{k} (AUC {v['auc']:.3f})")
ax[0].plot([0, 1], [0, 1], '--', color='grey', lw=.8); ax[0].set_xlabel('False positive rate'); ax[0].set_ylabel('True positive rate'); ax[0].set_title('ROC - hallucination detection'); ax[0].legend(fontsize=7, loc='lower right')
for k, c in zip(['Arch-0: D only (BM25)', 'Arch-1: F+P+D masked'], [col[0], col[2]]):
    v = sys_[k]; pr_y, pr_s = v['test'].hall.values, 1 - v['g_test']
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(pr_y, pr_s); ax[1].plot(r, p, color=c, label=f"{k} (AP {v['ap']:.3f})")
ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_title('Precision-recall'); ax[1].legend(fontsize=7, loc='lower left'); plt.tight_layout(); plt.close("all")

# where do the errors of each verifier come from? (hallucinated claims that PASS the gate, by error kind)
leak = []
for k, v in sys_.items():
    t = v['test'].assign(dec=v['decisions']); h = t[t.hall == 1]
    leak.append(pd.Series({kind: ((h.kind == kind) & (h.dec == 'PASS')).sum() for kind in ['perturbed', 'stale', 'fabricated']}, name=k))
print('Hallucinated test claims that still reach PASS, by error kind (count)'); display(pd.DataFrame(leak))

E0 = evaluate(feat_v0, 'D_bm25', masked=True, drop=('F', 'P'))
E1 = evaluate(feat_v1, 'D_hybsup', masked=True)
plain_h = raw[(raw['mode'] == 'plain') & (raw.split == 'test')].hall.mean(); v0_h = E0['raw_halluc']; v1_h = E1['raw_halluc']
e2 = pd.DataFrame([
    dict(system='Plain LLM (no retrieval, no verification)', output_halluc=plain_h, claims_kept_PASS='100%', flagged='-', blocked='-', true_claims_retained='100%', halluc_reaching_PASS='100%'),
    dict(system='RAG-only (BM25)',                           output_halluc=v0_h,    claims_kept_PASS='100%', flagged='-', blocked='-', true_claims_retained='100%', halluc_reaching_PASS='100%'),
    dict(system='Arch-0 (BM25 RAG + grounding gate)',        output_halluc=E0['pass_halluc'], claims_kept_PASS=f"{E0['pass_share']:.1%}", flagged=f"{E0['flag_share']:.1%}", blocked=f"{E0['block_share']:.1%}", true_claims_retained=f"{E0['true_retained']:.1%}", halluc_reaching_PASS=f"{E0['halluc_leak']:.1%}"),
    dict(system='Arch-1 (hybrid+supersession RAG + F/P/D gate)', output_halluc=E1['pass_halluc'], claims_kept_PASS=f"{E1['pass_share']:.1%}", flagged=f"{E1['flag_share']:.1%}", blocked=f"{E1['block_share']:.1%}", true_claims_retained=f"{E1['true_retained']:.1%}", halluc_reaching_PASS=f"{E1['halluc_leak']:.1%}")]).set_index('system')
print('output_halluc = hallucination rate among claims the user sees as verified (PASS); for the first two rows every claim is shown.')
print(f"Arch-1 weights (F,P,D) = {E1['w']} | thresholds (t_pass, t_block) = ({E1['thr'][0]:.2f}, {E1['thr'][1]:.2f})   Arch-0 thresholds = ({E0['thr'][0]:.2f}, {E0['thr'][1]:.2f})")
display(e2)

# per-category chart (mirrors the plot planned in the proposal)
def pass_rate_by_cat(E): t = E['test'].assign(dec=E['decisions']); t = t[t.dec == 'PASS']; return t.groupby('cat').hall.mean()
cats = ['exam_dates', 'fee_deadlines', 'hostel_rules', 'circulars', 'syllabus']
bar = pd.DataFrame({'Plain LLM': raw[(raw['mode'] == 'plain') & (raw.split == 'test')].groupby('cat').hall.mean(),
                    'RAG-only': raw[(raw['mode'] == 'rag_v0') & (raw.split == 'test')].groupby('cat').hall.mean(),
                    'Arch-0 (PASS)': pass_rate_by_cat(E0), 'Arch-1 (PASS)': pass_rate_by_cat(E1)}).loc[cats] * 100
ax = bar.plot.bar(figsize=(8.5, 3.5), rot=0, color=['#a5b0c2', '#e0872b', '#6c8ebf', '#1f4e8c']); ax.set_ylabel('Hallucination rate (%)'); ax.set_title('Hallucination rate in what the user sees, by category (institution A, test)'); ax.legend(fontsize=7); plt.tight_layout(); plt.close("all")

# confusion matrices: PASS = predicted grounded, FLAG/BLOCK = predicted hallucinated
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
for a, (k, E) in zip(ax, [('Arch-0', E0), ('Arch-1', E1)]):
    y = E['test'].hall.values; pr = (E['decisions'] != 'PASS').astype(int); cm = np.array([[((y == i) & (pr == j)).sum() for j in (0, 1)] for i in (0, 1)])
    a.imshow(cm, cmap='Blues'); a.grid(False); a.set_xticks([0, 1], ['PASS', 'FLAG/BLOCK']); a.set_yticks([0, 1], ['grounded', 'hallucinated']); a.set_title(f'{k} - actual vs gate'); a.set_xlabel('gate decision')
    for i in (0, 1):
        for j in (0, 1): a.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=11)
plt.tight_layout(); plt.close("all")

abl = {'Full: F+P+D (hybrid+supersession)': dict(dcol='D_hybsup'),
       '- Fact-Check (F)':                  dict(dcol='D_hybsup', drop=('F',)),
       '- Policy (P)':                      dict(dcol='D_hybsup', drop=('P',)),
       '- Grounding (F+P only)':            dict(dcol=None),
       '- supersession filter':             dict(dcol='D_hyb'),
       '- dense/hybrid (BM25 grounding)':   dict(dcol='D_bm25sup'),
       '- both (BM25, no supersession)':    dict(dcol='D_bm25')}
rows = []
for k, kw in abl.items():
    r = evaluate(feat_v0, **kw); rows.append(dict(variant=k, AUC=r['auc'], AUC_lo=r['auc_ci'][0], AUC_hi=r['auc_ci'][1], AP=r['ap'], PASS_halluc=r['pass_halluc'], true_retained=r['true_retained']))
abl_df = pd.DataFrame(rows).set_index('variant'); display(abl_df)
fig, ax = plt.subplots(figsize=(7.5, 3.3)); y = np.arange(len(abl_df))[::-1]
ax.barh(y, abl_df.AUC, xerr=[abl_df.AUC - abl_df.AUC_lo, abl_df.AUC_hi - abl_df.AUC], color=['#1f4e8c'] + ['#e0872b'] * (len(abl_df) - 1), capsize=2)
ax.set_yticks(y, abl_df.index); ax.set_xlim(max(.5, abl_df.AUC_lo.min() - .03), 1.005); ax.set_xlabel('Test ROC-AUC (95% cluster-bootstrap CI)'); ax.set_title('Ablation'); plt.tight_layout(); plt.close("all")

claims_B = build_claims(B, 'rag_v1', B.R['hybsup'], SEED)
fB = featurize(B, claims_B, ['hybsup']); print('B extraction failures:', fB.attrs['extract_fail'], '| claims:', len(fB), '| raw halluc rate:', round(fB.hall.mean(), 3))
covB = fB.assign(F_app=fB.F.notna(), P_app=fB.P.notna()).groupby('cat')[['F_app', 'P_app']].mean(); covB.loc['ALL'] = fB[['F', 'P']].notna().mean().values
print('Signal applicability, institution B'); display(covB.rename(columns={'F_app': 'F applicable', 'P_app': 'P applicable'}))

# weights/thresholds from A (Arch-1 setting) ----------------------------------------------------------
wA, thrA = E1['w'], E1['thr']
A_dev = feat_v1[feat_v1.split == 'dev']
wA_naive = fit_weights(A_dev, 'D_hybsup', masked=False); gA_naive = G_naive(*arrays(A_dev, 'D_hybsup'), wA_naive); thrA_naive = tune_gate(gA_naive, A_dev.hall.values)

tests = {'A-fitted weights + thresholds, masked (no re-tune)': evaluate(fB, 'D_hybsup', True, w=wA, thr=thrA),
         'A-fitted, naive missing-as-0 (no re-tune)':          evaluate(fB, 'D_hybsup', False, w=wA_naive, thr=thrA_naive),
         'Grounding only (D)':                                 evaluate(fB, 'D_hybsup', True, drop=('F', 'P')),
         'Oracle: re-fitted on B dev (masked)':                evaluate(fB, 'D_hybsup', True)}
rows = [dict(setting=k, weights_FPD=v['w'], AUC=v['auc'], AUC_95CI=f"[{v['auc_ci'][0]:.3f}, {v['auc_ci'][1]:.3f}]", AP=v['ap'], PASS_halluc=v['pass_halluc'], PASS_share=v['pass_share'], true_retained=v['true_retained']) for k, v in tests.items()]
print('Institution B, test split (raw hallucination rate %.3f)' % tests['Oracle: re-fitted on B dev (masked)']['raw_halluc']); display(pd.DataFrame(rows).set_index('setting'))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.3)); labs = ['A-weights\nmasked', 'A-weights\nnaive', 'D only', 'Oracle\n(B-tuned)']
ax[0].bar(labs, [v['auc'] for v in tests.values()], color=['#1f4e8c', '#c0504d', '#a5b0c2', '#6c8ebf']); ax[0].set_ylim(.5, 1.0); ax[0].set_title('B test: ROC-AUC'); ax[0].set_ylabel('AUC')
ax[1].bar(labs, [v['true_retained'] for v in tests.values()], color=['#1f4e8c', '#c0504d', '#a5b0c2', '#6c8ebf']); ax[1].set_title('B test: share of true claims reaching PASS'); ax[1].set_ylim(0, 1)
for a, k in zip(ax, ['auc', 'true_retained']):
    for i, v in enumerate(tests.values()): a.text(i, v[k] + .01, f"{v[k]:.2f}", ha='center', fontsize=8)
plt.tight_layout(); plt.close("all")

rows = []
for sd in [11, 12, 13, 14, 15]:
    c0 = build_claims(A, 'rag_v0', A.R['bm25'], sd);  c1 = build_claims(A, 'rag_v1', A.R['hybsup'], sd)
    f0 = featurize(A, c0, ['bm25']); f1 = featurize(A, c1, ['hybsup'])
    a0 = evaluate(f0, 'D_bm25', True, drop=('F', 'P')); a1 = evaluate(f1, 'D_hybsup', True)
    rows.append(dict(seed=sd, raw_halluc_v0=a0['raw_halluc'], raw_halluc_v1=a1['raw_halluc'], AUC_arch0=a0['auc'], AUC_arch1=a1['auc'],
                     PASS_halluc_arch0=a0['pass_halluc'], PASS_halluc_arch1=a1['pass_halluc'], retained_arch0=a0['true_retained'], retained_arch1=a1['true_retained']))
sd_df = pd.DataFrame(rows).set_index('seed'); display(sd_df); print('mean +- std'); display(pd.DataFrame({'mean': sd_df.mean(), 'std': sd_df.std()}).T)

