"""Real CPU embedding benchmark on a small, explicit synthetic evaluation set."""
import json
import platform
import re
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.embeddings import EmbeddingEngine
from embedding_model import MODEL_ID

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/'tests/fixtures/retrieval_eval.json').read_text())
engine = EmbeddingEngine()
start = time.monotonic(); engine.initialize(allow_download=False)
loaded = time.monotonic()
vectors = np.array(engine.encode(data['documents']))
indexed = time.monotonic()
results, latencies = [], []
stop = {'the','this','that','with','from','what','does','explain','please','about','document','and','for','are','how'}
for case in data['queries']:
    began = time.monotonic(); query = np.array(engine.encode([case['text']])[0]); latencies.append(time.monotonic()-began)
    terms = [t for t in re.split(r'[^\w.-]+',case['text'].lower()) if len(t)>2 and t not in stop]
    lexical = [sum(4 if any(c.isdigit() for c in t) else 1 for t in terms if t in doc.lower()) for doc in data['documents']]
    scores = vectors @ query
    lex_order = sorted(range(len(vectors)), key=lambda i: (-lexical[i], i))
    sem_order = sorted(range(len(vectors)), key=lambda i: (-scores[i], i))
    identifiers = [t for t in terms if re.fullmatch('[a-z][a-z0-9_.-]*',t) and re.search('[0-9]',t)]
    exact = [sum(bool(re.search(r'(?<![\w.-])'+re.escape(t)+r'(?![\w.-])',doc.lower())) for t in identifiers) for doc in data['documents']]
    candidates=[i for i in range(len(vectors)) if (exact[i]>0 if identifiers else scores[i]>=0.2)]
    hybrid = sorted(candidates, key=lambda i: (-exact[i], -(scores[i] + 0.02*lexical[i]/max(1,max(lexical))), i))
    results.append({**case, 'lexicalTop1':lex_order[0], 'semanticTop1':sem_order[0], 'hybridTop1':hybrid[0], 'hybridRecall3':case['expected'] in hybrid[:3]})
metrics = {}
for kind in ('exact','paraphrase'):
    cases = [c for c in results if c['kind']==kind]
    metrics[kind] = {method:sum(c[method]==c['expected'] for c in cases)/len(cases) for method in ('lexicalTop1','semanticTop1','hybridTop1')}
    metrics[kind]['hybridRecall3'] = sum(c['hybridRecall3'] for c in cases)/len(cases)
report = {'modelId':MODEL_ID, 'platform':platform.platform(), 'loadSeconds':loaded-start, 'index20Seconds':indexed-loaded,
          'queryP95Milliseconds':float(np.percentile(latencies,95)*1000), 'metrics':metrics, 'cases':results}
long_text='A reference circuit uses a resistor, capacitor, microcontroller and a serial interface. ' * 20
began=time.monotonic(); engine.encode([long_text]*32)
report['longBatch32Seconds']=time.monotonic()-began
if sys.platform=='win32':
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    memory=Memory(); memory.cb=ctypes.sizeof(memory)
    if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(memory),memory.cb):
        report['peakProcessMemoryMiB']=round(memory.PeakWorkingSetSize/1024**2,2)
# Persist actual vectors for a Java parity test of the production fusion algorithm.
(ROOT/'docs/AIR-05_BENCHMARK.json').write_text(json.dumps(report,indent=2)+'\n')
fixture={'documents':data['documents'],'vectors':vectors.round(7).tolist(),
    'queries':[{**c,'vector':np.array(engine.encode([c['text']])[0]).round(7).tolist()} for c in data['queries']],
    'negativeQueries':[{'text':q,'vector':np.array(engine.encode([q])[0]).round(7).tolist()} for q in data.get('negative_queries',[])]}
(ROOT/'.bootstrap').mkdir(exist_ok=True)
(ROOT/'.bootstrap/retrieval-vectors.json').write_text(json.dumps(fixture)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
engine.session.close()
