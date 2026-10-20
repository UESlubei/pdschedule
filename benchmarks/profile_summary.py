"""Read-only SQLite aggregation; graph durations and exposed kernels stay separate."""
import json
import sqlite3
from apb.common import DATA,ROOT,dump,stats


def extract(path,granularity,policy):
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as connection:
        tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        result=dict(path=str(path),granularity=granularity,policy=policy)
        for table,label in [('CUPTI_ACTIVITY_KIND_GRAPH_TRACE','gpu_graph_ms'),('CUPTI_ACTIVITY_KIND_KERNEL','exposed_kernel_ms')]:
            result[label]=stats((r[0]/1e6 for r in connection.execute(f'SELECT end-start FROM "{table}"')) if table in tables else [])
        result['graph_internal_kernel_count']=connection.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL AND graphNodeId!=0').fetchone()[0]
        if granularity=='node':
            query='SELECT s.value,COUNT(*),SUM(k.end-k.start)/1000000.0 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC'
            names=[dict(name=name,count=count,duration_ms=duration) for name,count,duration in connection.execute(query)]
            result['top_kernels']=names[:20];categories={}
            for row in names:
                name=row['name'].lower()
                label=('Attention' if 'flash' in name or 'attention' in name else 'GEMM' if any(x in name for x in ['gemm','matmul','s16816']) else
                       'Norm' if 'rms' in name or 'layer_norm' in name else 'RoPE' if 'rotary' in name or 'rope' in name else
                       'Activation' if 'silu' in name or 'gelu' in name else 'Sampling/softmax' if 'argmax' in name or 'softmax' in name or 'sample' in name else 'Other')
                target=categories.setdefault(label,dict(count=0,duration_ms=0.));target['count']+=row['count'];target['duration_ms']+=row['duration_ms']
            result['name_based_categories']=categories
        return result


def main():
    receipts=json.loads((DATA/'profiles/receipt.json').read_text())
    assert len(receipts)==5 and all(r['returncode']==0 and r['kernel_count']>0 for r in receipts)
    rows=[extract(DATA/'profiles'/f"{r['policy']}.sqlite",'graph',r['policy']) for r in receipts]
    node=json.loads((DATA/'profiles_node/receipt.json').read_text());assert node['complete']
    rows.append(extract(DATA/'profiles_node/joint-node.sqlite','node','joint'))
    extra=DATA/'profiles_node/observe/receipt.json'
    if extra.exists() and json.loads(extra.read_text()).get('complete'):
        rows.append(extract(DATA/'profiles_node/observe/observe-node.sqlite','node','observe'))
    dump(ROOT/'reports/profile_summary.json',dict(captures=rows,
         interpretation='Graph-mode kernel tables exclude graph-internal nodes. Node-mode categories are heuristic function-name groups; summed activity time is not wall time or utilization. Captures cover 128 completion callbacks with possible in-flight edge work; no profile run is performance evidence.'))
    print('PROFILE_SUMMARIES',len(rows))


if __name__=='__main__':main()
