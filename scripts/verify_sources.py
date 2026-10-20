"""Verify original installation, overlays, weights, and frozen online sources."""
import argparse
import json
import time
from apb.common import DATA, MODEL, INSTALLED, ROOT, digest, dump
from scripts.prepare import patch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weights',action='store_true')
    args=parser.parse_args()
    prepared=json.loads((DATA/'manifests/prepared.json').read_text())
    checks={}
    for rel,record in prepared['source'].items():
        checks['original:'+rel]=digest(record['path'])==record['sha256']
        checks['stock:'+rel]=digest(DATA/'overlays/stock/vllm'/rel)==record['sha256']
        if rel!='v1/core/sched/scheduler.py':
            checks['candidate:'+rel]=digest(DATA/'overlays/candidate/vllm'/rel)==record['sha256']
    rel='v1/core/sched/scheduler.py'
    checks['candidate:patched-scheduler']=(DATA/'overlays/candidate/vllm'/rel).read_text()==patch((INSTALLED/rel).read_text())
    frozen=json.loads((DATA/'manifests/formal-source.json').read_text())
    for path,sha in frozen['hashes'].items():checks['frozen:'+path]=digest(path)==sha
    if args.weights:
        for name,record in prepared['model_files'].items():
            checks['model:'+name]=digest(MODEL/name)==record['sha256']
    result=dict(checked_unix=time.time(),weights_verified=args.weights,checks=checks,
                passed=all(checks.values()))
    dump(DATA/'manifests/final-source-verification.json',result)
    print('SOURCE_VERIFICATION',len(checks),result['passed'])
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
