"""Read-only allocation sidecar; never rewrite frozen timing outputs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quota(receipt):
    found=[]
    if '/sys/fs/cgroup/cpu.max' in receipt:
        amount,interval=receipt['/sys/fs/cgroup/cpu.max'].split()
        period=int(interval)
        if period<=0:
            raise ValueError('Invalid cgroup v2 quota period')
        found.append(dict(version=2,quota_us=None if amount=='max' else int(amount),period_us=period))
    for key,amount in receipt.items():
        if key.endswith('/cpu.cfs_quota_us'):
            period_key=key.removesuffix('cpu.cfs_quota_us')+'cpu.cfs_period_us'
            period=int(receipt[period_key])
            if period<=0:
                raise ValueError('Invalid cgroup v1 quota period')
            amount=int(amount)
            found.append(dict(version=1,quota_us=None if amount==-1 else amount,period_us=period))
    if not found:
        raise ValueError('No cgroup quota receipt')
    for item in found:
        if item['quota_us'] is not None and item['quota_us']<=0:
            raise ValueError('Invalid CPU quota')
        item['cpu_quota_cores']=None if item['quota_us'] is None else item['quota_us']/item['period_us']
    if len({item['cpu_quota_cores'] for item in found})!=1:
        raise ValueError('Conflicting cgroup quota receipts')
    return found[0]


def build(evidence,analysis_path,allocation_path):
    evidence=Path(evidence)
    paths={key:evidence/name for key,name in
           [('cpu_quota','cpu-quota.json'),('cpu','cpu.txt'),('gpu','gpu.xml'),('runtime','runtime.json')]}
    allocation=json.loads(Path(allocation_path).read_text())
    analysis=json.loads(Path(analysis_path).read_text())
    actual=quota(json.loads(paths['cpu_quota'].read_text()))
    cpu=dict(line.split(':',1) for line in paths['cpu'].read_text().splitlines() if ':' in line)
    model=cpu['Model name'].strip()
    gpu=ET.parse(paths['gpu']).getroot().findall('gpu')
    if len(gpu)!=1:
        raise ValueError('Expected one allocated GPU')
    gpu_name=gpu[0].findtext('product_name')
    gpu_uuid=gpu[0].findtext('uuid')
    environment=analysis['environment']
    timed_gpu=next(csv.reader(io.StringIO(environment['nvidia_smi'])))
    declared_gpu=allocation.get('gpu',allocation.get('gpu_type',allocation.get('requested_gpu')))
    if (timed_gpu[0].strip()!=gpu_name or timed_gpu[1].strip()!=gpu_uuid or
            declared_gpu!=gpu_name or allocation.get('cpu_model',model)!=model):
        raise ValueError('Timing and actual hardware allocation receipts differ')
    if allocation.get('cpu_quota_cores',actual['cpu_quota_cores'])!=actual['cpu_quota_cores']:
        raise ValueError('Declared CPU allocation differs from the raw cgroup quota')
    if allocation.get('cpu_quota_source_sha256',sha(paths['cpu_quota']))!=sha(paths['cpu_quota']):
        raise ValueError('Allocation identifies a different raw CPU quota receipt')
    recorded=environment.get('cpu_quota_cores')
    if recorded is not None and recorded!=actual['cpu_quota_cores']:
        raise ValueError('Timing quota contradicts the raw allocation')
    return dict(schema_version=1,scope='Separate factual allocation sidecar; frozen timings and their hashes remain unchanged',
        gpu=gpu_name,gpu_uuid=gpu_uuid,cpu_model=model,cpu_quota_cores=actual['cpu_quota_cores'],
        cgroup=actual,original_timing_cpu_quota_cores=recorded,
        allocation_note='The frozen environment reader only parses cgroup v2; this sidecar also reads v1. Host-visible CPU count is not the quota.',
        numerical_threads=json.loads(paths['runtime'].read_text())['threads'],
        compute_usd_per_hour=allocation.get('compute_usd_per_hour',allocation.get('rate_usd_per_hour')),
        sources={key:dict(file=path.name,sha256=sha(path)) for key,path in
                 dict(paths,original_timing_analysis=Path(analysis_path),actual_allocation=Path(allocation_path)).items()},
        measurements_modified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','analysis','allocation','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new allocation sidecar path')
    result=build(args.evidence,args.analysis,args.allocation)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
