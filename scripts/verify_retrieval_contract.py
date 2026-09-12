"""Exercise corrupt observation and local-demo request handling against a fitted model."""
import argparse,json,sys
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from simulation_lab.retrieval_policy import RetrievalPolicy,ObservationRejected,KEYS

def main(a):
    torch.set_num_threads(2);checks={}
    policy=RetrievalPolicy(a.checkpoint)
    for fault in ['blank','nonfinite_joint','nonfinite_camera','wrong_camera_shape']:
        batch={'observation.state':torch.zeros(1,24),**{k:torch.zeros(1,3,240,320) for k in KEYS}}
        if fault=='nonfinite_joint':batch['observation.state'][0,0]=float('nan')
        if fault=='nonfinite_camera':batch[KEYS[0]][0,0,0,0]=float('inf')
        if fault=='wrong_camera_shape':batch[KEYS[0]]=torch.zeros(1,3,120,160)
        try:policy.predict_action_chunk(batch)
        except ObservationRejected as exc:checks[fault]={'rejected':True,'reason':str(exc)}
        else:raise AssertionError(fault+' was not rejected')
        assert not policy.trace and policy.cursor is None
    for name,payload,origin,expected in [
        ('unknown_start',{'episode':'not-a-supported-scene'},None,400),
        ('cross_origin',{'episode':'upright-01'},'https://example.invalid',403)]:
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        request=Request(a.url+'/run',data=json.dumps(payload).encode(),headers=headers,method='POST')
        try:
            with urlopen(request,timeout=5) as response:code=response.status
        except HTTPError as exc:code=exc.code
        assert code==expected,(name,code)
        checks[name]={'http_status':code,'expected':expected}
    with urlopen(a.url+'/state',timeout=5) as response:state=json.load(response)
    assert not state['running'],'Request rejection unexpectedly started a job.'
    checks['rejected_requests_did_not_start_job']=True
    Path(a.output).write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--url',default='http://127.0.0.1:8766');p.add_argument('--output',required=True);main(p.parse_args())
