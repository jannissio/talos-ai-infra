import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.collect_compact_bottle import main,validate_specs
from scripts.plan_bottle_coverage import protocol


class CompactProtocolTests(unittest.TestCase):
    def test_declared_splits_do_not_leak_into_training(self):
        plan=protocol()
        validate_specs(plan['training'])
        seeds=[{s['seed'] for s in plan[split]} for split in ('training','development','evaluation')]
        self.assertEqual([len(x) for x in seeds],[24,6,12])
        self.assertFalse(seeds[0]&seeds[1] or seeds[0]&seeds[2] or seeds[1]&seeds[2])
        with self.assertRaises(ValueError):validate_specs(plan['evaluation'])
        altered=copy.deepcopy(plan['training']);altered[0]['split']='evaluation'
        with self.assertRaises(ValueError):validate_specs(altered)

    def test_rejects_path_traversal_duplicates_and_nonfinite_poses(self):
        for field,value in [('id','../models/bottle_visual'),('seed',True)]:
            specs=protocol()['training'];specs[0][field]=value
            with self.assertRaises(ValueError):validate_specs(specs)
        specs=protocol()['training'];specs[0]['pose']['x']=float('nan')
        with self.assertRaises(ValueError):validate_specs(specs)
        specs=protocol()['training'];specs[1]['id']=specs[0]['id']
        with self.assertRaises(ValueError):validate_specs(specs)

    def test_changed_protocol_cannot_resume_over_existing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch=root/'batch';batch.mkdir()
            frozen=protocol();path=root/'input.json';path.write_text(json.dumps(frozen))
            saved=copy.deepcopy(frozen['training']);saved[0]['seed']+=1
            existing=json.dumps({'specs':saved});(batch/'protocol.json').write_text(existing)
            args=SimpleNamespace(output=str(batch),protocol=str(path),count=32,resume=True,limit=3)
            with patch('scripts.collect_compact_bottle.trial') as collect:
                with self.assertRaises(ValueError):main(args)
                collect.assert_not_called()
            self.assertEqual((batch/'protocol.json').read_text(),existing)

    def test_resume_retains_failed_attempt_and_respects_new_attempt_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch=root/'batch';batch.mkdir()
            specs=protocol()['training'][:3];source=root/'input.json';source.write_text(json.dumps({'training':specs}))
            (batch/'protocol.json').write_text(json.dumps({'specs':specs}))
            old=batch/specs[0]['id'];old.mkdir()
            manifest={'spec':specs[0],'training_eligible':False,'outcome':{'status':'failed','message':'Unreachable'},'replays':[]}
            saved=json.dumps(manifest);(old/'manifest.json').write_text(saved)
            outcome={'id':specs[1]['id'],'seed':specs[1]['seed'],'training_eligible':True,'status':'succeeded','message':'Passed','replays':[]}
            args=SimpleNamespace(output=str(batch),protocol=str(source),count=32,resume=True,limit=1)
            fake_torch=SimpleNamespace(cuda=SimpleNamespace(init=lambda:None))
            with patch.dict('sys.modules',{'torch':fake_torch}),patch('scripts.collect_compact_bottle.trial',return_value=outcome) as collect:
                main(args)
                collect.assert_called_once_with(specs[1],batch)
            summary=json.loads((batch/'summary.json').read_text())
            self.assertEqual(summary['planned'],3)
            self.assertEqual([r['status'] for r in summary['episodes']],['failed','succeeded'])
            self.assertEqual(summary['eligible'],1)
            self.assertEqual((old/'manifest.json').read_text(),saved)


if __name__=='__main__':unittest.main()
