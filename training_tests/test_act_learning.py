"""Learning boundary and numerical normalization checks (training environment)."""
import unittest
import numpy as np
import torch
from simulation_lab.act_learning import IMAGE_KEYS,normalization,normalize,denormalize

class LearningBoundary(unittest.TestCase):
    def test_normalization_filters_privileged_inputs_and_round_trips(self):
        stats=normalization({k:{'mean':np.zeros(n),'std':np.zeros(n)} for k,n in [('observation.state',24),('action',12)]})
        source={'observation.state':torch.zeros(1,24),'action':torch.ones(1,20,12),
                'bottle_pose':torch.ones(1,7),'teacher_stage':'hold','contacts':'privileged',
                **{k:torch.zeros(1,3,240,320) for k in IMAGE_KEYS}}
        result=normalize(source,stats,'cpu')
        self.assertEqual(set(result),{'observation.state','action',*IMAGE_KEYS})
        torch.testing.assert_close(denormalize(result['action'],stats),source['action'])
        self.assertTrue(all(torch.isfinite(x).all() for x in result.values()))

    def test_observation_camera_and_proprioception_only(self):
        from scripts.evaluate_act import observation
        class Data:
            qpos=np.arange(30,dtype=float)
            qvel=np.arange(29,dtype=float)
        class Scene:flags=np.ones(20)
        class Renderer:
            scene=Scene()
            def update_scene(self,*a,**kw):pass
            def render(self):return np.full((240,320,3),128,dtype=np.uint8)
        data=Data();obs=observation(None,data,Renderer(),None)
        self.assertEqual(set(obs),{'observation.state',*IMAGE_KEYS})
        self.assertEqual(tuple(obs['observation.state'].shape),(1,24))
        self.assertEqual(float(obs['observation.state'][0,11]),11)
        self.assertTrue(all(tuple(obs[k].shape)==(1,3,240,320) for k in IMAGE_KEYS))
        self.assertTrue(all(torch.count_nonzero(v)==0 for k,v in observation(None,data,Renderer(),None,True).items() if k in IMAGE_KEYS))

if __name__=='__main__':unittest.main()
