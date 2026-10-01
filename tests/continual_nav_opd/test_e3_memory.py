"""Forward-local caches preserve gradients; decoded maps remove reset disk reads."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from tasks.continual_nav_opd.config import load_config,raw_config_hash,parse_config,resolved_dict
from tasks.continual_nav_opd.models import StandalonePolicy,DualPolicy,save_snapshot,load_snapshot
from tasks.continual_nav_opd.models.policy import _sequence
from tasks.continual_nav_opd.envs import MazeEnv,MazeMapCache
from tasks.continual_nav_opd.envs.map_cache import shared_map_cache
from tasks.continual_nav_opd.evaluate import create_panel_env
from tasks.continual_nav_opd.data import MazeEntry
from tests.continual_nav_opd.test_teachers_envs import maze_image


class CacheTests(unittest.TestCase):
    def test_optional_runtime_config_and_old_inference_hash(self):
        config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        raw=resolved_dict(config)
        del raw['environment']['map_cache']
        del raw['optimization']['e3_cache'],raw['optimization']['ctm_compile']
        legacy=parse_config(raw)
        self.assertFalse(legacy.optimization.e3_cache)
        self.assertEqual(legacy.environment.map_cache,'disk')
        old=torch.get_num_threads(); torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'old.pt'; policy=StandalonePolicy(legacy)
                save_snapshot(policy,path)
                artifact=torch.load(path,weights_only=True)
                artifact.update(config=raw,config_hash=raw_config_hash(raw))
                torch.save(artifact,path)
                restored=load_snapshot(path)
                for key,value in policy.state_dict().items():
                    torch.testing.assert_close(value,restored.state_dict()[key],atol=0,rtol=0)
                artifact['config']['environment']['map_cache']='memory'
                torch.save(artifact,path)
                with self.assertRaisesRegex(ValueError,'hash mismatch'): load_snapshot(path)
        finally: torch.set_num_threads(old)

    def test_sequence_prepares_static_work_once(self):
        old=torch.get_num_threads(); torch.set_num_threads(2)
        try:
            policy=StandalonePolicy(load_config('tasks/continual_nav_opd/configs/smoke.yaml'))
            controller=policy.controller; rgb=torch.zeros(3,2,3,84,84,dtype=torch.uint8)
            with patch.object(controller.attention,'prepare',wraps=controller.attention.prepare) as keys, \
                 patch.object(controller.action_sync,'prepare',wraps=controller.action_sync.prepare) as action, \
                 patch.object(controller.out_sync,'prepare',wraps=controller.out_sync.prepare) as output:
                _sequence(policy,rgb,policy.initial_state(2),torch.zeros(3,2,dtype=torch.bool),torch.ones(3,2,dtype=torch.bool))
                self.assertEqual(keys.call_count,3)
                self.assertEqual(action.call_count,1)
                self.assertEqual(output.call_count,1)
        finally: torch.set_num_threads(old)

    def test_maps_no_warm_reads_and_same_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); entries=[]
            for i in range(2):
                path=root/f'train/0/{i}.png'; path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(maze_image(goal=(1+i,2))).save(path)
                entries.append(MazeEntry(path.relative_to(root).as_posix(),hashlib.sha256(path.read_bytes()).hexdigest()))
            cache=shared_map_cache(root,entries)
            self.assertFalse(cache.images.flags.writeable)
            disk=MazeEnv(root,tuple(entries),seed=17)
            memory=MazeEnv(root,tuple(entries),seed=17,map_cache=cache)
            for _ in range(5):
                expected,info=disk.reset()
                with patch.object(Path,'read_bytes',side_effect=AssertionError('warm file read')):
                    self.assertIs(shared_map_cache(root,entries),cache)
                    actual,other=memory.reset()
                    # Spawn workers receive this same decoded payload, not a file path to load.
                    panel=create_panel_env(('maze_medium','validation',root,memory.entry,cache.get(root,memory.entry)))
                    panel.reset(); panel.step(4); panel.close()
                np.testing.assert_array_equal(expected.student_rgb,actual.student_rgb)
                self.assertEqual(info,other)
                self.assertEqual(disk.np_random.bit_generator.state,memory.np_random.bit_generator.state)
                for action in (0,4,2):
                    a=disk.step(action); b=memory.step(action)
                    np.testing.assert_array_equal(a[0].student_rgb,b[0].student_rgb)
                    self.assertEqual(a[1:],b[1:])
            (root/'train/0/0.png').write_bytes(b'corrupted')
            with self.assertRaises(ValueError): MazeMapCache(root,entries)
            cache.close(); self.assertIsNone(cache.images)

    def test_forward_local_e3_gradients_and_optimizer_reuse(self):
        old=torch.get_num_threads(); torch.set_num_threads(2)
        try:
            config=load_config('tasks/continual_nav_opd/configs/smoke.yaml')
            for dual in (False,True):
                cached=StandalonePolicy(config)
                if dual: cached=DualPolicy(config,cached,kb_ready=True)
                plain=deepcopy(cached)
                plain.config=replace(config,optimization=replace(config.optimization,e3_cache=False))
                if dual:
                    plain.kb.config=plain.active.config=plain.config
                rgb=torch.randint(256,(3,2,3,84,84),dtype=torch.uint8)
                starts=torch.tensor([[True,True],[False,True],[True,False]])
                valid=torch.ones(3,2,dtype=torch.bool)
                opt=torch.optim.Adam([p for p in cached.parameters() if p.requires_grad],lr=1e-4)
                for iteration in range(2):
                    plain.load_state_dict(cached.state_dict()); plain.zero_grad(); cached.zero_grad()
                    a=_sequence(cached,rgb,cached.initial_state(2),starts,valid)
                    b=_sequence(plain,rgb,plain.initial_state(2),starts,valid)
                    torch.testing.assert_close(a.logits,b.logits,atol=2e-6,rtol=1e-5)
                    a.logits.square().sum().backward(); b.logits.square().sum().backward()
                    for (name,x),(_,y) in zip(cached.named_parameters(),plain.named_parameters()):
                        self.assertEqual(x.grad is None,y.grad is None,name)
                        if x.grad is not None:
                            torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=3e-4,msg=name)
                    opt.step()
                # No cached autograd graph survives an update or enters state_dict.
                self.assertEqual(cached.state_dict().keys(),plain.state_dict().keys())
        finally: torch.set_num_threads(old)
