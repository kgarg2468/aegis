import sys
import types

import numpy as np

from backend.app.rl import eval as eval_module


def test_load_trained_policy_registers_components_and_uses_absolute_checkpoint(monkeypatch, tmp_path):
    ckpt = tmp_path / "final"
    ckpt.mkdir()

    calls: dict[str, object] = {"ray_init": 0, "register": 0, "checkpoint": None}

    class FakeAlgorithm:
        @staticmethod
        def from_checkpoint(path: str):
            calls["checkpoint"] = path

            class AlgoObj:
                def compute_single_action(self, _obs, explore=False):
                    return [2, 7]

            return AlgoObj()

    fake_ray = types.ModuleType("ray")

    def fake_init(*args, **kwargs):
        calls["ray_init"] = int(calls["ray_init"]) + 1

    fake_ray.init = fake_init

    fake_algorithm_mod = types.ModuleType("ray.rllib.algorithms.algorithm")
    fake_algorithm_mod.Algorithm = FakeAlgorithm

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.rllib", types.ModuleType("ray.rllib"))
    monkeypatch.setitem(sys.modules, "ray.rllib.algorithms", types.ModuleType("ray.rllib.algorithms"))
    monkeypatch.setitem(sys.modules, "ray.rllib.algorithms.algorithm", fake_algorithm_mod)

    def fake_register() -> None:
        calls["register"] = int(calls["register"]) + 1

    monkeypatch.setattr(eval_module, "_register_rllib_components", fake_register)

    policy = eval_module._load_trained_policy(str(ckpt))

    assert calls["ray_init"] == 1
    assert calls["register"] == 1
    assert calls["checkpoint"] == str(ckpt.resolve())
    assert policy({"node_features": []}) == [2, 7]


def test_load_trained_policy_falls_back_when_checkpoint_missing():
    policy = eval_module._load_trained_policy("/path/that/does/not/exist")
    action = policy({"node_features": np.zeros((28, 22), dtype=np.float32)})
    assert isinstance(action, list)
    assert len(action) == 2
