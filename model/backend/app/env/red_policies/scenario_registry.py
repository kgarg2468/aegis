from __future__ import annotations

import numpy as np

from backend.app.env.red_policies.base_policy import RedPolicy
from backend.app.env.red_policies.eduroam_harvest import EduroamCredentialHarvest
from backend.app.env.red_policies.faculty_phish import FacultySpearPhish
from backend.app.env.red_policies.insider_threat import InsiderThreat
from backend.app.env.red_policies.iot_botnet import IoTBotnet
from backend.app.env.red_policies.ransomware_cascade import RansomwareCascade


SCENARIO_REGISTRY: dict[str, type[RedPolicy]] = {
    "eduroam_harvest": EduroamCredentialHarvest,
    "faculty_phish": FacultySpearPhish,
    "iot_botnet": IoTBotnet,
    "insider_threat": InsiderThreat,
    "ransomware_cascade": RansomwareCascade,
}


def sample_scenario(
    rng: np.random.Generator,
    scenario_weights: list[float] | None = None,
    force_scenario: str | None = None,
) -> RedPolicy:
    if force_scenario:
        scenario_cls = SCENARIO_REGISTRY[force_scenario]
        return scenario_cls(rng=rng)

    names = list(SCENARIO_REGISTRY.keys())
    if scenario_weights is None or len(scenario_weights) != len(names):
        scenario_weights = [1.0 / len(names)] * len(names)

    scenario_name = str(rng.choice(names, p=np.array(scenario_weights) / np.sum(scenario_weights)))
    scenario_cls = SCENARIO_REGISTRY[scenario_name]
    return scenario_cls(
        entry_point_jitter=True,
        step_delay_range=(1, 3),
        exploit_failure_prob=0.15,
        target_shuffle=True,
        rng=rng,
    )
