from __future__ import annotations

from backend.app.env.red_policies.base_policy import AttackStage, RedPolicy


class IoTBotnet(RedPolicy):
    scenario_id = "iot_botnet"
    kill_chain = [
        AttackStage("scan_host", ["iot_projector_01"], "Reconnaissance", "Scan IoT subnet for weak credentials", "low"),
        AttackStage("exploit_vulnerability", ["iot_projector_01"], "Initial Access", "Compromise IoT projector using default credentials", "medium"),
        AttackStage("scan_host", ["lab_workstation_01", "student_device_01"], "Reconnaissance", "Scan campus subnet from IoT foothold", "medium"),
        AttackStage("lateral_move", ["lab_workstation_01", "lab_workstation_02"], "Lateral Movement", "Propagate to lab workstations", "high"),
        AttackStage("lateral_move", ["student_device_01", "student_device_02", "student_device_03"], "Lateral Movement", "Spread payload to student endpoints", "high"),
        AttackStage("exploit_vulnerability", ["ALL_COMPROMISED"], "Execution", "Deploy miner payload on compromised fleet", "critical"),
    ]
