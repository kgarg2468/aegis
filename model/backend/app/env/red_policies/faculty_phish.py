from __future__ import annotations

from backend.app.env.red_policies.base_policy import AttackStage, RedPolicy


class FacultySpearPhish(RedPolicy):
    scenario_id = "faculty_phish"
    kill_chain = [
        AttackStage("exploit_vulnerability", ["faculty_device_01", "faculty_device_02"], "Initial Access", "Deliver spear-phish payload to faculty endpoint", "high"),
        AttackStage("exploit_vulnerability", ["faculty_device_01"], "Execution", "Execute payload and establish C2", "high"),
        AttackStage("scan_host", ["faculty_device_01"], "Reconnaissance", "Local network recon from compromised faculty host", "medium"),
        AttackStage("enumerate_service", ["research_server_01", "shared_storage"], "Reconnaissance", "Enumerate research storage services", "medium"),
        AttackStage("lateral_move", ["research_server_01"], "Lateral Movement", "Lateral move into research server", "high"),
        AttackStage("lateral_move", ["shared_storage"], "Lateral Movement", "Mount shared research storage", "critical"),
        AttackStage("exfiltrate_data", ["shared_storage"], "Exfiltration", "Exfiltrate research data over covert channel", "critical"),
    ]
