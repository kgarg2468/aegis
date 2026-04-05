from __future__ import annotations

from backend.app.env.red_policies.base_policy import AttackStage, RedPolicy


class RansomwareCascade(RedPolicy):
    scenario_id = "ransomware_cascade"
    kill_chain = [
        AttackStage("scan_host", ["print_server"], "Reconnaissance", "Enumerate print spooler attack surface", "medium"),
        AttackStage("exploit_vulnerability", ["print_server"], "Initial Access", "Exploit print spooler RCE", "critical"),
        AttackStage("exploit_vulnerability", ["print_server"], "Execution", "Deploy ransomware loader on print server", "critical"),
        AttackStage("lateral_move", ["student_device_01", "faculty_device_01", "lab_workstation_01"], "Lateral Movement", "First ransomware propagation wave", "critical"),
        AttackStage("lateral_move", ["student_device_04", "faculty_device_03", "lab_workstation_02"], "Lateral Movement", "Second ransomware propagation wave", "critical"),
        AttackStage("exfiltrate_data", ["shared_storage", "research_server_01"], "Impact", "Encrypt shared storage and research assets", "critical"),
    ]
