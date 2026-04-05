from __future__ import annotations

from backend.app.env.red_policies.base_policy import AttackStage, RedPolicy


class InsiderThreat(RedPolicy):
    scenario_id = "insider_threat"
    kill_chain = [
        AttackStage("exploit_vulnerability", ["active_directory"], "Credential Access", "Use stolen admin credentials to access AD", "high"),
        AttackStage("privilege_escalate", ["active_directory"], "Persistence", "Create hidden elevated service account", "high"),
        AttackStage("exploit_vulnerability", ["vpn_gateway"], "Defense Evasion", "Modify firewall rule set for covert channel", "high"),
        AttackStage("lateral_move", ["auth_server"], "Lateral Movement", "Install persistence foothold on auth server", "critical"),
        AttackStage("enumerate_service", ["auth_server"], "Credential Access", "Harvest credential hashes from auth server", "high"),
        AttackStage("exfiltrate_data", ["auth_server"], "Exfiltration", "Slow credential hash exfiltration", "high"),
        AttackStage("lateral_move", ["finance_server"], "Lateral Movement", "Pivot into finance server", "critical"),
        AttackStage("exfiltrate_data", ["finance_server"], "Exfiltration", "Exfiltrate financial records in small batches", "critical"),
    ]
