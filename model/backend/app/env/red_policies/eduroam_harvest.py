from __future__ import annotations

from backend.app.env.red_policies.base_policy import AttackStage, RedPolicy


class EduroamCredentialHarvest(RedPolicy):
    scenario_id = "eduroam_harvest"
    kill_chain = [
        AttackStage("scan_host", ["eduroam_ap_01", "eduroam_ap_02"], "Reconnaissance", "Scan eduroam APs to profile auth behavior", "medium"),
        AttackStage("exploit_vulnerability", ["eduroam_ap_01"], "Credential Access", "Harvest RADIUS credentials via evil twin AP", "high"),
        AttackStage("exploit_vulnerability", ["vpn_gateway"], "Initial Access", "Authenticate into VPN using stolen credentials", "high"),
        AttackStage("lateral_move", ["auth_server"], "Lateral Movement", "Pivot from VPN segment to auth server", "critical"),
        AttackStage("enumerate_service", ["sis_server"], "Reconnaissance", "Enumerate SIS attack surface via trust path", "medium"),
        AttackStage("exploit_vulnerability", ["sis_server"], "Initial Access", "Exploit SIS API weakness for record access", "critical"),
        AttackStage("exfiltrate_data", ["sis_server"], "Exfiltration", "Exfiltrate student PII to external drop", "critical"),
    ]
