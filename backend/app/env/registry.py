"""Canonical integration contract registry and enums.

This module is the source of truth for node IDs and enum values emitted in replay bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class NodeSpec:
    id: str
    type: str
    zone: str
    shape: str
    size: int


NODE_REGISTRY: Final[list[NodeSpec]] = [
    NodeSpec("internet", "external", "external", "hexagon", 40),
    NodeSpec("vpn_gateway", "infrastructure", "perimeter", "round-rectangle", 44),
    NodeSpec("web_portal", "infrastructure", "perimeter", "round-rectangle", 44),
    NodeSpec("dns_server", "infrastructure", "perimeter", "round-rectangle", 44),
    NodeSpec("eduroam_ap_01", "infrastructure", "campus", "round-rectangle", 44),
    NodeSpec("eduroam_ap_02", "infrastructure", "campus", "round-rectangle", 44),
    NodeSpec("eduroam_ap_03", "infrastructure", "campus", "round-rectangle", 44),
    NodeSpec("student_device_01", "endpoint", "campus", "ellipse", 36),
    NodeSpec("student_device_02", "endpoint", "campus", "ellipse", 36),
    NodeSpec("student_device_03", "endpoint", "campus", "ellipse", 36),
    NodeSpec("student_device_04", "endpoint", "campus", "ellipse", 36),
    NodeSpec("student_device_05", "endpoint", "campus", "ellipse", 36),
    NodeSpec("faculty_device_01", "endpoint", "campus", "ellipse", 36),
    NodeSpec("faculty_device_02", "endpoint", "campus", "ellipse", 36),
    NodeSpec("faculty_device_03", "endpoint", "campus", "ellipse", 36),
    NodeSpec("print_server", "infrastructure", "campus", "round-rectangle", 44),
    NodeSpec("iot_projector_01", "iot", "campus", "triangle", 32),
    NodeSpec("lab_workstation_01", "endpoint", "campus", "ellipse", 36),
    NodeSpec("lab_workstation_02", "endpoint", "campus", "ellipse", 36),
    NodeSpec("auth_server", "hvt", "admin", "diamond", 52),
    NodeSpec("active_directory", "hvt", "admin", "diamond", 52),
    NodeSpec("sis_server", "hvt", "admin", "diamond", 52),
    NodeSpec("finance_server", "hvt", "admin", "diamond", 52),
    NodeSpec("hr_server", "hvt", "admin", "diamond", 52),
    NodeSpec("research_server_01", "infrastructure", "research", "round-rectangle", 44),
    NodeSpec("research_server_02", "infrastructure", "research", "round-rectangle", 44),
    NodeSpec("shared_storage", "hvt", "research", "diamond", 52),
    NodeSpec("irb_system", "hvt", "research", "diamond", 52),
]

NODE_REGISTRY_IDS: Final[list[str]] = [node.id for node in NODE_REGISTRY]
NODE_INDEX_BY_ID: Final[dict[str, int]] = {node_id: idx for idx, node_id in enumerate(NODE_REGISTRY_IDS)}

ZONE_MEMBER_IDS: Final[dict[str, list[str]]] = {
    "external": ["internet"],
    "perimeter": ["vpn_gateway", "web_portal", "dns_server"],
    "campus": [
        "eduroam_ap_01",
        "eduroam_ap_02",
        "eduroam_ap_03",
        "student_device_01",
        "student_device_02",
        "student_device_03",
        "student_device_04",
        "student_device_05",
        "faculty_device_01",
        "faculty_device_02",
        "faculty_device_03",
        "print_server",
        "iot_projector_01",
        "lab_workstation_01",
        "lab_workstation_02",
    ],
    "admin": ["auth_server", "active_directory", "sis_server", "finance_server", "hr_server"],
    "research": ["research_server_01", "research_server_02", "shared_storage", "irb_system"],
}

ZONE_LABELS: Final[dict[str, str]] = {
    "perimeter": "PERIMETER / DMZ",
    "campus": "CAMPUS NETWORK",
    "admin": "ADMIN BACKBONE",
    "research": "RESEARCH SEGMENT",
}

HOST_SERVICES: Final[dict[str, list[str]]] = {
    "internet": [],
    "vpn_gateway": ["vpn", "ssh"],
    "web_portal": ["http", "https"],
    "dns_server": ["dns"],
    "eduroam_ap_01": ["radius_client", "wifi"],
    "eduroam_ap_02": ["radius_client", "wifi"],
    "eduroam_ap_03": ["radius_client", "wifi"],
    "student_device_01": ["browser", "email_client"],
    "student_device_02": ["browser", "email_client"],
    "student_device_03": ["browser", "email_client"],
    "student_device_04": ["browser", "email_client"],
    "student_device_05": ["browser", "email_client"],
    "faculty_device_01": ["browser", "email_client", "ssh", "file_share"],
    "faculty_device_02": ["browser", "email_client", "ssh", "file_share"],
    "faculty_device_03": ["browser", "email_client", "ssh", "file_share"],
    "print_server": ["smb", "lpd", "http"],
    "iot_projector_01": ["http", "telnet"],
    "lab_workstation_01": ["ssh", "rdp", "file_share"],
    "lab_workstation_02": ["ssh", "rdp", "file_share"],
    "auth_server": ["radius", "ldap", "kerberos"],
    "active_directory": ["ldap", "kerberos", "dns"],
    "sis_server": ["https", "database", "api"],
    "finance_server": ["https", "database"],
    "hr_server": ["https", "database"],
    "research_server_01": ["ssh", "nfs", "http"],
    "research_server_02": ["ssh", "nfs", "http"],
    "shared_storage": ["nfs", "smb", "ssh"],
    "irb_system": ["https", "database"],
}

HOST_CRITICALITY: Final[dict[str, float]] = {
    "internet": 0.0,
    "vpn_gateway": 0.7,
    "web_portal": 0.5,
    "dns_server": 0.6,
    "eduroam_ap_01": 0.4,
    "eduroam_ap_02": 0.4,
    "eduroam_ap_03": 0.4,
    "student_device_01": 0.2,
    "student_device_02": 0.2,
    "student_device_03": 0.2,
    "student_device_04": 0.2,
    "student_device_05": 0.2,
    "faculty_device_01": 0.4,
    "faculty_device_02": 0.4,
    "faculty_device_03": 0.4,
    "print_server": 0.3,
    "iot_projector_01": 0.1,
    "lab_workstation_01": 0.3,
    "lab_workstation_02": 0.3,
    "auth_server": 1.0,
    "active_directory": 1.0,
    "sis_server": 0.95,
    "finance_server": 0.9,
    "hr_server": 0.85,
    "research_server_01": 0.7,
    "research_server_02": 0.7,
    "shared_storage": 0.8,
    "irb_system": 0.85,
}

BLUE_ACTIONS: Final[dict[int, str]] = {
    0: "monitor_host",
    1: "patch_service",
    2: "isolate_host",
    3: "block_connection",
    4: "rotate_credentials",
    5: "deploy_deception",
}

BLUE_ACTION_INDEX: Final[dict[str, int]] = {v: k for k, v in BLUE_ACTIONS.items()}

VALID_NODE_STATES: Final[set[str]] = {
    "neutral",
    "monitored",
    "probed",
    "compromised",
    "critical",
    "isolated",
    "patched",
}

VALID_EDGE_STATES: Final[set[str]] = {
    "normal",
    "scanning",
    "lateral_movement",
    "exfiltration",
    "credential_flow",
    "blocked",
}

VALID_ACTORS: Final[set[str]] = {"RED", "BLUE", "ENV"}

VALID_ACTION_TYPES: Final[set[str]] = {
    "scan_host",
    "enumerate_service",
    "exploit_vulnerability",
    "lateral_move",
    "privilege_escalate",
    "exfiltrate_data",
    *BLUE_ACTIONS.values(),
}

VALID_OUTCOMES: Final[set[str]] = {"success", "failure", "partial", "blocked"}
VALID_SEVERITIES: Final[set[str]] = {"critical", "high", "medium", "low", "info"}
VALID_MITRE_TACTICS: Final[set[str]] = {
    "Reconnaissance",
    "Initial Access",
    "Execution",
    "Lateral Movement",
    "Credential Access",
    "Exfiltration",
    "Defense Evasion",
    "Persistence",
    "Collection",
    "Impact",
}

SCENARIO_DISPLAY_NAMES: Final[dict[str, str]] = {
    "eduroam_harvest": "Eduroam Credential Harvesting → SIS Data Breach",
    "faculty_phish": "Faculty Spear Phish → Research Data Theft",
    "iot_botnet": "IoT Botnet → Resource Exhaustion",
    "insider_threat": "Insider Threat → Active Directory Backdoor",
    "ransomware_cascade": "Print Server Ransomware Cascade",
}
