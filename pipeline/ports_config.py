"""
7_ports_config.py
Waze Cargo — Chilean Maritime Port Catalogue
Zone-aware closure thresholds based on DIRECTEMAR standards and academic research
(Winckler et al., Coastal Engineering Journal)

Zones:
  NORTH  — open-sea ports, exposed to long-period SW swell from South Pacific
  CENTRAL — bay ports (Valparaíso, San Antonio), sensitive to N/NW swell into bay mouth
  SOUTH  — wind-dominated, fjord/channel ports, less swell but strong gales

Each port has:
  lat / lon              — coordinates for Open-Meteo API call
  zone                   — NORTH / CENTRAL / SOUTH
  port_mouth_bearing     — compass degrees the harbour entrance faces (swell hits directly)
  protected_arc_deg      — +/- degrees either side of OPPOSITE bearing that land blocks
  hs_threshold_m         — significant wave height (m) triggering operational restriction
  hs_closure_m           — Hs (m) triggering full port closure
  wind_restriction_kt    — wind speed (knots) triggering restriction
  wind_closure_kt        — wind speed (knots) triggering full closure
  period_concern_s       — wave period (seconds) above which long-swell penalty applies
  primary_swell_risk_dir — compass bearing range (degrees) of dangerous swell approach
  notes                  — operational context
"""

PORTS = {
    # ── NORTHERN ZONE ──────────────────────────────────────────────────────────
    # Open-sea ports directly exposed to South Pacific SW swell.
    # Even though far from storm generation zones, long-period swell travels
    # unimpeded — these ports historically have the MOST closure hours/year.

    "ARICA": {
        "lat": -18.475, "lon": -70.322,
        "zone": "NORTH",
        "port_mouth_bearing": 270,       # faces due West
        "protected_arc_deg": 90,
        "hs_threshold_m": 0.8,           # very low threshold — little natural shelter
        "hs_closure_m": 1.5,
        "wind_restriction_kt": 25,
        "wind_closure_kt": 35,
        "period_concern_s": 12,
        "primary_swell_risk_dir": (200, 270),   # SW to W swell
        "notes": "Breakwater 1100m. Exposed to Bolivian altiplano winds (surazos). "
                 "310 hrs downtime/yr historically.",
    },
    "IQUIQUE": {
        "lat": -20.213, "lon": -70.152,
        "zone": "NORTH",
        "port_mouth_bearing": 265,
        "protected_arc_deg": 80,
        "hs_threshold_m": 0.8,
        "hs_closure_m": 1.5,
        "wind_restriction_kt": 25,
        "wind_closure_kt": 35,
        "period_concern_s": 12,
        "primary_swell_risk_dir": (200, 270),
        "notes": "100 hrs downtime/yr. Climate change projected to increase. "
                 "Exposed to same SW swell corridor as Arica.",
    },
    "ANTOFAGASTA": {
        "lat": -23.650, "lon": -70.400,
        "zone": "NORTH",
        "port_mouth_bearing": 270,
        "protected_arc_deg": 70,
        "hs_threshold_m": 0.9,
        "hs_closure_m": 1.8,
        "wind_restriction_kt": 28,
        "wind_closure_kt": 38,
        "period_concern_s": 11,
        "primary_swell_risk_dir": (200, 275),
        "notes": "1,88 hrs downtime/yr — highest in Chile. Worst-case climate projection. "
                 "Copper export hub — closure costs ~$345M/yr nationally.",
    },
    "MEJILLONES": {
        "lat": -23.100, "lon": -70.449,
        "zone": "NORTH",
        "port_mouth_bearing": 280,
        "protected_arc_deg": 100,        # Mejillones bay offers some natural shelter
        "hs_threshold_m": 1.2,
        "hs_closure_m": 2.0,
        "wind_restriction_kt": 28,
        "wind_closure_kt": 40,
        "period_concern_s": 11,
        "primary_swell_risk_dir": (220, 280),
        "notes": "Bay position gives better shelter than Antofagasta. "
                 "Climate projections show IMPROVED conditions by mid-century.",
    },
    "CALDERA": {
        "lat": -27.68, "lon": -70.819,
        "zone": "NORTH",
        "port_mouth_bearing": 275,
        "protected_arc_deg": 85,
        "hs_threshold_m": 1.0,
        "hs_closure_m": 1.8,
        "wind_restriction_kt": 28,
        "wind_closure_kt": 38,
        "period_concern_s": 11,
        "primary_swell_risk_dir": (210, 280),
        "notes": "Key early-season grape and copper export port. "
                 "High congestion forecast Jan 2026.",
    },
    "COQUIMBO": {
        "lat": -29.953, "lon": -71.343,
        "zone": "NORTH",
        "port_mouth_bearing": 290,       # faces NW into protected bay
        "protected_arc_deg": 110,        # Punta Tortuga provides SW protection
        "hs_threshold_m": 1.2,
        "hs_closure_m": 2.0,
        "wind_restriction_kt": 30,
        "wind_closure_kt": 40,
        "period_concern_s": 12,
        "primary_swell_risk_dir": (270, 330),   # NW swell into bay mouth
        "notes": "Bay orientation shifts dominant risk to NW swell. "
                 "Arica and Coquimbo historically show no relevant change under climate projections.",
    },

    # ── CENTRAL ZONE ───────────────────────────────────────────────────────────
    # Bay-facing ports. Different risk profile: sensitive to N/NW swell that
    # wraps into the bay. SW swell is partially blocked by Punta Ángeles / terrain.
    # These are Chile's highest-volume ports.

    "VALPARAISO": {
        "lat": -33.35, "lon": -71.628,
        "zone": "CENTRAL",
        "port_mouth_bearing": 320,       # faces NW — key difference from northern ports
        "protected_arc_deg": 120,        # hills and Punta Ángeles block S/SW
        "hs_threshold_m": 1.5,
        "hs_closure_m": 2.5,
        "wind_restriction_kt": 30,
        "wind_closure_kt": 45,
        "period_concern_s": 10,
        "primary_swell_risk_dir": (280, 360),   # NW to N swell into bay
        "notes": "Second largest container port. 30%+ of Chile international trade. "
                 "Climate projections show SLIGHTLY IMPROVED conditions. "
                 "Sensitive to norte (N swell events) not SW swell.",
    },
    "SAN_ANTONIO": {
        "lat": -33.593, "lon": -71.621,
        "zone": "CENTRAL",
        "port_mouth_bearing": 270,       # faces W — more exposed than Valparaíso
        "protected_arc_deg": 100,
        "hs_threshold_m": 1.5,           # 1.8m for vessels >300m (San Antonio Terminal)
        "hs_closure_m": 2.0,
        "wind_restriction_kt": 30,
        "wind_closure_kt": 45,
        "period_concern_s": 10,
        "primary_swell_risk_dir": (240, 320),   # SW to NW
        "notes": "Largest Chilean port by volume. Fruit/wine export peak Jan-Apr. "
                 "Climate change projects 72 additional downtime hrs/yr. "
                 "Operational Hs limit 2.0m for large vessels at STI terminal.",
    },
    "QUINTERO": {
        "lat": -32.777, "lon": -71.533,
        "zone": "CENTRAL",
        "port_mouth_bearing": 300,
        "protected_arc_deg": 110,
        "hs_threshold_m": 1.5,
        "hs_closure_m": 2.5,
        "wind_restriction_kt": 30,
        "wind_closure_kt": 45,
        "period_concern_s": 10,
        "primary_swell_risk_dir": (270, 350),
        "notes": "Industrial port. Closes together with Valparaíso and San Antonio "
                 "during major weather fronts from the north.",
    },

    # ── SOUTHERN ZONE ──────────────────────────────────────────────────────────
    # Wind-dominated. Swell generation zones are nearby so periods are shorter.
    # Fjords and channels provide natural swell shelter, but gale-force winds
    # (surazos and nortes) are the primary closure trigger.

    "SAN_VICENTE": {
        "lat": -36.977, "lon": -73.151,
        "zone": "SOUTH",
        "port_mouth_bearing": 290,
        "protected_arc_deg": 130,        # Arauco Bay provides strong natural shelter
        "hs_threshold_m": 1.8,
        "hs_closure_m": 3.0,
        "wind_restriction_kt": 35,
        "wind_closure_kt": 50,
        "period_concern_s": 8,
        "primary_swell_risk_dir": (240, 310),
        "notes": "Climate projections show SIGNIFICANTLY IMPROVED conditions. "
                 "Bay orientation provides excellent SW swell protection. "
                 "Wind from NW fronts is main risk.",
    },
    "CORONEL": {
        "lat": -37.30, "lon": -73.148,
        "zone": "SOUTH",
        "port_mouth_bearing": 285,
        "protected_arc_deg": 125,
        "hs_threshold_m": 1.8,
        "hs_closure_m": 3.0,
        "wind_restriction_kt": 35,
        "wind_closure_kt": 50,
        "period_concern_s": 8,
        "primary_swell_risk_dir": (240, 310),
        "notes": "Timber/bulk export port. High congestion Jan forecast. "
                 "Protected by Arauco peninsula from worst SW swell.",
    },
    "LIRQUEN": {
        "lat": -36.888, "lon": -73.20,
        "zone": "SOUTH",
        "port_mouth_bearing": 10,       # faces N — very sheltered from Pacific swell
        "protected_arc_deg": 150,
        "hs_threshold_m": 2.0,
        "hs_closure_m": 3.5,
        "wind_restriction_kt": 38,
        "wind_closure_kt": 55,
        "period_concern_s": 7,
        "primary_swell_risk_dir": (330, 60),   # N swell from Concepción Bay
        "notes": "Most sheltered industrial port in Bio-Bío. "
                 "Wind is primary closure trigger, not swell.",
    },
    "TALCAHUANO": {
        "lat": -36.700, "lon": -73.117,
        "zone": "SOUTH",
        "port_mouth_bearing": 350,       # faces N into Concepción Bay
        "protected_arc_deg": 150,
        "hs_threshold_m": 2.0,
        "hs_closure_m": 3.5,
        "wind_restriction_kt": 38,
        "wind_closure_kt": 55,
        "period_concern_s": 7,
        "primary_swell_risk_dir": (310, 50),
        "notes": "Naval base + commercial. Concepción Bay provides strong swell attenuation. "
                 "Winter nortes (N gales) main hazard.",
    },
    "PUERTO_ANGAMOS": {
        "lat": -23.43, "lon": -70.449,
        "zone": "NORTH",
        "port_mouth_bearing": 270,
        "protected_arc_deg": 75,
        "hs_threshold_m": 1.0,
        "hs_closure_m": 1.8,
        "wind_restriction_kt": 28,
        "wind_closure_kt": 38,
        "period_concern_s": 11,
        "primary_swell_risk_dir": (200, 275),
        "notes": "Chile's top copper export port by FOB value. "
                 "Serves Escondida and Collahuasi mines. "
                 "Industrial terminal with 24/7 operations.",
    },
}

# ── Zone-level defaults (fallback if a port threshold is missing) ────────────
ZONE_DEFAULTS = {
    "NORTH": {
        "hs_threshold_m": 1.0,
        "hs_closure_m": 1.8,
        "wind_restriction_kt": 28,
        "wind_closure_kt": 38,
        "period_concern_s": 11,
    },
    "CENTRAL": {
        "hs_threshold_m": 1.5,
        "hs_closure_m": 2.5,
        "wind_restriction_kt": 30,
        "wind_closure_kt": 45,
        "period_concern_s": 10,
    },
    "SOUTH": {
        "hs_threshold_m": 2.0,
        "hs_closure_m": 3.5,
        "wind_restriction_kt": 38,
        "wind_closure_kt": 55,
        "period_concern_s": 8,
    },
}

# ── Risk label mapping ────────────────────────────────────────────────────────
RISK_LABELS = {
    0: "NORMAL",        # fully operational
    1: "WATCH",         # conditions deteriorating, monitoring required
    2: "ADVISORY",      # operational restrictions possible, slow vessels advised
    3: "WARNING",       # significant restrictions, port may reduce operations
    4: "CLOSED",        # port authority likely to suspend operations
}

RISK_COLORS = {
    "NORMAL":   "#2ECC71",
    "WATCH":    "#F1C40F",
    "ADVISORY": "#E67E22",
    "WARNING":  "#E74C3C",
    "CLOSED":   "#8E44AD",
}
