
neut_reactions = {
    # ######### NEUTRINO MODE #########
    "neutrino": {
        "CC": {  # Charged Current
            "elastic": {
                1: "NEU,N --> LEPTON-,P",
                2: "NEU,N+X --> LEPTON-,P+X  (X=(N or P))",
            },
            "single_pi_from_delta": {
                11: "NEU,P --> LEPTON-,P,PI+",
                12: "NEU,N --> LEPTON-,P,PI0",
                13: "NEU,N --> LEPTON-,N,PI+",
                15: "NEU,P --> LEPTON-,P,PI+  (diffractive)",
                16: "NEU,O(16) --> LEPTON-,O(16),PI+",
            },
            "single_gamma_from_delta": {
                17: "NEU,N --> LEPTON-,P,GAMMA",
            },
            "single_K_strangeness_violation": {
                18: "NEU,N --> LEPTON-,N,K+",
                19: "NEU,N --> LEPTON-,P,K0",
                20: "NEU,N --> LEPTON-,P,K+",
            },
            "multi_pi": {
                21: "NEU,(N OR P) --> LEPTON-,(N OR P),MULTI PI",
            },
            "single_eta_from_delta": {
                22: "NEU,N --> LEPTON-,P,ETA0",
            },
            "single_K_from_delta": {
                23: "NEU,N --> LEPTON-,LAMBDA,K+",
            },
            "deep_inelastic": {
                26: "NEU,(N OR P) --> LEPTON-,(N OR P),MESONS",
            },
        },
        "NC": {  # Neutral Current
            "single_pi_from_delta": {
                31: "NEU,N --> NEU,N,PI0",
                32: "NEU,P --> NEU,P,PI0",
                33: "NEU,N --> NEU,P,PI-",
                34: "NEU,P --> NEU,N,PI+",
                35: "NEU,P --> NEU,P,PI0 (diffractive)",
                36: "NEU,O(16) --> NEU,O(16),PI0",
            },
            "single_gamma_from_delta": {
                38: "NEU,N --> NEU,N,GAMMA",
                39: "NEU,P --> NEU,P,GAMMA",
            },
            "multi_pi": {
                41: "NEU,(N OR P) --> NEU,(N OR P),MULTI PI",
            },
            "single_eta_from_delta": {
                42: "NEU,N --> NEU,N,ETA0",
                43: "NEU,P --> NEU,P,ETA0",
            },
            "single_K_from_delta": {
                44: "NEU,N --> NEU,LAMBDA,K0",
                45: "NEU,P --> NEU,LAMBDA,K+",
            },
            "deep_inelastic": {
                46: "NEU,(N OR P) --> NEU,(N OR P),MESONS",
            },
            "elastic": {
                51: "NEU,P --> NEU,P",
                52: "NEU,N --> NEU,N",
            },
        },
    },

    # ######### ANTI NEUTRINO MODE #########
    "antineutrino": {
        "CC": {
            "elastic": {
                -1: "NEUBAR,P --> LEPTON+,N",
                -2: "NEU,P+X --> LEPTON-,N+X  (X=(N or P))",
            },
            "single_pi_from_delta": {
                -11: "NEUBAR,N --> LEPTON+,N,PI-",
                -12: "NEUBAR,P --> LEPTON+,N,PI0",
                -13: "NEUBAR,P --> LEPTON+,P,PI-",
                -15: "NEUBAR,P --> LEPTON+,P,PI-  (diffractive)",
                -16: "NEUBAR,O(16) --> LEPTON+,O(16),PI-",
            },
            "single_gamma_from_delta": {
                -17: "NEUBAR,P --> LEPTON+,N,GAMMA",
            },
            "multi_pi": {
                -21: "NEUBAR,(N OR P) --> LEPTON+,(N OR P),MULTI PI",
            },
            "single_eta_from_delta": {
                -22: "NEUBAR,P --> LEPTON+,N,ETA0",
            },
            "single_K_from_delta": {
                -23: "NEUBAR,P --> LEPTON+,LAMBDA,K0",
            },
            "deep_inelastic": {
                -26: "NEUBAR,(N OR P) --> LEPTON+,(N OR P),MESONS",
            },
        },
        "NC": {
            "single_pi_from_delta": {
                -31: "NEUBAR,N --> NEUBAR,N,PI0",
                -32: "NEUBAR,P --> NEUBAR,P,PI0",
                -33: "NEUBAR,N --> NEUBAR,P,PI-",
                -34: "NEUBAR,P --> NEUBAR,N,PI+",
                -35: "NEUBAR,P --> LEPTON+,P,PI0  (diffractive)",
                -36: "NEUBAR,O(16) --> NEUBAR,O(16),PI0",
            },
            "single_gamma_from_delta": {
                -38: "NEUBAR,N --> NEUBAR,N,GAMMA",
                -39: "NEUBAR,P --> NEUBAR,P,GAMMA",
            },
            "multi_pi": {
                -41: "NEUBAR,(N OR P) --> NEUBAR,(N OR P),MULTI PI",
            },
            "single_eta_from_delta": {
                -42: "NEUBAR,N --> NEUBAR,N,ETA0",
                -43: "NEUBAR,P --> NEUBAR,P,ETA0",
            },
            "single_K_from_delta": {
                -44: "NEUBAR,N --> NEUBAR,LAMBDA,K0",
                -45: "NEUBAR,P --> NEUBAR,LAMBDA,K+",
            },
            "deep_inelastic": {
                -46: "NEUBAR,(N OR P) --> NEUBAR,(N OR P),MESONS",
            },
            "elastic": {
                -51: "NEUBAR,P --> NEUBAR,P",
                -52: "NEUBAR,N --> NEUBAR,N",
            },
        },
    },
}

neut_compact = {
    # ===== NEUTRINO MODE =====
    # Charged Current
    1: "CC elastic (n→p)",
    2: "CC quasi-elastic (n→p+X)",
    11: "CC single-pi (p, π+)",
    12: "CC single-pi (n, π0)",
    13: "CC single-pi (n, π+)",
    15: "CC single-pi diffractive (p, π+)",
    16: "CC single-pi (O16, π+)",
    17: "CC single-gamma (n→pγ)",
    18: "CC single-K (n→nK+)",
    19: "CC single-K (n→pK0)",
    20: "CC single-K (n→pK+)",
    21: "CC multi-pi (W<2 GeV)",
    22: "CC single-eta (n→pη)",
    23: "CC single-K (n→ΛK+)",
    26: "CC DIS (W>2 GeV)",

    # Neutral Current
    31: "NC single-pi (n, π0)",
    32: "NC single-pi (p, π0)",
    33: "NC single-pi (n, π-)",
    34: "NC single-pi (p, π+)",
    35: "NC single-pi diffractive (p, π0)",
    36: "NC single-pi (O16, π0)",
    38: "NC single-gamma (n)",
    39: "NC single-gamma (p)",
    41: "NC multi-pi (W<2 GeV)",
    42: "NC single-eta (n, η)",
    43: "NC single-eta (p, η)",
    44: "NC single-K (n→ΛK0)",
    45: "NC single-K (p→ΛK+)",
    46: "NC DIS (W>2 GeV)",
    51: "NC elastic (p)",
    52: "NC elastic (n)",

    # ===== ANTI-NEUTRINO MODE =====
    # Charged Current
    -1: "anti-CC elastic (p→n)",
    -2: "anti-CC quasi-elastic (p→n+X)",
    -11: "anti-CC single-pi (n, π-)",
    -12: "anti-CC single-pi (p, π0)",
    -13: "anti-CC single-pi (p, π-)",
    -15: "anti-CC single-pi diffractive (p, π-)",
    -16: "anti-CC single-pi (O16, π-)",
    -17: "anti-CC single-gamma (p→nγ)",
    -21: "anti-CC multi-pi (W>1.4 GeV)",
    -22: "anti-CC single-eta (p→nη)",
    -23: "anti-CC single-K (p→ΛK0)",
    -26: "anti-CC DIS (W>2 GeV)",

    # Neutral Current
    -31: "anti-NC single-pi (n, π0)",
    -32: "anti-NC single-pi (p, π0)",
    -33: "anti-NC single-pi (n, π-)",
    -34: "anti-NC single-pi (p, π+)",
    -35: "anti-NC single-pi diffractive (p, π0)",
    -36: "anti-NC single-pi (O16, π0)",
    -38: "anti-NC single-gamma (n)",
    -39: "anti-NC single-gamma (p)",
    -41: "anti-NC multi-pi (W>1.4 GeV)",
    -42: "anti-NC single-eta (n, η)",
    -43: "anti-NC single-eta (p, η)",
    -44: "anti-NC single-K (n→ΛK0)",
    -45: "anti-NC single-K (p→ΛK+)",
    -46: "anti-NC DIS (W>2 GeV)",
    -51: "anti-NC elastic (p)",
    -52: "anti-NC elastic (n)",
}

def neut_group(code):
    """
    Given a NEUT reaction code (int), return its:
      - mode ('neutrino' or 'antineutrino')
      - interaction type ('CC' or 'NC')
      - subcategory (e.g. 'single_pi_from_delta')
      - description string
    Returns None if not found.
    """
    for mode, mode_dict in neut_reactions.items():
        for interaction_type, int_dict in mode_dict.items():
            for subcat, subdict in int_dict.items():
                if code in subdict:
                    return {
                        "mode": mode,
                        "interaction_type": interaction_type,
                        "subcategory": subcat,
                        "description": subdict[code],
                    }
    return None
