# ============================================================
# top5_matcher.py
# Score all donors for a patient and return the top 5 matches
# ============================================================

import pandas as pd
from compute_hla import extract_hla_profile, compute_hla_differences
from predict_core import build_and_predict





def score_donor_against_patient(donor_row: dict, patient_row: dict):
    donor_hla   = extract_hla_profile(donor_row,   'donor')
    patient_hla = extract_hla_profile(patient_row, 'patient')
    antigen_diff, allel_diff = compute_hla_differences(donor_hla, patient_hla)

    donor = {
        'donor_age':         donor_row['donor_age'],
        'donor_ABO':         donor_row['donor_ABO'],
        'donor_CMV':         donor_row['donor_CMV'],
        'donor_gender':      donor_row['donor_gender'],
        'antigen':           antigen_diff,
        'allel':             allel_diff,
        'CD34_x1e6_per_kg':  donor_row['CD34_x1e6_per_kg'],
        'CD3_x1e8_per_kg':   donor_row['CD3_x1e8_per_kg'],
        'CD3_to_CD34_ratio': donor_row['CD3_to_CD34_ratio'],
        'stem_cell_source':  donor_row['stem_cell_source'],
    }
    patient = {
        'recipient_age':       patient_row['recipient_age'],
        'recipient_gender':    patient_row['recipient_gender'],
        'recipient_body_mass': patient_row['recipient_body_mass'],
        'recipient_ABO':       patient_row['recipient_ABO'],
        'recipient_rh':        patient_row['recipient_rh'],
        'recipient_CMV':       patient_row['recipient_CMV'],
        'disease':             patient_row['disease'],
        'disease_group':       patient_row['disease_group'],
        'risk_group':          patient_row['risk_group'],
        'tx_post_relapse':     patient_row['tx_post_relapse'],
    }

    compat, derived, s_pred, s_prob, r_pred, r_prob, g_pred, g_prob = \
        build_and_predict(donor, patient, antigen_diff, allel_diff)

    return {
        'donor_id':            int(donor_row['donor_id']),
        'donor_age':           donor['donor_age'],
        'donor_ABO':           donor['donor_ABO'],
        'donor_CMV':           donor['donor_CMV'],
        'donor_gender':        donor['donor_gender'],
        'stem_cell_source':    donor['stem_cell_source'],
        'antigen_diff':        antigen_diff,
        'allel_diff':          allel_diff,
        'hla_match':           derived['HLA_match'],
        'abo_match':           derived['ABO_match'],
        'compatibility_score': compat['compatibility_score'],
        'grade':               compat['grade'],
        'alive_probability':   round(float(s_prob[0]) * 100, 1),
        'relapse_risk':        round(float(r_prob[1]) * 100, 1),
        'gvhd_risk':           round(float(g_prob[1]) * 100, 1),
    }





def find_top5_donors(patient: dict, donor_db: pd.DataFrame):

    results = []

    for _, donor_row in donor_db.iterrows():
        try:
            result = score_donor_against_patient(
                donor_row.to_dict(),
                patient
            )
            results.append(result)
        except Exception:
            continue

    results.sort(
        key=lambda x: (x['compatibility_score'], x['alive_probability']),
        reverse=True
    )

    top5 = results[:5]

    return top5