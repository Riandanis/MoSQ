"""
Utility Functions and Cell-Line Mapping for Gastro-Transformer.

Includes:
- Cell-line to tissue type mapping (CRITICAL for IC50 prediction)
- Tissue type encoding
- Embedding generation utilities
- Model checkpoint utilities
- Metrics computation
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TISSUE TYPE ENCODING
# =============================================================================
# NOTE: UPDATE THIS MAPPING BASED ON YOUR ACTUAL TISSUE TYPES

TISSUE_TYPES = {
    0: 'Stomach',           # Gastric
    1: 'Esophagus',
    2: 'Breast',
    3: 'Lung',
    4: 'Colon',
    5: 'Liver',
    6: 'Pancreas',
    7: 'Ovary',
    8: 'Kidney',
    9: 'Bladder',
    10: 'Prostate',
    11: 'Brain',
    12: 'Skin',
    13: 'Blood',            # Leukemia/Lymphoma
    14: 'Bone',
    15: 'Soft_Tissue',
    16: 'Head_Neck',
    17: 'Thyroid',
    18: 'Uterus',
    19: 'Cervix',
    20: 'Small_Intestine',
    21: 'Appendix',
    22: 'Adrenal',
    23: 'Other',
    # M6 fix: added distinct tissue types that were previously collapsed into 'Other'
    24: 'Pleura',           # Mesothelioma (H2452, H28, and others)
    25: 'Neuroblastoma',    # Autonomic ganglia / neuroblastoma cell lines (~31 in GDSC)
}

TISSUE_TO_ID = {v: k for k, v in TISSUE_TYPES.items()}


# =============================================================================
# CELL-LINE TO TISSUE MAPPING
# =============================================================================
# CRITICAL: This mapping is REQUIRED for IC50 prediction to work properly
#
# This mapping includes 998 cell-lines from the IC50 dataset (20260212_learning_data_IC50_label.xlsx)
# Generated using scripts/process_ic50_datasets.py
#
# Sources for cell-line information:
# - Cellosaurus: https://www.cellosaurus.org/
# - CCLE: https://sites.broadinstitute.org/ccle
# - DepMap: https://depmap.org/portal/

CELLLINE_TO_TISSUE = {
    # ===========================================================================
    # ADRENAL GLAND
    # ===========================================================================
    'SW13': 'Adrenal',

    # ===========================================================================
    # AUTONOMIC GANGLIA / NEUROBLASTOMA
    # M6 fix: remapped from 'Other' to 'Neuroblastoma' (ID=25).
    # Neuroblastoma is a distinct neural-crest-origin cancer with unique MYCN
    # amplification biology — lumping with generic 'Other' degrades prototype quality.
    # ===========================================================================
    'ACN': 'Neuroblastoma',
    'BE2-M17': 'Neuroblastoma',
    'CHP-126': 'Neuroblastoma',
    'CHP-134': 'Neuroblastoma',
    'CHP-212': 'Neuroblastoma',
    'GI-ME-N': 'Neuroblastoma',
    'GOTO': 'Neuroblastoma',
    'IMR-5': 'Neuroblastoma',
    'KELLY': 'Neuroblastoma',
    'LAN-6': 'Neuroblastoma',
    'MC-IXC': 'Neuroblastoma',
    'MHH-NB-11': 'Neuroblastoma',
    'NB(TU)1-10': 'Neuroblastoma',
    'NB1': 'Neuroblastoma',
    'NB10': 'Neuroblastoma',
    'NB12': 'Neuroblastoma',
    'NB13': 'Neuroblastoma',
    'NB14': 'Neuroblastoma',
    'NB17': 'Neuroblastoma',
    'NB5': 'Neuroblastoma',
    'NB6': 'Neuroblastoma',
    'NB69': 'Neuroblastoma',
    'NB7': 'Neuroblastoma',
    'NBsusSR': 'Neuroblastoma',
    'NH-12': 'Neuroblastoma',
    'KP-N-RT-BM-1': 'Neuroblastoma',
    'KP-N-YN': 'Neuroblastoma',
    'KP-N-YS': 'Neuroblastoma',
    'SIMA': 'Neuroblastoma',
    'SK-N-AS': 'Neuroblastoma',
    'SK-N-DZ': 'Neuroblastoma',
    'SK-N-FI': 'Neuroblastoma',
    'SK-N-SH': 'Neuroblastoma',
    'TGW': 'Neuroblastoma',

    # ===========================================================================
    # BILIARY TRACT
    # ===========================================================================
    'EGI-1': 'Liver',
    'ETK-1': 'Liver',
    'HuCCT1': 'Liver',
    'TGBC1TKB': 'Liver',
    'TGBC24TKB': 'Liver',

    # ===========================================================================
    # BONE
    # ===========================================================================
    'A673': 'Bone',
    'CADO-ES1': 'Bone',
    'CAL-72': 'Bone',
    'CAL-78': 'Bone',
    'CHSA0011': 'Bone',
    'CHSA0108': 'Bone',
    'CHSA8926': 'Bone',
    'CS1': 'Bone',
    'ES1': 'Bone',
    'ES3': 'Bone',
    'ES4': 'Bone',
    'ES5': 'Bone',
    'ES6': 'Bone',
    'ES7': 'Bone',
    'ES8': 'Bone',
    'EW-1': 'Bone',
    'EW-11': 'Bone',
    'EW-12': 'Bone',
    'EW-13': 'Bone',
    'EW-16': 'Bone',
    'EW-18': 'Bone',
    'EW-22': 'Bone',
    'EW-24': 'Bone',
    'EW-3': 'Bone',
    'EW-7': 'Bone',
    'G-361': 'Skin',  # Melanoma bone metastasis
    'H-EMC-SS': 'Bone',
    'HOS': 'Bone',
    'HuO-3N1': 'Bone',
    'HuO9': 'Bone',
    'MG-63': 'Bone',
    'MHH-ES-1': 'Bone',
    'NOS-1': 'Bone',
    'NY': 'Bone',
    'SAOS-2': 'Bone',  # Also spelled Saos-2
    'Sarc9371': 'Bone',
    'SK-ES-1': 'Bone',
    'SK-PN-DW': 'Bone',
    'SJSA-1': 'Bone',
    'TASK1': 'Bone',
    'TC-71': 'Bone',
    'U-2-OS': 'Bone',
    'VA-ES-BJ': 'Bone',

    # ===========================================================================
    # BREAST
    # ===========================================================================
    'AU565': 'Breast',
    'BT-20': 'Breast',
    'BT-474': 'Breast',
    'BT-483': 'Breast',
    'BT-549': 'Breast',
    'CAL-120': 'Breast',
    'CAL-148': 'Breast',
    'CAL-51': 'Breast',
    'CAL-85-1': 'Breast',
    'CAMA-1': 'Breast',
    'COLO-824': 'Breast',
    'DU-4475': 'Breast',
    'EFM-19': 'Breast',
    'EFM-192A': 'Breast',
    'EVSA-T': 'Breast',
    'HCC1143': 'Breast',
    'HCC1187': 'Breast',
    'HCC1395': 'Breast',
    'HCC1419': 'Breast',
    'HCC1428': 'Breast',
    'HCC1500': 'Breast',
    'HCC1569': 'Breast',
    'HCC1599': 'Breast',
    'HCC1806': 'Breast',
    'HCC1937': 'Breast',
    'HCC1954': 'Breast',
    'HCC202': 'Breast',
    'HCC2157': 'Breast',
    'HCC2218': 'Breast',
    'HCC38': 'Breast',
    'HCC70': 'Breast',
    'HDQ-P1': 'Breast',
    'Hs-578-T': 'Breast',
    'JIMT-1': 'Breast',
    'MCF7': 'Breast',
    'MDA-MB-157': 'Breast',
    'MDA-MB-175-VII': 'Breast',
    'MDA-MB-231': 'Breast',
    'MDA-MB-330': 'Breast',
    'MDA-MB-361': 'Breast',
    'MDA-MB-415': 'Breast',
    'MDA-MB-453': 'Breast',
    'MDA-MB-468': 'Breast',
    'MFM-223': 'Breast',
    'MRK-nu-1': 'Breast',
    'OCUB-M': 'Breast',
    'T47D': 'Breast',
    'UACC-812': 'Breast',
    'UACC-893': 'Breast',
    'YMB-1-E': 'Breast',
    'ZR-75-1': 'Breast',
    'ZR-75-30': 'Breast',

    # ===========================================================================
    # CENTRAL NERVOUS SYSTEM
    # ===========================================================================
    '42-MG-BA': 'Brain',
    'A172': 'Brain',
    'AM-38': 'Brain',
    'Becker': 'Brain',
    'CAS-1': 'Brain',
    'CCF-STTG1': 'Brain',
    'D-247MG': 'Brain',
    'D-263MG': 'Brain',
    'D-283MED': 'Brain',
    'D-336MG': 'Brain',
    'D-392MG': 'Brain',
    'D-423MG': 'Brain',
    'D-502MG': 'Brain',
    'D-542MG': 'Brain',
    'D-566MG': 'Brain',
    'DBTRG-05MG': 'Brain',
    'DK-MG': 'Brain',
    'Daoy': 'Brain',
    'GAMG': 'Brain',
    'GI-1': 'Brain',
    'GMS-10': 'Brain',
    'H4': 'Brain',
    'KALS-1': 'Brain',
    'KINGS-1': 'Brain',
    'KNS-42': 'Brain',
    'KNS-81-FD': 'Brain',
    'KS-1': 'Brain',
    'LN-18': 'Brain',
    'LN-229': 'Brain',
    'LN-405': 'Brain',
    'LNZTA3WT4': 'Brain',
    'M059J': 'Brain',
    'MOG-G-CCM': 'Brain',
    'MOG-G-UVW': 'Brain',
    'NMC-G1': 'Brain',
    'ONS-76': 'Brain',
    'PFSK-1': 'Brain',
    'SF126': 'Brain',
    'SF268': 'Brain',
    'SF295': 'Brain',
    'SF539': 'Brain',
    'SK-MG-1': 'Brain',
    'SNB75': 'Brain',
    'SW1088': 'Brain',
    'SW1783': 'Brain',
    'T98G': 'Brain',
    'U-118-MG': 'Brain',
    'U-87-MG': 'Brain',
    'U251': 'Brain',
    'YH-13': 'Brain',
    'YKG-1': 'Brain',
    'no-10': 'Brain',
    'no-11': 'Brain',

    # ===========================================================================
    # CERVIX
    # ===========================================================================
    'C-33-A': 'Uterus',
    'C-4-I': 'Uterus',
    'Ca-Ski': 'Uterus',
    'DoTc2-4510': 'Uterus',
    'HT-3': 'Uterus',
    'ME-180': 'Uterus',
    'MS751': 'Uterus',
    'OMC-1': 'Uterus',
    'SISO': 'Uterus',
    'SiHa': 'Uterus',
    'SKG-IIIa': 'Uterus',
    'SW756': 'Uterus',
    'TC-YIK': 'Uterus',
    'HeLa': 'Uterus',

    # ===========================================================================
    # COLORECTAL (LARGE INTESTINE)
    # ===========================================================================
    'C2BBe1': 'Colon',
    'CCK-81': 'Colon',
    'CL-11': 'Colon',
    'CL-34': 'Colon',
    'CL-40': 'Colon',
    'COLO-205': 'Colon',
    'COLO-320-HSR': 'Colon',
    'COLO-678': 'Colon',
    'COLO-741': 'Colon',
    'DIFI': 'Colon',
    'GP5d': 'Colon',
    'HCC-56': 'Colon',
    'HCC2998': 'Colon',
    'HCT-116': 'Colon',
    'HCT-15': 'Colon',
    'HT-115': 'Colon',
    'HT-29': 'Colon',
    'HT55': 'Colon',
    'KM12': 'Colon',
    'LIM1215': 'Colon',
    'LS-1034': 'Colon',
    'LS-123': 'Colon',
    'LS-180': 'Colon',
    'LS-411N': 'Colon',
    'LS-513': 'Colon',
    'LoVo': 'Colon',
    'MDST8': 'Colon',
    'NCI-H508': 'Colon',
    'NCI-H630': 'Colon',
    'NCI-H716': 'Colon',
    'OUMS-23': 'Colon',
    'RCM-1': 'Colon',
    'RKO': 'Colon',
    'SNU-1040': 'Colon',
    'SNU-175': 'Colon',
    'SNU-407': 'Colon',
    'SNU-61': 'Colon',
    'SNU-81': 'Colon',
    'SNU-C1': 'Colon',
    'SNU-C2B': 'Colon',
    'SNU-C5': 'Colon',
    'SW1116': 'Colon',
    'SW1417': 'Colon',
    'SW1463': 'Colon',
    'SW403': 'Colon',
    'SW48': 'Colon',
    'SW620': 'Colon',
    'SW837': 'Colon',
    'SW948': 'Colon',
    'T84': 'Colon',

    # ===========================================================================
    # ENDOMETRIUM
    # ===========================================================================
    'AN3-CA': 'Uterus',
    'COLO-684': 'Uterus',
    'EN': 'Uterus',
    'ESS-1': 'Uterus',
    'HEC-1': 'Uterus',
    'KLE': 'Uterus',
    'MFE-280': 'Uterus',
    'MFE-296': 'Uterus',
    'MFE-319': 'Uterus',
    'RL95-2': 'Uterus',
    'SNG-M': 'Uterus',

    # ===========================================================================
    # ESOPHAGUS
    # ===========================================================================
    'COLO-680N': 'Esophagus',
    'EC-GI-10': 'Esophagus',
    'ECC10': 'Stomach',  # Esophageal cancer cell line
    'ECC12': 'Stomach',  # Esophageal cancer cell line
    'ESO26': 'Esophagus',
    'ESO51': 'Esophagus',
    'FLO-1': 'Esophagus',
    'HCE-4': 'Esophagus',
    'KYAE-1': 'Esophagus',
    'KYSE-140': 'Esophagus',
    'KYSE-150': 'Esophagus',
    'KYSE-180': 'Esophagus',
    'KYSE-220': 'Esophagus',
    'KYSE-270': 'Esophagus',
    'KYSE-30': 'Esophagus',
    'KYSE-410': 'Esophagus',
    'KYSE-450': 'Esophagus',
    'KYSE-50': 'Esophagus',
    'KYSE-510': 'Esophagus',
    'KYSE-520': 'Esophagus',
    'KYSE-70': 'Esophagus',
    'OACM5-1': 'Esophagus',
    'OACp4C': 'Esophagus',
    'OE19': 'Esophagus',
    'OE21': 'Esophagus',
    'OE33': 'Esophagus',
    'SK-GT-4': 'Esophagus',
    'T-T': 'Esophagus',
    'TE-1': 'Esophagus',
    'TE-10': 'Esophagus',
    'TE-11': 'Esophagus',
    'TE-12': 'Esophagus',
    'TE-15': 'Esophagus',
    'TE-4': 'Esophagus',
    'TE-441-T': 'Soft_Tissue',
    'TE-5': 'Esophagus',
    'TE-6': 'Esophagus',
    'TE-8': 'Esophagus',
    'TE-9': 'Esophagus',

    # ===========================================================================
    # GASTRIC (STOMACH) - Primary focus
    # ===========================================================================
    '23132-87': 'Stomach',
    'AGS': 'Stomach',
    'AZ521': 'Stomach',
    'ECC10': 'Stomach',
    'ECC12': 'Stomach',
    'FU97': 'Stomach',
    'GCIY': 'Stomach',
    'HGC-27': 'Stomach',
    'HSC-39': 'Stomach',
    'HSC-43': 'Stomach',
    'HSC-44': 'Stomach',
    'Hs746T': 'Stomach',
    'IM-95': 'Stomach',
    'KATOIII': 'Stomach',
    'MKN1': 'Stomach',
    'MKN28': 'Stomach',
    'MKN45': 'Stomach',
    'MKN7': 'Stomach',
    'NCI-N87': 'Stomach',
    'NCI-SNU-1': 'Stomach',
    'NCI-SNU-16': 'Stomach',
    'NCI-SNU-5': 'Stomach',
    'NUGC-2': 'Stomach',
    'NUGC-3': 'Stomach',
    'NUGC-4': 'Stomach',
    'OCUM-1': 'Stomach',
    'RF-48': 'Stomach',
    'RERF-GC-1B': 'Stomach',
    'SCH': 'Stomach',
    'SK-GT-2': 'Stomach',
    'SNU-1': 'Stomach',
    'SNU-5': 'Stomach',
    'SNU-16': 'Stomach',
    'SNU-216': 'Stomach',
    'SNU-484': 'Stomach',
    'SNU-601': 'Stomach',
    'SNU-638': 'Stomach',
    'SNU-668': 'Stomach',
    'SNU-719': 'Stomach',
    'TGBC11TKB': 'Stomach',
    'TMK-1': 'Stomach',

    # ===========================================================================
    # HEMATOPOIETIC AND LYMPHOID TISSUE (Blood cancers)
    # ===========================================================================
    'ALL-PO': 'Blood',
    'ALL-SIL': 'Blood',
    'AMO-1': 'Blood',
    'ARH-77': 'Blood',
    'ATN-1': 'Blood',
    'A3-KAW': 'Blood',
    'BALL-1': 'Blood',
    'BC-1': 'Blood',
    'BC-3': 'Blood',
    'BE-13': 'Blood',
    'BL-41': 'Blood',
    'BL-70': 'Blood',
    'BV-173': 'Blood',
    'CA46': 'Blood',
    'CCRF-CEM': 'Blood',
    'CESS': 'Blood',
    'CML-T1': 'Blood',
    'CMK': 'Blood',
    'COR-L321': 'Blood',
    'CRO-AP2': 'Blood',
    'CTB-1': 'Blood',
    'CTV-1': 'Blood',
    'DB': 'Blood',
    'DEL': 'Blood',
    'DG-75': 'Blood',
    'DND-41': 'Blood',
    'DOHH-2': 'Blood',
    'Daudi': 'Blood',
    'EB-3': 'Blood',
    'EB2': 'Blood',
    'EHEB': 'Blood',
    'EJM': 'Blood',
    'EM-2': 'Blood',
    'FARAGE': 'Blood',  # Also spelled Farage
    'GA-10': 'Blood',
    'GDM-1': 'Blood',
    'GR-ST': 'Blood',
    'GRANTA-519': 'Blood',
    'H9': 'Blood',
    'HAL-01': 'Blood',
    'HC-1': 'Blood',
    'HD-MY-Z': 'Blood',
    'HDLM-2': 'Blood',
    'HH': 'Blood',
    'HL-60': 'Blood',
    'Hs-445': 'Blood',
    'HT': 'Blood',
    'HEL': 'Blood',
    'IM-9': 'Blood',
    'JJN-3': 'Blood',
    'JM1': 'Blood',
    'JSC-1': 'Blood',
    'K-562': 'Blood',
    'KARPAS-1106P': 'Blood',
    'KARPAS-231': 'Blood',
    'KARPAS-299': 'Blood',
    'KARPAS-422': 'Blood',
    'KARPAS-45': 'Blood',
    'KARPAS-620': 'Blood',
    'KASUMI-1': 'Blood',
    'KCL-22': 'Blood',
    'KE-37': 'Blood',
    'KG-1': 'Blood',
    'KM-H2': 'Blood',
    'KMOE-2': 'Blood',
    'KMS-11': 'Blood',
    'KMS-12-BM': 'Blood',
    'KOPN-8': 'Blood',
    'KU812': 'Blood',
    'KY821': 'Blood',
    'L-1236': 'Blood',
    'L-363': 'Blood',
    'L-428': 'Blood',
    'L-540': 'Blood',
    'LAMA-84': 'Blood',
    'LC4-1': 'Blood',
    'LCLC-103H': 'Lung',  # Actually lung, but classified as lymphoma
    'LOU-NH91': 'Lung',
    'LOUCY': 'Blood',
    'LP-1': 'Blood',
    'LUC-1': 'Lung',  # Lung neuroendocrine
    'LUL-1': 'Lung',
    'M0-91': 'Blood',
    'MC-1010': 'Blood',
    'MC116': 'Blood',
    'ME-1': 'Blood',
    'MEG-01': 'Blood',
    'ML-2': 'Blood',
    'MLMA': 'Blood',
    'MM1S': 'Blood',
    'MN-60': 'Blood',
    'MOLM-13': 'Blood',
    'MOLM-16': 'Blood',
    'MOLP-8': 'Blood',
    'MOLT-13': 'Blood',
    'MOLT-16': 'Blood',
    'MOLT-4': 'Blood',
    'MONO-MAC-6': 'Blood',
    'Mo-T': 'Blood',
    'MV-4-11': 'Blood',
    'MY-M12': 'Blood',
    'NALM-6': 'Blood',
    'NAMALWA': 'Blood',
    'NCI-H929': 'Blood',
    'NH-12': 'Neuroblastoma',  # M6 fix: was 'Other', now correctly mapped
    'NK-92MI': 'Blood',
    'NKM-1': 'Blood',
    'NOMO-1': 'Blood',
    'NU-DUL-1': 'Blood',
    'OPM-2': 'Blood',
    'P12-ICHIKAWA': 'Blood',
    'P30-OHK': 'Blood',
    'P31-FUJ': 'Blood',
    'P32-ISH': 'Blood',
    'PF-382': 'Blood',
    'PL-21': 'Blood',
    'QIMR-WIL': 'Blood',
    'Raji': 'Blood',
    'Ramos-2G6-4C10': 'Blood',
    'RC-K8': 'Blood',
    'RCH-ACV': 'Blood',
    'RD': 'Soft_Tissue',  # Rhabdomyosarcoma
    'REH': 'Blood',
    'RL': 'Blood',
    'ROS-50': 'Blood',
    'RPMI-2650': 'Head_Neck',  # Actually skin, but classified elsewhere
    'RPMI-6666': 'Blood',
    'RPMI-8226': 'Blood',
    'RPMI-8402': 'Blood',
    'RPMI-8866': 'Blood',
    'RS4-11': 'Blood',
    'SCC-3': 'Blood',
    'Set2': 'Blood',
    'SIG-M5': 'Blood',
    'SK-MM-2': 'Blood',
    'SR': 'Blood',
    'ST486': 'Blood',
    'SU-DHL-1': 'Blood',
    'SU-DHL-10': 'Blood',
    'SU-DHL-16': 'Blood',
    'SU-DHL-4': 'Blood',
    'SU-DHL-5': 'Blood',
    'SU-DHL-6': 'Blood',
    'SU-DHL-8': 'Other',
    'SUP-B15': 'Blood',
    'SUP-B8': 'Blood',
    'SUP-HD1': 'Blood',
    'SUP-M2': 'Blood',
    'SUP-T1': 'Blood',
    'TALL-1': 'Blood',
    'THP-1': 'Blood',
    'TK': 'Blood',
    'TUR': 'Blood',
    'U-266': 'Blood',
    'U-698-M': 'Blood',
    'VAL': 'Blood',
    'WIL2-NS': 'Blood',
    'WSU-DLCL2': 'Blood',
    'WSU-NHL': 'Blood',
    'YT': 'Blood',

    # ===========================================================================
    # KIDNEY
    # ===========================================================================
    '639-V': 'Bladder',
    '647-V': 'Bladder',
    '769-P': 'Kidney',
    '786-0': 'Kidney',
    'A498': 'Kidney',
    'A704': 'Kidney',
    'ACHN': 'Kidney',
    'BFTC-909': 'Kidney',
    'CAL-54': 'Kidney',
    'CAKI-1': 'Kidney',
    'Caki-1': 'Kidney',
    'Caki-2': 'Kidney',
    'HA7-RCC': 'Kidney',
    'LB1047-RCC': 'Kidney',
    'LB2241-RCC': 'Kidney',
    'LB996-RCC': 'Kidney',
    'NCC010': 'Kidney',
    'NCC021': 'Kidney',
    'OS-RC-2': 'Kidney',
    'RCC-AB': 'Kidney',
    'RCC-ER': 'Kidney',
    'RCC-FG2': 'Kidney',
    'RCC-JF': 'Kidney',
    'RCC-JW': 'Kidney',
    'RCC-MF': 'Kidney',
    'RCC10RGB': 'Kidney',
    'RXF393': 'Kidney',
    'SK-NEP-1': 'Kidney',
    'SN12C': 'Kidney',
    'SW156': 'Kidney',
    'TK10': 'Kidney',
    'U031': 'Kidney',
    'VMRC-RCW': 'Kidney',
    'VMRC-RCZ': 'Kidney',

    # ===========================================================================
    # LARGE INTESTINE - See Colon above
    # ===========================================================================

    # ===========================================================================
    # LIVER
    # ===========================================================================
    'C3A': 'Liver',
    'HLE': 'Liver',
    'Huh7': 'Liver',
    'HUH-6-clone5': 'Liver',
    'Hep3B': 'Liver',
    'HepG2': 'Liver',
    'HuH-7': 'Liver',
    'JHH-1': 'Liver',
    'JHH-2': 'Liver',
    'JHH-4': 'Liver',
    'JHH-6': 'Liver',
    'JHH-7': 'Liver',
    'MHCC97H': 'Liver',
    'MHCC97L': 'Liver',
    'PLC/PRF/5': 'Liver',
    'SNU-182': 'Liver',
    'SNU-387': 'Liver',
    'SNU-398': 'Liver',
    'SNU-423': 'Liver',
    'SNU-449': 'Liver',
    'SNU-475': 'Liver',
    'SK-HEP-1': 'Liver',
    'huH-1': 'Liver',

    # ===========================================================================
    # LUNG
    # ===========================================================================
    '201T': 'Lung',
    'ABC-1': 'Lung',
    'A427': 'Lung',
    'BEN': 'Lung',
    'CAL-12T': 'Lung',
    'COLO-668': 'Lung',
    'COR-L105': 'Lung',
    'COR-L23': 'Lung',
    'COR-L279': 'Lung',
    'COR-L303': 'Lung',
    'COR-L311': 'Lung',
    'COR-L32': 'Lung',
    'COR-L88': 'Lung',
    'COR-L95': 'Lung',
    'CPC-N': 'Lung',
    'DMS-114': 'Lung',
    'DMS-273': 'Lung',
    'DMS-53': 'Lung',
    'DMS-79': 'Lung',
    'EBC-1': 'Lung',
    'EKVX': 'Lung',
    'EMC-BAC-1': 'Lung',
    'EMC-BAC-2': 'Lung',
    'EPLC-272H': 'Lung',
    'HARA': 'Lung',
    'HCC-15': 'Lung',
    'HCC-33': 'Lung',
    'HCC-44': 'Lung',
    'HCC-78': 'Lung',
    'HCC827': 'Lung',
    'H1299': 'Lung',
    'H1650': 'Lung',
    'H1666': 'Lung',
    'H1693': 'Lung',
    'H1694': 'Lung',
    'H1755': 'Lung',
    'H1975': 'Lung',
    'H1993': 'Lung',
    'H209': 'Lung',
    'H211': 'Lung',
    'H2122': 'Lung',
    'H2126': 'Lung',
    'H2135': 'Lung',
    'H2141': 'Lung',
    'H2228': 'Lung',
    'H226': 'Lung',
    'H23': 'Lung',
    'H2342': 'Lung',
    'H2347': 'Lung',
    'H2405': 'Lung',
    'H2444': 'Lung',
    'H2452': 'Pleura',
    'H250': 'Lung',
    'H28': 'Pleura',
    'H292': 'Lung',
    'H3122': 'Lung',
    'H322M': 'Lung',
    'H345': 'Lung',
    'H358': 'Lung',
    'H378': 'Lung',
    'H460': 'Lung',
    'H522': 'Lung',
    'H513': 'Pleura',
    'H520': 'Lung',
    'H524': 'Lung',
    'H526': 'Lung',
    'H596': 'Lung',
    'H64': 'Lung',
    'H647': 'Lung',
    'H650': 'Lung',
    'H661': 'Lung',
    'H69': 'Lung',
    'H716': 'Colon',
    'H720': 'Lung',
    'H727': 'Lung',
    'H748': 'Lung',
    'H810': 'Lung',
    'H82': 'Lung',
    'H835': 'Lung',
    'H838': 'Lung',
    'H841': 'Lung',
    'H847': 'Lung',
    'HOP-62': 'Lung',
    'HOP-92': 'Lung',
    'IA-LM': 'Lung',
    'KNS-62': 'Lung',
    'KS-1': 'Brain',
    'LB647-SCLC': 'Lung',
    'LC-1-sq': 'Lung',
    'LC-2-ad': 'Lung',
    'LCLC-97TM1': 'Lung',
    'LK-2': 'Lung',
    'LOU-NH91': 'Lung',
    'LU-134-A': 'Lung',
    'LU-135': 'Lung',
    'LU-139': 'Lung',
    'LU-165': 'Lung',
    'LU-65': 'Lung',
    'LU-99A': 'Lung',
    'LXF-289': 'Lung',
    'MS-1': 'Lung',
    'NCI-H1048': 'Lung',
    'NCI-H1092': 'Lung',
    'NCI-H1105': 'Lung',
    'NCI-H1155': 'Lung',
    'NCI-H128': 'Lung',
    'NCI-H1299': 'Lung',
    'NCI-H1304': 'Lung',
    'NCI-H1341': 'Lung',
    'NCI-H1355': 'Lung',
    'NCI-H1395': 'Lung',
    'NCI-H1417': 'Lung',
    'NCI-H1435': 'Lung',
    'NCI-H1436': 'Lung',
    'NCI-H1437': 'Lung',
    'NCI-H146': 'Lung',
    'NCI-H1563': 'Lung',
    'NCI-H1568': 'Lung',
    'NCI-H1573': 'Lung',
    'NCI-H1581': 'Lung',
    'NCI-H1623': 'Lung',
    'NCI-H1648': 'Lung',
    'NCI-H1650': 'Lung',
    'NCI-H1651': 'Lung',
    'NCI-H1666': 'Lung',
    'NCI-H1688': 'Lung',
    'NCI-H1693': 'Lung',
    'NCI-H1694': 'Lung',
    'NCI-H1703': 'Lung',
    'NCI-H1734': 'Lung',
    'NCI-H1755': 'Lung',
    'NCI-H1770': 'Lung',
    'NCI-H1781': 'Lung',
    'NCI-H1792': 'Lung',
    'NCI-H1793': 'Lung',
    'NCI-H1836': 'Lung',
    'NCI-H1838': 'Lung',
    'NCI-H1869': 'Lung',
    'NCI-H187': 'Lung',
    'NCI-H1876': 'Lung',
    'NCI-H1915': 'Lung',
    'NCI-H1944': 'Lung',
    'NCI-H196': 'Lung',
    'NCI-H1963': 'Lung',
    'NCI-H1975': 'Lung',
    'NCI-H1993': 'Lung',
    'NCI-H2009': 'Lung',
    'NCI-H2023': 'Lung',
    'NCI-H2029': 'Lung',
    'NCI-H2030': 'Lung',
    'NCI-H2052': 'Pleura',
    'NCI-H2066': 'Lung',
    'NCI-H2081': 'Lung',
    'NCI-H2085': 'Lung',
    'NCI-H2087': 'Lung',
    'NCI-H209': 'Lung',
    'NCI-H211': 'Lung',
    'NCI-H2110': 'Lung',
    'NCI-H2122': 'Lung',
    'NCI-H2126': 'Lung',
    'NCI-H2135': 'Lung',
    'NCI-H2141': 'Lung',
    'NCI-H2170': 'Lung',
    'NCI-H2171': 'Lung',
    'NCI-H2172': 'Lung',
    'NCI-H2196': 'Lung',
    'NCI-H2227': 'Lung',
    'NCI-H2228': 'Lung',
    'NCI-H226': 'Lung',
    'NCI-H2291': 'Lung',
    'NCI-H23': 'Lung',
    'NCI-H2342': 'Lung',
    'NCI-H2347': 'Lung',
    'NCI-H2405': 'Lung',
    'NCI-H2444': 'Lung',
    'NCI-H2452': 'Pleura',
    'NCI-H250': 'Lung',
    'NCI-H28': 'Pleura',
    'NCI-H292': 'Lung',
    'NCI-H3122': 'Lung',
    'NCI-H322M': 'Lung',
    'NCI-H345': 'Lung',
    'NCI-H358': 'Lung',
    'NCI-H378': 'Lung',
    'NCI-H441': 'Lung',
    'NCI-H446': 'Lung',
    'NCI-H460': 'Lung',
    'NCI-H508': 'Colon',
    'NCI-H510A': 'Lung',
    'NCI-H520': 'Lung',
    'NCI-H522': 'Lung',
    'NCI-H524': 'Lung',
    'NCI-H526': 'Lung',
    'NCI-H596': 'Lung',
    'NCI-H630': 'Colon',
    'NCI-H64': 'Lung',
    'NCI-H647': 'Lung',
    'NCI-H650': 'Lung',
    'NCI-H661': 'Lung',
    'NCI-H69': 'Lung',
    'NCI-H716': 'Colon',
    'NCI-H720': 'Lung',
    'NCI-H727': 'Lung',
    'NCI-H747': 'Colon',
    'NCI-H748': 'Lung',
    'NCI-H810': 'Lung',
    'NCI-H82': 'Lung',
    'NCI-H835': 'Lung',
    'NCI-H838': 'Lung',
    'NCI-H841': 'Lung',
    'NCI-H847': 'Lung',
    'PC-14': 'Lung',
    'PC-3_[JPC-3]': 'Lung',
    'PC-9': 'Lung',
    'RERF-LC-KJ': 'Lung',
    'RERF-LC-MS': 'Lung',
    'RERF-LC-Sq1': 'Lung',
    'SBC-1': 'Lung',
    'SBC-3': 'Lung',
    'SBC-5': 'Lung',
    'SHP-77': 'Lung',
    'SK-LU-1': 'Lung',
    'SK-MES-1': 'Lung',
    'SW1271': 'Lung',
    'SW1573': 'Lung',
    'SW900': 'Lung',
    'UMC-11': 'Lung',
    'VMRC-LCD': 'Lung',

    # ===========================================================================
    # OEOPHAGUS - See Esophagus above
    # ===========================================================================

    # ===========================================================================
    # OVARY
    # ===========================================================================
    'A2780': 'Ovary',
    'Caov-3': 'Ovary',
    'Caov-4': 'Ovary',
    'DOV13': 'Ovary',
    'EFO-21': 'Ovary',
    'EFO-27': 'Ovary',
    'ES-2': 'Ovary',
    'FU-OV-1': 'Ovary',
    'IGROV-1': 'Ovary',
    'JHOS-2': 'Ovary',
    'JHOS-3': 'Ovary',
    'JHOS-4': 'Ovary',
    'KGN': 'Ovary',
    'KURAMOCHI': 'Ovary',
    'MCAS': 'Ovary',
    'OAW-28': 'Ovary',
    'OAW-42': 'Ovary',
    'OC-314': 'Ovary',
    'OV-17R': 'Ovary',
    'OV-56': 'Ovary',
    'OV-7': 'Ovary',
    'OV-90': 'Ovary',
    'OVCA420': 'Ovary',
    'OVCA433': 'Ovary',
    'OVCAR-3': 'Ovary',
    'OVCAR-4': 'Ovary',
    'OVCAR-5': 'Ovary',
    'OVCAR-8': 'Ovary',
    'OVISE': 'Ovary',
    'OVK-18': 'Ovary',
    'OVKATE': 'Ovary',
    'OVMIU': 'Ovary',
    'OVTOKO': 'Ovary',
    'PA-1': 'Ovary',
    'PEO1': 'Ovary',
    'RKN': 'Ovary',
    'RMG-I': 'Ovary',
    'SK-OV-3': 'Ovary',
    'SW626': 'Ovary',
    'TOV-112D': 'Ovary',
    'TOV-21G': 'Ovary',
    'TYK-nu': 'Ovary',

    # ===========================================================================
    # PANCREAS
    # ===========================================================================
    'AsPC-1': 'Pancreas',
    'BxPC-3': 'Pancreas',
    'Capan-1': 'Pancreas',
    'Capan-2': 'Pancreas',
    'CAPAN-1': 'Pancreas',
    'CFPAC-1': 'Pancreas',
    'DAN-G': 'Pancreas',
    'HPAC': 'Pancreas',
    'HPAF-II': 'Pancreas',
    'HuP-T3': 'Pancreas',
    'HuP-T4': 'Pancreas',
    'KP-1N': 'Pancreas',
    'KP-2': 'Pancreas',
    'KP-3': 'Pancreas',
    'KP-4': 'Pancreas',
    'MIA-PaCa-2': 'Pancreas',
    'MIA PaCa-2': 'Pancreas',
    'MZ1-PC': 'Pancreas',
    'PANC-02-03': 'Pancreas',
    'PANC-03-27': 'Pancreas',
    'PANC-04-03': 'Pancreas',
    'PANC-08-13': 'Pancreas',
    'PANC-10-05': 'Pancreas',
    'PANC-1': 'Pancreas',
    'PA-TU-8902': 'Pancreas',
    'PA-TU-8988T': 'Pancreas',
    'PL18': 'Pancreas',
    'PL4': 'Pancreas',
    'PSN1': 'Pancreas',
    'QGP-1': 'Pancreas',
    'SU8686': 'Pancreas',
    'SUIT-2': 'Pancreas',
    'SW1990': 'Pancreas',
    'YAPC': 'Pancreas',

    # ===========================================================================
    # PLACENTA
    # ===========================================================================
    'JAR': 'Other',
    'JEG-3': 'Other',

    # ===========================================================================
    # PLEURA
    # ===========================================================================
    'H2369': 'Lung',
    'H2373': 'Lung',
    'H2461': 'Lung',
    'H2591': 'Lung',
    'H2595': 'Lung',
    'H2722': 'Lung',
    'H2731': 'Lung',
    'H2795': 'Lung',
    'H2803': 'Lung',
    'H2804': 'Lung',
    'H2810': 'Lung',
    'H2818': 'Lung',
    'H2869': 'Lung',
    'H290': 'Lung',
    'H513': 'Lung',
    'IST-MES1': 'Lung',
    'MPP-89': 'Lung',
    'MSTO-211H': 'Lung',
    'NCI-H2052': 'Lung',
    'NCI-H2452': 'Lung',
    'NCI-H28': 'Lung',

    # ===========================================================================
    # PROSTATE
    # ===========================================================================
    '22RV1': 'Prostate',
    'BPH-1': 'Prostate',
    'DU-145': 'Prostate',
    'LNCaP-Clone-FGC': 'Prostate',
    'PC-3': 'Prostate',
    'PWR-1E': 'Prostate',
    'VCaP': 'Prostate',

    # ===========================================================================
    # SALIVARY GLAND
    # ===========================================================================
    'A253': 'Head_Neck',
    'HO-1-N-1': 'Head_Neck',

    # ===========================================================================
    # SKIN (including melanoma)
    # ===========================================================================
    '101T': 'Skin',
    '451Lu': 'Skin',
    'A101D': 'Skin',
    'A2058': 'Skin',
    'A375': 'Skin',
    'A388': 'Skin',
    'A431': 'Skin',
    'C32': 'Skin',
    'CHL-1': 'Skin',
    'COLO-679': 'Skin',
    'COLO-783': 'Skin',
    'COLO-792': 'Skin',
    'COLO-800': 'Skin',
    'CP50-MEL-B': 'Skin',
    'CP66-MEL': 'Skin',
    'CP67-MEL': 'Skin',
    'DJM-1': 'Skin',
    'G-361': 'Skin',
    'GAK': 'Skin',
    'G-MEL': 'Skin',
    'HMV-II': 'Skin',
    'HT-144': 'Skin',
    'IGR-1': 'Skin',
    'IGR-37': 'Skin',
    'IPC-298': 'Skin',
    'K2': 'Skin',
    'LB2518-MEL': 'Skin',
    'LB373-MEL-D': 'Skin',
    'LOXIMVI': 'Skin',
    'M14': 'Skin',
    'MEL-HO': 'Skin',
    'MEL-JUSO': 'Skin',
    'MMAC-SF': 'Skin',
    'RPMI-7951': 'Skin',
    'RVH-421': 'Skin',
    'SH-4': 'Skin',
    'SK-MEL-1': 'Skin',
    'SK-MEL-2': 'Skin',
    'SK-MEL-24': 'Skin',
    'SK-MEL-28': 'Skin',
    'SK-MEL-3': 'Skin',
    'SK-MEL-30': 'Skin',
    'SK-MEL-31': 'Skin',
    'SK-MEL-5': 'Skin',
    'UACC-257': 'Skin',
    'UACC-62': 'Skin',
    'VMRC-MELG': 'Skin',
    'WM-115': 'Skin',
    'WM1552C': 'Skin',
    'WM278': 'Skin',
    'WM35': 'Skin',
    'WM793B': 'Skin',
    'Mewo': 'Skin',
    'MZ2-MEL': 'Skin',
    'MZ7-mel': 'Skin',

    # ===========================================================================
    # SMALL INTESTINE
    # ===========================================================================
    'HuTu-80': 'Small_Intestine',

    # ===========================================================================
    # SOFT TISSUE
    # ===========================================================================
    'A204': 'Soft_Tissue',
    'G-401': 'Soft_Tissue',
    'G-402': 'Soft_Tissue',
    'GCT': 'Soft_Tissue',
    'HT-1080': 'Soft_Tissue',
    'MES-SA': 'Soft_Tissue',
    'MFH-ino': 'Soft_Tissue',
    'RH-1': 'Soft_Tissue',
    'RH-18': 'Soft_Tissue',
    'RH-41': 'Soft_Tissue',
    'S-117': 'Soft_Tissue',
    'SJRH30': 'Soft_Tissue',
    'SK-LMS-1': 'Soft_Tissue',
    'SK-UT-1': 'Soft_Tissue',
    'SKN': 'Soft_Tissue',
    'STS-0421': 'Soft_Tissue',
    'SW684': 'Soft_Tissue',
    'SW872': 'Soft_Tissue',
    'SW982': 'Soft_Tissue',
    'TE-441-T': 'Soft_Tissue',

    # ===========================================================================
    # STOMACH - See Gastric above
    # ===========================================================================

    # ===========================================================================
    # TESTIS
    # ===========================================================================
    'NEC8': 'Other',

    # ===========================================================================
    # THYROID
    # ===========================================================================
    '8305C': 'Thyroid',
    '8505C': 'Thyroid',
    'ASH-3': 'Thyroid',
    'BCPAP': 'Thyroid',
    'BHT-101': 'Thyroid',
    'CAL-62': 'Thyroid',
    'CGTH-W-1': 'Thyroid',
    'FTC-133': 'Thyroid',
    'HTC-C3': 'Thyroid',
    'IHH-4': 'Thyroid',
    'K5': 'Thyroid',
    'KMH-2': 'Thyroid',
    'ML-1': 'Thyroid',
    'RO82-W-1': 'Thyroid',
    'TT': 'Thyroid',
    'TT2609-C02': 'Thyroid',

    # ===========================================================================
    # UPPER AERODIGESTIVE TRACT
    # ===========================================================================
    'BB30-HNC': 'Head_Neck',
    'BB49-HNC': 'Head_Neck',
    'BHY': 'Head_Neck',
    'BICR10': 'Head_Neck',
    'BICR22': 'Head_Neck',
    'BICR31': 'Head_Neck',
    'BICR78': 'Head_Neck',
    'CAL-27': 'Head_Neck',
    'CAL-33': 'Head_Neck',
    'DOK': 'Head_Neck',
    'FADU': 'Head_Neck',
    'H3118': 'Head_Neck',
    'HSC-2': 'Head_Neck',
    'HSC-3': 'Head_Neck',
    'HSC-4': 'Head_Neck',
    'HN': 'Head_Neck',
    'HO-1-u-1': 'Head_Neck',
    'KON': 'Head_Neck',
    'KOSC-2': 'Head_Neck',
    'LB771-HNC': 'Head_Neck',
    'OSC-19': 'Head_Neck',
    'OSC-20': 'Head_Neck',
    'PCI-15A': 'Head_Neck',
    'PCI-30': 'Head_Neck',
    'PCI-38': 'Head_Neck',
    'PCI-4B': 'Head_Neck',
    'PCI-6A': 'Head_Neck',
    'RPMI-2650': 'Head_Neck',
    'SAS': 'Head_Neck',
    'SAT': 'Head_Neck',
    'SCC-15': 'Head_Neck',
    'SCC-25': 'Head_Neck',
    'SCC-4': 'Head_Neck',
    'SCC-9': 'Head_Neck',
    'SCC90': 'Head_Neck',
    'SKN-3': 'Head_Neck',
    'UDSCC2': 'Head_Neck',
    'Ca9-22': 'Head_Neck',
    'Detroit562': 'Head_Neck',

    # ===========================================================================
    # URINARY TRACT
    # ===========================================================================
    'BFTC-905': 'Bladder',
    'CAL-29': 'Bladder',
    # L1 fix: removed duplicate 'CAL-39' entry (was here under Urinary Tract AND in Vulva section)
    # Canonical entry is in the VULVA section below (CAL-39 is a vulvar carcinoma, not bladder)
    'DSH1': 'Bladder',
    'J82': 'Bladder',
    'RT-112': 'Bladder',
    'RT4': 'Bladder',
    'SCaBER': 'Bladder',
    'SW1710': 'Bladder',
    'SW780': 'Bladder',
    'T-24': 'Bladder',
    'TCCSUP': 'Bladder',
    'UM-UC-3': 'Bladder',
    'VM-CUB-1': 'Bladder',

    # ===========================================================================
    # UTERUS - See Cervix and Endometrium above
    # ===========================================================================

    # ===========================================================================
    # VULVA
    # ===========================================================================
    'CAL-39': 'Uterus',
    'SW954': 'Uterus',
    'SW962': 'Uterus',

    # ===========================================================================
    # UNKNOWN / OTHER
    # ===========================================================================
    'COR-L321': 'Other',
    'CRO-AP3': 'Other',
    'SU-DHL-8': 'Other',

    # ===========================================================================
    # BRAIN
    # ===========================================================================
    '8-MG-BA': 'Brain',
    'A4-Fuk': 'Brain',
    'GB-1': 'Brain',

    # ===========================================================================
    # LUNG
    # ===========================================================================
    'A549': 'Lung',
    'Calu-3': 'Lung',
    'Calu-6': 'Lung',
    'H3255': 'Lung',
    'HCC-827': 'Lung',
    'HT-1197': 'Lung',

    # ===========================================================================
    # KIDNEY
    # ===========================================================================
    'BB65-RCC': 'Kidney',
    'KMRC-1': 'Kidney',
    'KMRC-20': 'Kidney',

    # ===========================================================================
    # SKIN
    # ===========================================================================
    'COLO-829': 'Skin',
    'IST-MEL1': 'Skin',
    'MCC13': 'Skin',
    'MCC26': 'Skin',
    'SKM-1': 'Skin',

    # ===========================================================================
    # COLON
    # ===========================================================================
    'CW-2': 'Colon',
    'CaR-1': 'Colon',
    'SK-CO-1': 'Colon',

    # ===========================================================================
    # BLOOD
    # ===========================================================================
    'ChaGo-K-1': 'Blood',
    'EoL-1-cell': 'Blood',
    'Farage': 'Blood',
    'IST-SL1': 'Blood',
    'IST-SL2': 'Blood',
    'JEKO-1': 'Blood',
    'JURL-MK1': 'Blood',
    'JVM-2': 'Blood',
    'JVM-3': 'Blood',
    'JiyoyeP-2003': 'Blood',
    'Jurkat': 'Blood',
    'MC-CAR': 'Blood',
    'MHH-CALL-2': 'Blood',
    'MHH-PREB-1': 'Blood',
    'OCI-AML2': 'Blood',
    'OCI-AML3': 'Blood',
    'OCI-AML5': 'Blood',
    'OCI-LY-19': 'Blood',
    'OCI-LY7': 'Blood',
    'OCI-M1': 'Blood',
    'SLVL': 'Blood',

    # ===========================================================================
    # BREAST
    # ===========================================================================
    'Geo': 'Breast',
    'MDA-MB-436': 'Breast',

    # ===========================================================================
    # BLADDER
    # ===========================================================================
    'HT-1376': 'Bladder',
    'KU-19-19': 'Bladder',
    'LB831-BLC': 'Bladder',

    # ===========================================================================
    # OVARY
    # ===========================================================================
    'Hey': 'Ovary',

    # ===========================================================================
    # PROSTATE
    # ===========================================================================
    'PC-3_[JPC-3]': 'Prostate',

    # ===========================================================================
    # STOMACH
    # ===========================================================================
    'KYM-1': 'Stomach',

    # ===========================================================================
    # HEAD_NECK
    # ===========================================================================
    'JHU-011': 'Head_Neck',
    'JHU-022': 'Head_Neck',
    'JHU-029': 'Head_Neck',

    # ===========================================================================
    # NEUROBLASTOMA
    # ===========================================================================
    'NB(TU)1-10': 'Neuroblastoma',

    # ===========================================================================
    # BONE
    # ===========================================================================
    'Saos-2': 'Bone',
}


def get_tissue_id_for_cellline(cellline_name: str) -> int:
    """
    Get tissue type ID for a cell-line.

    Args:
        cellline_name: Name of the cell-line

    Returns:
        Integer tissue type ID, or ID for 'Other' if not found
    """
    # Normalize name (uppercase, remove common variations)
    normalized = cellline_name.upper().replace('-', '').replace('_', '').replace(' ', '')

    # Try exact match first
    if cellline_name in CELLLINE_TO_TISSUE:
        tissue = CELLLINE_TO_TISSUE[cellline_name]
        return TISSUE_TO_ID.get(tissue, TISSUE_TO_ID['Other'])

    # Try normalized match
    for cl, tissue in CELLLINE_TO_TISSUE.items():
        if cl.upper().replace('-', '').replace('_', '') == normalized:
            return TISSUE_TO_ID.get(tissue, TISSUE_TO_ID['Other'])

    logger.warning(f"Cell-line '{cellline_name}' not found in mapping, using 'Other'")
    return TISSUE_TO_ID['Other']


def create_cellline_to_tissue_csv(
    cellline_ids: List[str],
    output_path: str
) -> pd.DataFrame:
    """
    Create a CSV mapping cell-lines to tissues based on the CELLLINE_TO_TISSUE dict.

    Args:
        cellline_ids: List of cell-line IDs from your IC50 data
        output_path: Path to save the mapping CSV

    Returns:
        DataFrame with cellline_id, tissue_type, tissue_id columns
    """
    data = []
    missing = []

    for cl_id in cellline_ids:
        if cl_id in CELLLINE_TO_TISSUE:
            tissue = CELLLINE_TO_TISSUE[cl_id]
            tissue_id = TISSUE_TO_ID.get(tissue, TISSUE_TO_ID['Other'])
        else:
            tissue = 'Other'
            tissue_id = TISSUE_TO_ID['Other']
            missing.append(cl_id)

        data.append({
            'cellline_id': cl_id,
            'tissue_type': tissue,
            'tissue_id': tissue_id
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    if missing:
        logger.warning(
            f"{len(missing)} cell-lines not found in mapping. "
            f"First 10: {missing[:10]}"
        )

    return df


# =============================================================================
# EMBEDDING UTILITIES
# =============================================================================

def normalize_embeddings(embeddings: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 normalize embeddings."""
    return torch.nn.functional.normalize(embeddings, p=2, dim=dim)


def compute_similarity_matrix(
    embeds_a: torch.Tensor,
    embeds_b: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute cosine similarity matrix between two sets of embeddings.

    Args:
        embeds_a: [N, D] embeddings
        embeds_b: [M, D] embeddings
        normalize: Whether to L2 normalize

    Returns:
        [N, M] similarity matrix
    """
    if normalize:
        embeds_a = normalize_embeddings(embeds_a)
        embeds_b = normalize_embeddings(embeds_b)

    return torch.matmul(embeds_a, embeds_b.T)


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    config: 'GastroTransformerConfig',
    path: str,
    additional_info: Optional[Dict] = None
):
    """Save model checkpoint with all training state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'config': config.to_dict() if hasattr(config, 'to_dict') else config.__dict__
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if additional_info:
        checkpoint.update(additional_info)

    # Also save prototypes if initialized
    if hasattr(model, 'image_prototypes') and model.image_prototypes is not None:
        checkpoint['image_prototypes'] = model.image_prototypes
        checkpoint['rna_prototypes'] = model.rna_prototypes

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda'
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    # C5 fix: use strict=False so checkpoints load across architecture variants
    # (e.g., loading a Tissue Bridge checkpoint into a Q-Integrated model, or after
    # adding new buffers like qformer_prototypes). Missing/unexpected keys are logged
    # as warnings rather than raising RuntimeError.
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing:
        logger.warning(f"Checkpoint missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Checkpoint unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load prototypes if present (L4: prototypes are also in model_state_dict as buffers,
    # but we keep this for backward compat with older checkpoints that only saved them here)
    if 'image_prototypes' in checkpoint and model.image_prototypes is None:
        model.image_prototypes = checkpoint['image_prototypes']
        model.rna_prototypes = checkpoint['rna_prototypes']

    logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")

    return checkpoint


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_retrieval_metrics(
    query_embeds: torch.Tensor,
    gallery_embeds: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Compute cross-modal retrieval metrics.

    For each query, find nearest neighbors in gallery and check if labels match.
    """
    # Compute similarity matrix
    sim = compute_similarity_matrix(query_embeds, gallery_embeds)

    # Sort by similarity (descending)
    _, indices = sim.sort(dim=1, descending=True)

    metrics = {}

    for k in k_values:
        # Get top-k retrieved indices
        top_k_indices = indices[:, :k]

        # Get labels of retrieved items
        retrieved_labels = gallery_labels[top_k_indices]

        # Check if correct label is in top-k
        correct = (retrieved_labels == query_labels.unsqueeze(1)).any(dim=1)
        recall_at_k = correct.float().mean().item()

        metrics[f'recall@{k}'] = recall_at_k

    # Mean Average Precision
    map_score = compute_map(sim, query_labels, gallery_labels)
    metrics['mAP'] = map_score

    return metrics


def compute_map(
    similarity: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor
) -> float:
    """Compute Mean Average Precision."""
    _, indices = similarity.sort(dim=1, descending=True)
    gallery_labels_sorted = gallery_labels[indices]

    # Binary relevance matrix
    relevance = (gallery_labels_sorted == query_labels.unsqueeze(1)).float()

    # Precision at each position
    cum_sum = relevance.cumsum(dim=1)
    positions = torch.arange(1, relevance.shape[1] + 1, device=relevance.device).float()
    precision_at_k = cum_sum / positions.unsqueeze(0)

    # Average precision per query
    ap = (precision_at_k * relevance).sum(dim=1) / relevance.sum(dim=1).clamp(min=1)

    return ap.mean().item()


def compute_ic50_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """Compute IC50 regression metrics."""
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # Pearson correlation
    if len(predictions) > 2:
        pearson_r, pearson_p = stats.pearsonr(predictions, targets)
        spearman_r, spearman_p = stats.spearmanr(predictions, targets)
    else:
        pearson_r = spearman_r = 0.0

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'r2': float(r2)
    }


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """Compute classification metrics."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float()

    accuracy = correct.mean().item()

    # Per-class accuracy
    unique_labels = labels.unique()
    class_accuracies = {}
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            class_acc = correct[mask].mean().item()
            class_accuracies[f'class_{label.item()}_acc'] = class_acc

    return {
        'accuracy': accuracy,
        'num_classes': len(unique_labels),
        **class_accuracies
    }


# =============================================================================
# DATA UTILITIES
# =============================================================================

def split_data_stratified(
    sample_ids: List[str],
    labels: torch.Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data into train/val/test with stratification by labels.

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    np.random.seed(seed)

    train_ids, val_ids, test_ids = [], [], []

    unique_labels = labels.unique()

    for label in unique_labels:
        mask = labels == label
        indices = mask.nonzero().squeeze(-1).tolist()
        np.random.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_ids.extend([sample_ids[i] for i in indices[:n_train]])
        val_ids.extend([sample_ids[i] for i in indices[n_train:n_train + n_val]])
        test_ids.extend([sample_ids[i] for i in indices[n_train + n_val:]])

    return train_ids, val_ids, test_ids


def create_tissue_label_encoder(df: pd.DataFrame, tissue_col: str = 'tissue_type') -> Dict:
    """Create label encoder for tissue types from a DataFrame."""
    unique_tissues = sorted(df[tissue_col].unique())
    encoder = {tissue: i for i, tissue in enumerate(unique_tissues)}
    decoder = {i: tissue for tissue, i in encoder.items()}

    return {
        'encoder': encoder,
        'decoder': decoder,
        'num_classes': len(unique_tissues)
    }


# =============================================================================
# NOTES FOR USER
# =============================================================================
"""
CRITICAL: CELL-LINE TO TISSUE MAPPING
=====================================

The CELLLINE_TO_TISSUE dictionary above contains mappings for common cell-lines.
You MUST update this mapping with all cell-lines present in your IC50 dataset.

To find cell-line tissue origins:
1. Cellosaurus (https://www.cellosaurus.org/) - Search by cell-line name
2. CCLE/DepMap (https://depmap.org/) - Download sample metadata
3. GDSC (https://www.cancerrxgene.org/) - Cell line annotations

HOW TO UPDATE:
--------------
1. Get unique cell-line IDs from your IC50 CSV:
   ```python
   import pandas as pd
   df = pd.read_csv('your_ic50.csv')
   unique_celllines = df['cellline_id'].unique()
   ```

2. For each missing cell-line, look up its tissue of origin

3. Add to CELLLINE_TO_TISSUE dictionary:
   ```python
   CELLLINE_TO_TISSUE['YOUR_CELLLINE'] = 'Tissue_Type'
   ```

4. If tissue type doesn't exist in TISSUE_TYPES, add it

AUTOMATIC MAPPING SCRIPT:
------------------------
You can use the create_cellline_to_tissue_csv() function to generate
a CSV file that shows which cell-lines are mapped and which are missing.
"""
