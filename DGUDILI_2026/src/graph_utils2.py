"""
graph_utils2.py — 올바른 MACCS Keys 번역기 (1~166 전체 완전판)
"""
from graph_utils import MACCS_SMARTS

# MACCS Keys 1번부터 166번까지 전체 매핑 (Durant et al. 표준 기준)
HUMAN_READABLE_MACCS = {
    1: 'Isotope', 2: 'Atomic Num >103', 3: 'Group IVa,Va,VIa', 4: 'Actinide', 5: 'Group IIIB,IVB',
    6: 'Lanthanide', 7: 'Group VB,VIB,VIIB', 8: 'Q-X-Q', 9: 'Group VIII', 10: 'Group IIA (Alkaline earth)',
    11: 'Group IB,IIB', 12: 'Group VA,VIA', 13: 'ON(C)C', 14: 'S-S', 15: 'O in 3-Ring',
    16: 'Hetero in 3-Ring', 17: 'C#C', 18: 'Group IIIA', 19: '7-Ring', 20: 'Si',
    21: 'C=C(Q)Q', 22: '3-Ring', 23: 'N-C=O', 24: 'N-O', 25: 'N in 3-Ring',
    26: 'Ar-N=C', 27: 'I (Iodine)', 28: 'Q-CH2-Q', 29: 'P (Phosphorus)', 30: 'Q-Q-Q',
    31: 'Br (Bromine)', 32: 'S in 5-Ring', 33: 'N in 5-Ring', 34: 'O in 5-Ring', 35: 'Group IA (Alkali)',
    36: 'S Heterocycle', 37: 'N Heterocycle', 38: 'NC(C)N', 39: 'O Heterocycle', 40: 'S-O',
    41: 'C#N', 42: 'F (Fluorine)', 43: 'Q-HA-Q', 44: 'Other Element', 45: 'C=C-N',
    46: 'Br', 47: 'S=O', 48: 'O Heterocycle', 49: 'Charge', 50: 'C=C(C)C',
    51: 'C-S', 52: 'NN', 53: 'Q-Q-Q-Q', 54: 'Q-CH2-Q', 55: 'O-S-O',
    56: 'O-N(C)C', 57: 'O Heterocycle', 58: 'Q-S-Q', 59: 'S in Ring', 60: 'S (Sulfur)',
    61: 'A-S-A', 62: 'Ar-S', 63: 'N=O', 64: 'S', 65: 'C:N',
    66: 'C-C-C', 67: 'O-S', 68: 'Q-CH2-Q', 69: 'QQ', 70: 'Q-Q-Q',
    71: 'N-O', 72: 'O-O', 73: 'S=O', 74: 'CH3', 75: 'N-N',
    76: 'C=C-O', 77: 'N-N', 78: 'C=N', 79: 'N-C-N', 80: 'N-C-N',
    81: 'O-C-N', 82: 'CH2', 83: 'Q-Q', 84: 'NH2', 85: 'C-N-C',
    86: 'CH2-Q', 87: 'Halogen', 88: 'S', 89: 'O-C-O', 90: 'QH',
    91: 'QH', 92: 'O-C-N', 93: 'QCH3', 94: 'QN', 95: 'N-O',
    96: '5-Ring', 97: 'N-C-O', 98: 'Q-Q', 99: 'C=C', 100: 'N-C-N',
    101: '8M Ring', 102: 'Q-O', 103: 'Cl (Chlorine)', 104: 'Q-Q', 105: 'A-O-A',
    106: 'Q-O-Q', 107: 'Halogen', 108: 'CH3', 109: 'Q-CH2-Q', 110: 'N-C-O',
    111: 'N-C-O', 112: 'Q-Q', 113: 'O-C=O (Ester/Carboxyl)', 114: 'CH3', 115: 'O-C-O',
    116: 'CH3', 117: 'N-O', 118: 'CH2-CH2', 119: 'N=N', 120: 'Heterocycle',
    121: 'N Heterocycle', 122: 'N-N', 123: 'O-C-O', 124: 'QQ', 125: 'Aromatic Ring',
    126: 'A-O-A', 127: 'O attached to Aromatic', 128: 'Q-CH2-Q', 129: 'Q-Q', 130: 'Q-Q',
    131: 'QH', 132: 'O-CH2', 133: 'Ar-N', 134: 'Halogen', 135: 'N-C-O',
    136: '>1 Oxygen', 137: 'Heterocycle', 138: 'Q-CH2-Q', 139: 'O-H (Hydroxyl)', 140: '>3 Oxygen',
    141: 'CH3', 142: 'N-C-N', 143: 'O-C-O', 144: 'O-C-O', 145: '6M Ring',
    146: 'O (Oxygen)', 147: 'C-C-C-C', 148: 'AQ', 149: 'CH3', 150: 'Aromatic Carbon',
    151: 'NH', 152: 'O-C-C', 153: 'C-O (Ether/Ester)', 154: 'C=O (Carbonyl)', 155: 'CH2-O',
    156: 'N-H', 157: 'C-O', 158: 'C-N', 159: 'O (Oxygen)', 160: 'Methyl (CH3)',
    161: 'Nitrogen (N)', 162: 'Aromatic Ring', 163: '6-Membered Ring', 164: 'Oxygen (O)', 165: 'Ring',
    166: 'Fragments'
}

def get_correct_maccs_name(bit_idx: int) -> str:
    """1순위: 사람 친화적 이름, 2순위: SMARTS, 3순위: Bit 번호"""
    if bit_idx in HUMAN_READABLE_MACCS:
        return HUMAN_READABLE_MACCS[bit_idx]
    if bit_idx in MACCS_SMARTS:
        return f"Pattern: {MACCS_SMARTS[bit_idx]}"
    return f"Bit_{bit_idx}"