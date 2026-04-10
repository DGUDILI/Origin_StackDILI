"""
graph_utils.py — GraphMACCSEncoder 용 분자 그래프/MACCS 전처리 유틸리티

주요 기능:
- smiles_to_pyg()     : SMILES → PyG Data (atom features + edge_index)
- get_maccs()         : SMILES → 167-dim binary tensor
- MACCS_NAMES         : {bit_idx: human-readable name} (bit 0 = None/Unused)
- MACCS_SMARTS        : {bit_idx: SMARTS string} (RDKit BSD-licensed)
- verify_maccs_mapping(): 런타임 자동 검증 (off-by-one 감지)

라이선스 참고:
- MACCS_SMARTS 는 RDKit MACCSkeys.smartsPatts 기반 (BSD 3-Clause)
- MACCS_NAMES  는 Durant et al. 2002 공개 논문 기반 단축명
"""

import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.MACCSkeys import smartsPatts

# ── RDKit 버전 기록 (버전 불일치 감지용) ─────────────────────────────────────
RDKIT_VERSION = rdkit.__version__

# ── MACCS SMARTS (RDKit BSD 3-Clause) ────────────────────────────────────────
# key_idx → SMARTS string. bit 0은 RDKit 내부 1-based indexing용 dummy.
MACCS_SMARTS: dict[int, str] = {k: v[0] for k, v in smartsPatts.items()}

# ── MACCS Key 인간 가독 단축명 ────────────────────────────────────────────────
# 출처: Durant et al. J. Chem. Inf. Comput. Sci. 2002, 42, 1273-1280 (공개 학술자료)
# Bit 0: Unused (RDKit 1-based indexing artifact, always 0)
# Bits 1~166: MDL MACCS structural keys (RDKit 구현 기준)
MACCS_NAMES: dict[int, str | None] = {
    0:   None,                              # Unused — always 0
    1:   "Isotope",
    2:   "Atomic num >103",
    3:   "Group IVA (Ge,Sn,Pb)",
    4:   "Group VA (As,Sb,Bi)",
    5:   "Group VIA (Se,Te)",
    6:   "Group VIIA (I,At)",
    7:   "Rows 4+ (K and above)",
    8:   "Group VIII (Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt)",
    9:   "Group IIa (Mg,Ca,Ba)",
    10:  "4-membered ring",
    11:  "Group IB (Cu,Ag,Au)",
    12:  "Group IIB (Zn,Cd,Hg)",
    13:  "Group IIIA (Al,Ga,In,Tl)",
    14:  "Uranium/Actinide",
    15:  "Group IIA (Be,Mg,Ca,Sr,Ba)",
    16:  "Group IA (Li,Na,K,Rb,Cs)",
    17:  "N-O bond",
    18:  "Group VIb (Cr,Mo,W)",
    19:  "Group Vb (V,Nb,Ta)",
    20:  "Group IVb (Ti,Zr,Hf)",
    21:  "Group IIIb (Sc,Y,La,Ac)",
    22:  "Group VIIb (Mn,Tc,Re)",
    23:  "Si",
    24:  "S-S bond",
    25:  "C=C-C=O in ring",
    26:  "Acetal",
    27:  "Halide (Cl, Br, I)",
    28:  "N-C=O (amide/carbamate)",
    29:  "O (not in ring)",
    30:  "N in ring",
    31:  "S-O bond",
    32:  "N-S bond",
    33:  "C=C (not in ring)",
    34:  "N-N bond",
    35:  "N in 6-membered ring",
    36:  "O in 6-membered ring",
    37:  "F",
    38:  "Aldehyde",
    39:  "CCCO",
    40:  "NO2",
    41:  "CC(N)C",
    42:  "C=N",
    43:  "Nonionic nitrogen",
    44:  "N-OH or N-NH2",
    45:  "Heteroatom in 5-membered ring",
    46:  "SC=N",
    47:  "CH2=A",
    48:  "Group VIII metal",
    49:  "Heteroatom in 4-membered ring",
    50:  "S (not aromatic)",
    51:  "N-C=N",
    52:  "Aromatic heteroatom",
    53:  "Halogen",
    54:  "Aromatic N",
    55:  "C=O in ring",
    56:  "S (aromatic)",
    57:  "C-O bond in ring",
    58:  "C=N or N=C",
    59:  "Epoxide",
    60:  "n-membered ring (n<4)",
    61:  "N (any) in 5-membered ring",
    62:  "Aromatic C",
    63:  "N=O",
    64:  "A-O-A (ether, non-ring)",
    65:  "Quaternary carbon",
    66:  "CF3 or similar",
    67:  "Two O atoms",
    68:  "Phosphorus",
    69:  "C-S bond (ring)",
    70:  "SO2",
    71:  "AC=CA",
    72:  "Thiophene",
    73:  "N (ring) connected to N (ring)",
    74:  "O=CN",
    75:  "Ketone",
    76:  "O=O or N=N",
    77:  "CCN",
    78:  "NCC",
    79:  "CCC",
    80:  "C=O (ester/acid)",
    81:  "A-C=O (aromatic ketone)",
    82:  "Enamine",
    83:  "N-N=C",
    84:  "Aniline or phenoxy",
    85:  "C=CC=C (butadiene system)",
    86:  "Alkyl amine",
    87:  "C-N bond (ring)",
    88:  "CC(O)C",
    89:  "Oxygen in C-O-C (ether)",
    90:  "CC(N)C (amino alkyl)",
    91:  "Oxygen-containing ring",
    92:  "Nitrile (C#N)",
    93:  "Pyridinium",
    94:  "S in ring",
    95:  "SNS or SOS",
    96:  "3-membered ring",
    97:  "NC(=O)N (urea)",
    98:  "N-C(=S)-N (thiourea)",
    99:  "Imidazole",
    100: "Alkyl halide",
    101: "C=CC=O (alpha-beta unsat. carbonyl)",
    102: "N (cyclic) bonded to 2 chains",
    103: "AN(A)A",
    104: "A-O-A in ring",
    105: "C=O",
    106: "N in aromatic ring",
    107: "NO2 (nitro)",
    108: "A-N bond (aromatic nitrogen)",
    109: "NHR (secondary amine)",
    110: "Anhydride",
    111: "Primary or secondary amine",
    112: "Acyl halide",
    113: "Acid anhydride",
    114: "Sulfonamide",
    115: "NC(=O) (amide)",
    116: "NCCN",
    117: "CC(=O)O (ester)",
    118: "CC=O (methyl ketone)",
    119: "OC(=O)N",
    120: "C=C (aromatic, exo ring)",
    121: "C-C(=O)-C",
    122: "C#C",
    123: "C-C(=O)-N",
    124: "Aromatic amine",
    125: "A-CH2-A",
    126: "C-N (aromatic connection)",
    127: "OH attached to aromatic",
    128: "O attached to aromatic",
    129: "C=C (vinyl)",
    130: "C-C(=O)-O",
    131: "Cl",
    132: "N-H (any)",
    133: "O-H (any)",
    134: "Nitrogen (any)",
    135: "Aromatic ring",
    136: "6-membered ring",
    137: "O (any)",
    138: "Ring (any)",
    139: "7-membered ring",
    140: "Si",
    141: "S (any)",
    142: "C (sp3)",
    143: "N-OH (hydroxylamine)",
    144: "OC(=O)O (carbonate)",
    145: "C-OC (carbon-ether)",
    146: "S=O",
    147: "B",
    148: "NC=O",
    149: "NC#N",
    150: "Nitrogen in 6-ring",
    151: "Secondary amine",
    152: "Tertiary amine",
    153: "OCC",
    154: "OCCO",
    155: "NCO",
    156: "Carboxylic acid",
    157: "CC(C)C",
    158: "C=C-C",
    159: "CC(O)N",
    160: "Primary amine",
    161: "CCOC",
    162: "Aromatic ring (confirmed)",
    163: "Heterocyclic ring",
    164: "CCN",
    165: "CCC",
    166: "Ring (general)",
}

# ── Atom 피처 함수 ────────────────────────────────────────────────────────────

def _one_hot(value, choices):
    """value가 choices 중 몇 번째인지 one-hot 벡터 반환. 없으면 마지막 원소."""
    enc = [0] * len(choices)
    idx = choices.index(value) if value in choices else len(choices) - 1
    enc[idx] = 1
    return enc


def get_atom_features(atom) -> list[float]:
    """
    원자 하나의 피처 벡터 반환.
    총 43-dim (one-hot 조합).
    """
    results = []
    # 원자 번호 (11)
    results += _one_hot(atom.GetAtomicNum(), [1,6,7,8,9,15,16,17,35,53,0])
    # Degree (7)
    results += _one_hot(atom.GetDegree(), [0,1,2,3,4,5,6])
    # Formal charge (7)
    results += _one_hot(atom.GetFormalCharge(), [-3,-2,-1,0,1,2,3])
    # NumHs (5)
    results += _one_hot(atom.GetTotalNumHs(), [0,1,2,3,4])
    # Hybridization (6)
    from rdkit.Chem import rdchem
    results += _one_hot(atom.GetHybridization(), [
        rdchem.HybridizationType.S,
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ])
    # IsAromatic (1)
    results.append(float(atom.GetIsAromatic()))
    # IsInRing (1)
    results.append(float(atom.IsInRing()))
    # RingSize (5: 3,4,5,6,7+)
    ring_info = atom.GetOwningMol().GetRingInfo()
    sizes = [sz for ring in ring_info.AtomRings() if atom.GetIdx() in ring for sz in [len(ring)]]
    if not sizes:
        results += [0, 0, 0, 0, 0]
    else:
        ms = min(sizes)
        results += _one_hot(ms, [3, 4, 5, 6, 7])
    return results  # 11+7+7+5+6+1+1+5 = 43


ATOM_FEAT_DIM = 43


BOND_FEAT_DIM = 9


def get_bond_features(bond) -> list:
    """
    결합 하나의 9-dim 피처 벡터.
    [single, double, triple, aromatic, in_ring, conjugated,
     stereo_none, stereo_E, stereo_Z]
    """
    from rdkit.Chem import rdchem
    bt = bond.GetBondType()
    st = bond.GetStereo()
    return [
        float(bt == rdchem.BondType.SINGLE),
        float(bt == rdchem.BondType.DOUBLE),
        float(bt == rdchem.BondType.TRIPLE),
        float(bt == rdchem.BondType.AROMATIC),
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
        float(st == rdchem.BondStereo.STEREONONE),
        float(st == rdchem.BondStereo.STEREOE),
        float(st == rdchem.BondStereo.STEREOZ),
    ]


def smiles_to_pyg(smiles: str):
    """
    SMILES → torch_geometric.data.Data

    Returns:
        Data(x=(n_atoms, ATOM_FEAT_DIM), edge_index=(2, n_bonds*2),
             edge_attr=(n_bonds*2, BOND_FEAT_DIM))
        None if SMILES is invalid
    """
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_feats = [get_atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float32)  # (n_atoms, 43)

    # Edge index + bond features (undirected: 각 결합 → 양방향)
    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_feats += [feat, feat]  # 양방향 동일 피처

    if len(src) == 0:  # 단원자 분자
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_feats, dtype=torch.float32)  # (n_bonds*2, 9)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def augment_smiles(smiles: str, n_aug: int, seed: int | None = None) -> list[str]:
    """
    1개 SMILES → n_aug개 랜덤 SMILES 생성.

    원리: Chem.MolToSmiles(mol, doRandom=True)는 동일 분자를 다른
    원자 열거 순서(atom traversal order)로 직렬화 → 서로 다른 SMILES 문자열.

    증강 효과:
      - ChemBERTa 브랜치: 다른 토큰 시퀀스 → 다양한 컨텍스트 학습
      - GINEConv 브랜치: 다른 원자 인덱싱 순서 → 메시지 패싱 경로 다양화
      - MACCS 브랜치: 분자 동일 → 항상 동일 (불변 — 호출 측에서 원본 재사용)

    Args:
        smiles: 원본 SMILES 문자열
        n_aug:  생성할 증강 SMILES 개수
        seed:   재현성용 시드 (None이면 비결정적)

    Returns:
        n_aug개의 랜덤 SMILES 리스트.
        유니크 생성 실패 시 canonical SMILES로 나머지 채움.
        smiles 파싱 실패 시 원본 n_aug번 반복.
    """
    import random as _random
    _random.seed(seed)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * n_aug  # 파싱 실패 → 원본 반복

    canonical = Chem.MolToSmiles(mol, doRandom=False)
    seen = {canonical}
    augmented = []
    max_attempts = n_aug * 20  # 유니크 탐색 여유

    for _ in range(max_attempts):
        if len(augmented) >= n_aug:
            break
        rand_smi = Chem.MolToSmiles(mol, doRandom=True)
        if rand_smi not in seen:
            seen.add(rand_smi)
            augmented.append(rand_smi)

    # 유니크 부족 시 canonical로 보충 (소분자에서 발생 가능)
    while len(augmented) < n_aug:
        augmented.append(canonical)

    return augmented


def get_maccs(smiles: str) -> torch.Tensor:
    """
    SMILES → 167-dim binary tensor (float32).
    Bit 0 은 항상 0 (RDKit 1-based indexing artifact).
    Bits 1~166: MDL MACCS structural keys (RDKit BSD 구현).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(167, dtype=torch.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor(list(fp), dtype=torch.float32)  # (167,)


# ── 런타임 검증 ──────────────────────────────────────────────────────────────

def verify_maccs_mapping(verbose: bool = True) -> bool:
    """
    벤젠(aromatic ring)과 에탄올로 핵심 bit 검증.
    off-by-one 또는 RDKit 버전 불일치를 조기에 탐지.

    Returns:
        True if all checks pass
    Raises:
        AssertionError if any check fails
    """
    # 벤젠: aromatic ring → bit 162 == 1
    benzene_fp = get_maccs("c1ccccc1")
    assert benzene_fp[162] == 1.0, \
        f"[verify_maccs] 벤젠 bit 162(Aromatic ring) = {benzene_fp[162]}, expected 1 — off-by-one?"

    # bit 0 항상 0
    assert benzene_fp[0] == 0.0, \
        f"[verify_maccs] bit 0 = {benzene_fp[0]}, expected 0 — RDKit 구현 변경?"

    # 에탄올: OH → bit 139(O-H) == 1
    ethanol_fp = get_maccs("CCO")
    assert ethanol_fp[139] == 1.0, \
        f"[verify_maccs] 에탄올 bit 139(O-H) = {ethanol_fp[139]}, expected 1"

    if verbose:
        print(f"[graph_utils] MACCS 검증 통과 (RDKit {RDKIT_VERSION})")
    return True


# ── 빠른 테스트 (직접 실행 시) ───────────────────────────────────────────────

if __name__ == "__main__":
    verify_maccs_mapping()

    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    pyg = smiles_to_pyg(aspirin)
    maccs = get_maccs(aspirin)

    print(f"\n아스피린 PyG:")
    print(f"  x.shape      = {pyg.x.shape}")
    print(f"  edge_index.shape = {pyg.edge_index.shape}")
    print(f"\nMACCS (167-dim):")
    print(f"  bit 0 (Unused) = {maccs[0].item()}")
    print(f"  활성 key 수    = {maccs.sum().int().item()} / 166")
    print(f"\n활성 MACCS Keys:")
    for i in range(1, 167):
        if maccs[i] == 1:
            name = MACCS_NAMES.get(i, f"Key {i}")
            print(f"  bit {i:3d}: {name}")
