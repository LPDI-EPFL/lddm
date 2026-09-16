import numpy as np
from rdkit import Chem
from rdkit.Chem.Features.FeatDirUtilsRD import ArbAxisRotation, GetAcceptor2FeatVects, GetAcceptor3FeatVects, GetDonor1FeatVects, GetDonor2FeatVects, GetDonor3FeatVects

from lddm.constants import aa_decoder


WATER_VDW_RADIUS = 1.6  # from https://water.lsbu.ac.uk/water/water_molecule.html
HBOND_DISTANCE = 2.8  # from https://bc401.bmb.colostate.edu/appendix/h-bonds.php

BACKBONE_DONORS = {k: ["N"] for k in aa_decoder}
BACKBONE_ACCEPTORS = {k: ["O"] for k in aa_decoder}

# from https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/charge/
SIDECHAIN_DONORS = {
    "R": ["NE", "NH1", "NH2"],
    "N": ["ND2"],
    "Q": ["NE2"],
    "H": ["ND1", "NE2"],
    "K": ["NZ"],
    "S": ["OG"],
    "T": ["OG1"],
    "W": ["NE1"],
    "Y": ["OH"],
}
SIDECHAIN_ACCEPTORS = {
    "N": ["OD1"],
    "D": ["OD1", "OD2"],
    "Q": ["OE1"],
    "E": ["OE1", "OE2"],
    "H": ["ND1", "NE2"],
    "S": ["OG"],
    "T": ["OG1"],
    "Y": ["OH"],
}

AMINO_ACID_DONORS = {k: BACKBONE_DONORS.get(k, []) + SIDECHAIN_DONORS.get(k, []) for k in aa_decoder}
AMINO_ACID_ACCEPTORS = {k: BACKBONE_ACCEPTORS.get(k, []) + SIDECHAIN_ACCEPTORS.get(k, []) for k in aa_decoder}

WATER_RESNAMES = {"HOH", "WAT"}


def get_hbond_donors(molecule):
    # from: https://github.com/rdkit/rdkit/blob/64061b6ca71121f7c3837393ff0a35f6261596a9/rdkit/Chem/Lipinski.py#L29
    HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')
    return [x[0] for x in molecule.GetSubstructMatches(HDonorSmarts)]


def get_hbond_acceptors(molecule):
    # from: https://github.com/rdkit/rdkit/blob/64061b6ca71121f7c3837393ff0a35f6261596a9/rdkit/Chem/Lipinski.py#L32
    HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                            '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                            '$([nH0,o,s;+0])]')
    return [x[0] for x in molecule.GetSubstructMatches(HAcceptorSmarts)]


"""
Overwrite the rdkit implementation of GetAcceptor1FeatVects() because of the issue discussed here:
https://github.com/rdkit/rdkit/issues/3433
"""
def GetAcceptor1FeatVects(conf, featAtoms, scale=1.5):
    """
    Get the direction vectors for Acceptor of type 1

    This is a acceptor with one heavy atom neighbor. There are two possibilities we will
    consider here
    1. The bond to the heavy atom is a single bond e.g. CO
     In this case we don't know the exact direction and we just use the inversion of this bond direction
     and mark this direction as a 'cone'
    2. The bond to the heavy atom is a double bond e.g. C=O
     In this case the we have two possible direction except in some special cases e.g. SO2
     where again we will use bond direction

    ARGUMENTS:
    featAtoms - list of atoms that are part of the feature
    scale - length of the direction vector
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    mol = conf.GetOwningMol()
    nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()

    cpt = conf.GetAtomPosition(aid)

    # find the adjacent heavy atom
    heavyAt = -1
    for nbr in nbrs:
        if nbr.GetAtomicNum() != 1:
          heavyAt = nbr
          break

#     singleBnd = mol.GetBondBetweenAtoms(aid, heavyAt.GetIdx()).GetBondType() > Chem.BondType.SINGLE
    singleBnd = mol.GetBondBetweenAtoms(aid,heavyAt.GetIdx()).GetBondType() == Chem.BondType.SINGLE  # MODIFIED HERE

    # special scale - if the heavy atom is a sulfur (we should proabably check phosphorous as well)
    sulfur = heavyAt.GetAtomicNum() == 16

    if singleBnd or sulfur:
        v1 = conf.GetAtomPosition(heavyAt.GetIdx())
        v1 -= cpt
        v1.Normalize()
        v1 *= (-1.0 * scale)
        v1 += cpt
        return ((cpt, v1), ), 'cone'

    # ok in this case we will assume that
    # heavy atom is sp2 hybridized and the direction vectors (two of them)
    # are in the same plane, we will find this plane by looking for one
    # of the neighbors of the heavy atom
    hvNbrs = heavyAt.GetNeighbors()
    hvNbr = -1
    for nbr in hvNbrs:
        if nbr.GetIdx() != aid:
          hvNbr = nbr
          break

    pt1 = conf.GetAtomPosition(hvNbr.GetIdx())
    v1 = conf.GetAtomPosition(heavyAt.GetIdx())
    pt1 -= v1
    v1 -= cpt
    rotAxis = v1.CrossProduct(pt1)
    rotAxis.Normalize()
    bv1 = ArbAxisRotation(120, rotAxis, v1)
    bv1.Normalize()
    bv1 *= scale
    bv1 += cpt
    bv2 = ArbAxisRotation(-120, rotAxis, v1)
    bv2.Normalize()
    bv2 *= scale
    bv2 += cpt
    return (
        (cpt, bv1),
        (cpt, bv2),
    ), 'linear'


FEAT_VEC_FUNCS = {
   "GetAcceptor1FeatVects": GetAcceptor1FeatVects,
   "GetAcceptor2FeatVects": GetAcceptor2FeatVects,
   "GetAcceptor3FeatVects": GetAcceptor3FeatVects,
   "GetDonor1FeatVects": GetDonor1FeatVects,
   "GetDonor2FeatVects": GetDonor2FeatVects,
   "GetDonor3FeatVects": GetDonor3FeatVects,
}


def get_donor_acceptor_type(atom):
    """
    see: https://www.rdkit.org/docs/source/rdkit.Chem.Features.FeatDirUtilsRD.html

    Acceptor/donor of type 1: acceptor/donor with one heavy atom neighbor.
    Acceptor/donor of type 2: acceptor/donor with two adjacent heavy atoms.
    Acceptor/donor of type 3: acceptor/donor with three adjacent heavy atoms.
    """
    return sum([a.GetSymbol() != 'H' for a in atom.GetNeighbors()])


def hbond_vectors(rdmol, atom_idx, type, length=1.0):
   assert type in {'acceptor', 'donor'}
   detailed_type = get_donor_acceptor_type(rdmol.GetAtomWithIdx(atom_idx))
   if detailed_type == 0:
       return None, None
   get_vectors = FEAT_VEC_FUNCS.get(f"Get{type.capitalize()}{detailed_type}FeatVects")
   if get_vectors is None:
       return None, None
   return get_vectors(rdmol.GetConformer(), [atom_idx], scale=length)
   

def dot(a, b):
    """Compute dot products for each pair of rows in a and b."""
    return np.sum(a * b, axis=-1)


class CircleIn3D:
    def __init__(self, center: np.array, radius: float, normal: np.array):
        self.c = center
        self.r = radius
        self.n = self._normalize(normal)
        self.v1, self.v2 = self._get_orthogonal_unit_vectors()

    @staticmethod
    def _normalize(vec):
        return vec / np.linalg.norm(vec, axis=-1, keepdims=True)

    def _get_orthogonal_unit_vectors(self):

        r = np.random.randn(3)
        v1 = r - r.T @ self.n * self.n
        v1 = self._normalize(v1)

        v2 = np.cross(self.n, v1)
        v2 = self._normalize(v2)

        assert abs(v1.T @ self.n) < 1e-6
        assert abs(v2.T @ self.n) < 1e-6
        assert abs(v1.T @ v2) < 1e-6

        return v1, v2
    
    def _point_on_circle(self, theta):
        """
        Returns points on the circle for given thetas, shape (n_thetas, 3)
        """
        theta = theta.reshape(-1, 1)
        return self.c[None, :] + self.r * np.cos(theta) * self.v1[None, :] + self.r * np.sin(theta) * self.v2[None, :]
    
    def map_onto_circle(self, query_point):
        """Maps query point to closest and farthest point on the circle."""
        # task: min_{\theta} ||p(\theta) - coord||^2
        # get \theta by solving d/d\theta) ||p(\theta) - coord||^2 == 0
        query_point = query_point.reshape(-1, 3)
        diff = self.c[None, :] - query_point
        # theta = np.arctan2(diff.T @ self.v2, diff.T @ self.v1)
        theta = np.arctan2(dot(diff, self.v2[None, :]), dot(diff, self.v1[None, :]))

        # get corresponding point in 3D
        p = self._point_on_circle(theta)
        
        # check second derivative
        # # second_derivative = 2 * (query_point - self.c).T @ (p - self.c)
        # if second_derivative < 0:
        #     # we found a maximum, the minimum must be on the other side of the circle
        #     p = self._point_on_circle(theta + np.pi)
        second_derivative = 2 * dot(query_point - self.c[None, :], p - self.c[None, :])
        is_maximum = second_derivative < 0
        
        p_closest = p
        p_farthest = p_closest.copy()
        
        # if maximum, the minimum must be on the other side of the circle
        opposite_points = self._point_on_circle(theta + np.pi)
        p_closest[is_maximum] = opposite_points[is_maximum]  

        p_farthest[~is_maximum] = opposite_points[~is_maximum] 
            
        return p_closest, p_farthest
    

def find_blocked_segment(circle: CircleIn3D, donut_radius: float, sphere_centers: np.array, sphere_radii: np.array):
    """
    Return range for theta such that spheres of size donut_radius centered at 
    any theta in this range will clash with the sphere at sphere_center.
    I.e.: ||p(theta) - sphere_center|| \leq (donut_radius + sphere_radius)

    :param circle: parameterized circle object
    :param donut_radius: float, radius of the donut around the circle
    :param sphere_centers: shape (n, 3); centers of spheres for which the clashing segments should be determined
    :param sphere_radii: shape (n,); radii of the spheres for which the clashing segments should be determined 
    """

    # a * cos(theta) + b * sin(theta) \leq c
    R = donut_radius + sphere_radii  # (n,)
    delta = circle.c[None, :] - sphere_centers  # (n, 3)
    # a = circle.v1.T @ delta
    a = dot(circle.v1[None, :], delta)  # (n,)
    # b = circle.v2.T @ delta
    b = dot(circle.v2[None, :], delta)  # (n,)
    c = (R**2 - np.linalg.norm(delta, axis=-1)**2 - circle.r**2) / (2 * circle.r)  # (n,)

    # print(f"a={a}, b={b}, c={c}")

    # WolframAlpha solution: https://www.wolframalpha.com/input?i=solve+a*cos%28x%29%2Bb*sin%28x%29%3Dc+for+x
    # using 
    # sin(x) = (2*tan(x/2)) / (1 + tan(x/2)^2)
    # cos(x) = (1 - tan(x/2)^2) / (1 + tan(x/2)^2)
    # and u = tan(theta/2)
    # we get inequality
    # -(a + c) u^2 + 2bu + (a - c) \leq 0

    # solve intermediate quadratic inequality
    radicant = a**2 + b**2 - c**2
    radicant[radicant < 0] = np.nan
    u1 = (b - np.sqrt(radicant)) / (a + c)
    u2 = (b + np.sqrt(radicant)) / (a + c)
    # u1 = (b / (a + c) - np.sqrt((a**2 + b**2 - c**2) / (a + c)**2))
    # u2 = (b / (a + c) + np.sqrt((a**2 + b**2 - c**2) / (a + c)**2))
    
    # final result
    _theta_1 = 2 * np.arctan(u1)
    _theta_2 = 2 * np.arctan(u2)
    _theta_1 = _theta_1 % (2 * np.pi)
    _theta_2 = _theta_2 % (2 * np.pi)
    theta_1 = np.minimum(_theta_1, _theta_2)
    theta_2 = np.maximum(_theta_1, _theta_2)

    # these two point split the circle into two segments
    # check in which one the inequality is satisfied
    inequality = lambda theta: a * np.cos(theta) + b * np.sin(theta) <= c
    midpoint = (theta_1 + theta_2) / 2
    # theta_lb, theta_ub = (theta_1, theta_2) if inequality(midpoint) else (theta_2, theta_1)
    theta_lb, theta_ub = (theta_1, theta_2)
    wrong_inds = ~inequality(midpoint)
    theta_lb[wrong_inds] = theta_2[wrong_inds]
    theta_ub[wrong_inds] = theta_1[wrong_inds]

    # Special case 1: sphere does not block any segment
    closest_points, farthest_points = circle.map_onto_circle(sphere_centers)
    cond = np.linalg.norm(sphere_centers - closest_points, axis=-1) > R
    theta_lb[cond] = 0.0
    theta_ub[cond] = 0.0

    # Special case 2: sphere blocks the whole donut
    cond = np.linalg.norm(sphere_centers - farthest_points, axis=-1) <= R
    theta_lb[cond] = 0.0
    theta_ub[cond] = np.inf

    return (theta_lb, theta_ub)
    

def full_circle_covered(segments):
    """ 
    Check if angular segments cover the full circle, i.e. [0, 2*pi)
    :param segments: list of tuples (lower bound, upper bound)
    """
    segments = sorted(segments, key=lambda x: x[0])
    global_min, global_max = None, None

    # iteratively combine segments
    for lb, ub in segments:

        if np.isinf(ub):  # special case where whole circle is already covered
            return True

        lb = lb % (2 * np.pi)
        ub = ub % (2 * np.pi)

        segment_wraps = (lb > ub)
        if segment_wraps:
            ub = ub + 2 * np.pi

        if global_max is None:
            global_min, global_max = lb, ub

        if lb > global_max:
            return False  # gap found, circle cannot be fully covered 

        global_max = max(global_max, ub)

        if global_max > 2 * np.pi + global_min:
            return True  # whole circle covered

    return False  # all segments processed 
