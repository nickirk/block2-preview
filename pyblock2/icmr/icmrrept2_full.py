
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

"""
Internally-Contracted MR-REPT2 (MRLCC2) [J. Chem. Theory Comput. 13, 488 (2017)]
with equations derived on the fly.
need internal contraction module of block2.

Full Hamiltonian matrix is build for solving the linear problem.
May consume large amount of memory.
"""

try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr, MapStrStr
except ImportError:
    raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")

import numpy as np
import time

try:
    from . import eri_helper, dmrg_helper
except ImportError:
    import eri_helper, dmrg_helper

def init_parsers():

    idx_map = MapWickIndexTypesSet()
    idx_map[WickIndexTypes.Inactive] = WickIndex.parse_set("mnxyijkl")
    idx_map[WickIndexTypes.Active] = WickIndex.parse_set("mnxyabcdefghpq")
    idx_map[WickIndexTypes.External] = WickIndex.parse_set("mnxyrstu")

    perm_map = MapPStrIntVectorWickPermutation()
    perm_map[("w", 4)] = WickPermutation.qc_phys()

    defs = MapStrPWickTensorExpr()
    p = lambda x: WickExpr.parse(x, idx_map, perm_map).substitute(defs)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    pd = lambda x: WickExpr.parse_def(x, idx_map, perm_map)

    return p, pt, pd, defs

P, PT, PD, DEF = init_parsers() # parsers
SP = lambda x: x.expand().add_spin_free_trans_symm().remove_external().simplify()
SPR = lambda x: x.expand().add_spin_free_trans_symm().remove_external().remove_inactive().simplify()
Comm = lambda b, h, k: SP(b.conjugate() * (h ^ k))
Rhs = lambda b, k: SPR(b.conjugate() * k)

# See J. Chem. Theory Comput. 15, 2291 (2019) Eq. (11)
# Fink's Hamiltonian
h1 = P("""
    SUM <ij> h[ij] E1[i,j]
    SUM <ab> h[ab] E1[a,b]
    SUM <rs> h[rs] E1[r,s]
""")
h2 = P("""
    0.5 SUM <ijkl> w[ijkl] E2[ij,kl]
    0.5 SUM <abcd> w[abcd] E2[ab,cd]
    0.5 SUM <rstu> w[rstu] E2[rs,tu]
    0.5 SUM <iajb> w[iajb] E2[ia,jb] \n + 0.5 SUM <iajb> w[iabj] E2[ia,bj]
    0.5 SUM <irjs> w[irjs] E2[ir,js] \n + 0.5 SUM <irjs> w[irsj] E2[ir,sj]
    0.5 SUM <aibj> w[aibj] E2[ai,bj] \n + 0.5 SUM <aibj> w[aijb] E2[ai,jb]
    0.5 SUM <arbs> w[arbs] E2[ar,bs] \n + 0.5 SUM <arbs> w[arsb] E2[ar,sb]
    0.5 SUM <risj> w[risj] E2[ri,sj] \n + 0.5 SUM <risj> w[rijs] E2[ri,js]
    0.5 SUM <rasb> w[rasb] E2[ra,sb] \n + 0.5 SUM <rasb> w[rabs] E2[ra,bs]
""")
hd = h1 + h2
hfull = P("SUM <mn> h[mn] E1[m,n] \n + 0.5 SUM <mnxy> w[mnxy] E2[mn,xy]")

# convert < E1[p,a] E1[q,b] > ("dm2") to < E2[pq,ab] > ("E2"), etc.
# E2[pq,ab] = E1[p,a] E1[q,b] - delta[aq] E1[p,b]
pdm_eqs = [
    "E1[p,a] = E1[p,a]\n - E1[p,a]\n + dm1[pa]",
    "E2[pq,ab] = E2[pq,ab]\n - E1[p,a] E1[q,b]\n + dm2[paqb]",
    "E3[pqg,abc] = E3[pqg,abc]\n - E1[p,a] E1[q,b] E1[g,c]\n + dm3[paqbgc]",
    "E4[abcd,efgh] = E4[abcd,efgh]\n - E1[a,e] E1[b,f] E1[c,g] E1[d,h]\n + dm4[aebfcgdh]"
]

for k, eq in enumerate(pdm_eqs):
    name, expr = PD(eq)
    pdm_eqs[k] = SP(expr).to_einsum(name)


# def of ic-mrrept2 sub-spaces
sub_spaces = {
    "ijrs+": "E1[r,i] E1[s,j] \n + E1[s,i] E1[r,j]",
    "ijrs-": "E1[r,i] E1[s,j] \n - E1[s,i] E1[r,j]",
    "rsiap+": "E1[r,i] E1[s,a] \n + E1[s,i] E1[r,a]",
    "rsiap-": "E1[r,i] E1[s,a] \n - E1[s,i] E1[r,a]",
    "ijrap+": "E1[r,j] E1[a,i] \n + E1[r,i] E1[a,j]",
    "ijrap-": "E1[r,j] E1[a,i] \n - E1[r,i] E1[a,j]",
    "rsabpq+": "E1[r,b] E1[s,a] \n + E1[s,b] E1[r,a]",
    "rsabpq-": "E1[r,b] E1[s,a] \n - E1[s,b] E1[r,a]",
    "ijabpq+": "E1[b,i] E1[a,j] \n + E1[b,j] E1[a,i]",
    "ijabpq-": "E1[b,i] E1[a,j] \n - E1[b,j] E1[a,i]",
    "irabpq1": "E1[r,i] E1[a,b]",
    "irabpq2": "E1[a,i] E1[r,b]",
    "rabcpqg*": "E1[r,b] E1[a,c]",
    "iabcpqg*": "E1[b,i] E1[a,c]"
}

ener_eqs = {} # Hamiltonian expectations
ener2_eqs = {} # Hamiltonian expectations
rhhk_eqs = {} # rhs equations

for key, expr in sub_spaces.items():
    l = len(key)
    ket_bra_map = { k: v for k, v in zip(key[9 - l:4 - l], key[4 - l:-1]) }
    ket = P(expr)
    bra = ket.index_map(MapStrStr(ket_bra_map))
    rhhk_eqs[key] = Rhs(bra, hfull)
    ener_eqs[key] = Comm(bra, hd, ket)
    if key[-1] in "12":
        ket2 = P(sub_spaces[key[:-1] + ('1' if key[-1] == '2' else '2')])
        ener2_eqs[key] = Comm(bra, hd, ket2)

allowed_perms = {"AAAA", "EAAA", "EAIA", "EAAI", "AAIA", "EEIA",
    "EAII", "EEAA", "AAII", "EEII", "EEEE", "IIII",
    "EIEI", "EAEA", "AIAI", "IIIE", "IAIE", "IIIA"}

def fix_eri_permutations(eq):
    imap = {WickIndexTypes.External: "E",  WickIndexTypes.Active: "A",
        WickIndexTypes.Inactive: "I"}
    for term in eq.terms:
        for wt in term.tensors:
            if wt.name == "w":
                k = ''.join([imap[wi.types] for wi in wt.indices])
                if k not in allowed_perms:
                    found = False
                    for perm in wt.perms:
                        wtt = wt * perm
                        k = ''.join([imap[wi.types] for wi in wtt.indices])
                        if k in allowed_perms:
                            wt.indices = wtt.indices
                            found = True
                            break
                    assert found

for eq in [*ener_eqs.values(), *ener2_eqs.values(), *rhhk_eqs.values()]:
    fix_eri_permutations(eq)

def _key_idx(key):
    t = [WickIndexTypes.Inactive, WickIndexTypes.Active, WickIndexTypes.External]
    return [t.index(wi.types) for wi in PT("x[%s]" % key).indices]

def _grid_restrict(key, grid, restrict_cas, no_eq):
    ixx = []
    wis = PT("x[%s]" % key).indices
    contig = lambda a, b: b == chr(ord(a) + 1)
    for i, wi in enumerate(wis):
        if not restrict_cas and wi.types == WickIndexTypes.Active:
            continue
        if i != 0 and wis[i].types == wis[i - 1].types:
            if wi.types == WickIndexTypes.Active:
                if not contig(wis[i - 1].name, wis[i].name):
                    continue
                if i + 1 < len(wis) and contig(wis[i].name, wis[i + 1].name):
                    continue
                if i - 2 >= 0 and contig(wis[i - 2].name, wis[i - 1].name):
                    continue
            if no_eq:
                ixx.append(grid[i - 1] < grid[i])
            else:
                ixx.append(grid[i - 1] <= grid[i])
    return np.bitwise_and.reduce(ixx)

def _linear_solve(a, b):
    c = np.zeros_like(b)
    for i in range(len(a)):
        c[i] = np.linalg.lstsq(a[i], b[i], rcond=None)[0]
    return c

from pyscf import lib

def kernel(ic, mc=None, mo_coeff=None, pdms=None, eris=None, root=None):
    if mc is None:
        mc = ic._mc
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if root is None and hasattr(ic, 'root'):
        root = ic.root
    ic.root = root
    ic.mo_coeff = mo_coeff
    ic.ci = mc.ci
    ic.mo_energy = mc.mo_energy
    tt = time.perf_counter()
    if pdms is None:
        t = time.perf_counter()
        pdms = eri_helper.init_pdms(mc=mc, pdm_eqs=pdm_eqs, root=root)
        tpdms = time.perf_counter() - t
    if eris is None:
        t = time.perf_counter()
        eris = eri_helper.init_eris(mc=mc, mo_coeff=mo_coeff, mrci=True)
        teris = time.perf_counter() - t
    ic.eris = eris
    assert isinstance(eris, eri_helper._ChemistsERIs)
    E1, E2, E3, E4 = pdms
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nvirt = len(ic.mo_energy) - nocc
    mdict = {
        "E1": E1, "E2": E2, "E3": E3, "E4": E4,
        "deltaII": np.eye(ncore), "deltaEE": np.eye(nvirt),
        "deltaAA": np.eye(ncas), 
        "ident1": np.ones((1, )),
        "ident2": np.ones((1, 1, )),
        "ident3": np.ones((1, 1, 1, )),
        **{ 'h' + a + b: eris.get_h1(a + b) for a in 'IAE' for b in 'IAE' },
        **{ 'w' + k: eris.get_phys(k) for k in allowed_perms }
    }

    ic.sub_eners = {}
    ic.sub_times = {'pdms': tpdms, 'eris': teris}
    for key in sub_spaces:
        t = time.perf_counter()
        l = len(key)
        skey = key[:9 - l]
        if E4 is None and "abc" in key:
            irkmap = {'i': 'aaac', 'r': 'aaav'}
            ic.sub_eners[skey] = dmrg_helper.dmrg_response_singles(mc, eris, E1,
                irkmap[key[0]], theory='mrrept2', root=root)
            lib.logger.note(ic, "E(%s-%4s) = %20.14f",
                ic.__class__.__name__.replace("IC", "UC"), 'mps' + skey, ic.sub_eners[skey])
            ic.sub_times[skey] = time.perf_counter() - t
            continue
        if key[-1] == '2':
            continue
        rkey = key[:9 - l] + key[4 - l:-1]
        hkey = key[:-1]
        rhhk = np.zeros([[ncore, ncas, nvirt][ix] for ix in _key_idx(rkey)])
        ener = np.zeros([[ncore, ncas, nvirt][ix] for ix in _key_idx(hkey)])
        pt2_eqs = rhhk_eqs[key].to_einsum(PT("rhhk[%s]" % rkey))
        pt2_eqs += ener_eqs[key].to_einsum(PT("ener[%s]" % hkey))
        if key[-1] == '1':
            key2 = key[:-1] + '2'
            rhhk2 = np.zeros_like(rhhk)
            ener12 = np.zeros_like(ener)
            ener21 = np.zeros_like(ener)
            ener22 = np.zeros_like(ener)
            pt2_eqs += rhhk_eqs[key2].to_einsum(PT("rhhk2[%s]" % rkey))
            pt2_eqs += ener2_eqs[key].to_einsum(PT("ener12[%s]" % hkey))
            pt2_eqs += ener_eqs[key2].to_einsum(PT("ener22[%s]" % hkey))
            pt2_eqs += ener2_eqs[key2].to_einsum(PT("ener21[%s]" % hkey))
            exec(pt2_eqs, globals(), {
                "rhhk": rhhk, "rhhk2": rhhk2, "ener": ener, "ener12": ener12,
                "ener22": ener22, "ener21": ener21, **mdict
            })
            rhhk = np.concatenate((rhhk[..., None], rhhk2[..., None]), axis=-1)
            ener = np.concatenate((ener[..., None], ener12[..., None],
                ener21[..., None], ener22[..., None]), axis=-1)
            dcas = ncas ** (len(key) - 5)
            xr = rhhk.reshape(-1, dcas * 2)
            xh = ener.reshape(-1, dcas, dcas, 2, 2).transpose(0, 1, 3, 2, 4)
            xh = xh.reshape(-1, dcas * 2, dcas * 2)
        else:
            exec(pt2_eqs, globals(), { "rhhk": rhhk, "ener": ener, **mdict })
            restrict_cas = key[-1] in '+-'
            if len(key) - 5 == 2 and restrict_cas:
                dcas = ncas * (ncas + (1 if key[-1] == '+' else -1)) // 2
            else:
                dcas = ncas ** (len(key) - 5)
            if 9 - len(key) >= 2:
                grid = np.indices(rhhk.shape, dtype=np.int16)
                idx = _grid_restrict(rkey, grid, restrict_cas, key[-1] == '-')
                xr = rhhk[idx].reshape(-1, dcas)
                grid = np.indices(ener.shape, dtype=np.int16)
                idx = _grid_restrict(hkey, grid, restrict_cas, key[-1] == '-')
                xh = ener[idx].reshape(-1, dcas, dcas)
            else:
                xr = rhhk.reshape(-1, dcas)
                xh = ener.reshape(-1, dcas, dcas)
        if skey not in ic.sub_eners:
            ic.sub_eners[skey] = -(_linear_solve(xh, xr) * xr).sum()
            ic.sub_times[skey] = time.perf_counter() - t
        else:
            ic.sub_eners[skey] += -(_linear_solve(xh, xr) * xr).sum()
            ic.sub_times[skey] += time.perf_counter() - t
        if key[-1] in "-1*":
            lib.logger.note(ic, "E(%s-%4s) = %20.14f",
                ic.__class__.__name__, skey, ic.sub_eners[skey])
    ic.e_corr = sum(ic.sub_eners.values())
    ic.sub_times['total'] = time.perf_counter() - tt
    lib.logger.note(ic, 'E(%s) = %.16g  E_corr_pt = %.16g',
        ic.__class__.__name__, ic.e_tot, ic.e_corr)
    lib.logger.note(ic, "Timings = %s",
        " | ".join(["%s = %7.2f" % (k, v) for k, v in ic.sub_times.items()]))

class WickICMRREPT2(lib.StreamObject):
    def __init__(self, mc):
        self._mc = mc
        assert mc.canonicalization
        self._scf = mc._scf
        self.mol = self._scf.mol
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.e_corr = None

    @property
    def e_tot(self):
        if hasattr(self._mc, 'e_states') and hasattr(self, 'root'):
            return np.asarray(self.e_corr) + self._mc.e_states[self.root]
        elif hasattr(self, 'root') and isinstance(self._mc.e_tot, np.ndarray):
            return np.asarray(self.e_corr) + self._mc.e_tot[self.root]
        else:
            return np.asarray(self.e_corr) + self._mc.e_tot

    kernel = kernel

ICMRREPT2 = WickICMRREPT2

if __name__ == "__main__":

    from pyscf import gto, scf, mcscf

    mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2)
    mf = scf.RHF(mol).run(conv_tol=1E-20)

    # Example 1 - single state
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.fcisolver.conv_tol = 1e-14
    mc.conv_tol = 1e-12
    mc.run()
    wsc = WickICMRREPT2(mc).run()
    # converged SCF energy = -149.608181589162
    # CASSCF energy = -149.708657771221
    # E(WickICMRREPT2) = -150.0162455392724  E_corr_pt = -0.3075877680517728

    # Example 2 - CASCI multi-state
    mc2 = mcscf.CASCI(mf, 6, 8)
    mc2.fcisolver.nroots = 3
    mc2.fcisolver.conv_tol = 1e-14
    mc2.canonicalization = True
    mc2.kernel(mc.mo_coeff)
    # [ -149.708657771221 -149.480534726188 -149.480534726188 ]

    wsc = WickICMRREPT2(mc2).run(root=0)
    wsc = WickICMRREPT2(mc2).run(root=1)
    wsc = WickICMRREPT2(mc2).run(root=2)
    # [ -0.3072831603857041 -0.301737869046007 -0.3018189555282041 ]

    # Example 3 - CASSCF state-average
    mc = mcscf.CASSCF(mf, 6, 8)
    mc.state_average_([1 / 3] * 3)
    mc.fcisolver.conv_tol = 1e-14
    mc.canonicalization = True
    mc.run()
    # [ -149.706807304716 -149.484319800832 -149.484319800828 ]
    wsc = WickICMRREPT2(mc).run(root=0)
    wsc = WickICMRREPT2(mc).run(root=1)
    wsc = WickICMRREPT2(mc).run(root=2)
    # E(WickICMRREPT2) = -150.0147593908998  E_corr_pt = -0.3079520861841749
    # E(WickICMRREPT2) = -149.7801934111490  E_corr_pt = -0.2958736103171785
    # E(WickICMRREPT2) = -149.7801590818412  E_corr_pt = -0.2958392810134889
