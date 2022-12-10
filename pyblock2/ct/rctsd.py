
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

# Author: Ke Liao <ke.liao.life@gmail.com>
# Based on rccsd.py


try:
    from block2 import WickIndexTypes, WickIndex, WickExpr, WickTensor, WickPermutation
    from block2 import MapWickIndexTypesSet, MapPStrIntVectorWickPermutation
    from block2 import MapStrPWickTensorExpr, MapStrStr
except ImportError:
    raise RuntimeError("block2 needs to be compiled with '-DUSE_IC=ON'!")
import numpy as np

def init_parsers():

    idx_map = MapWickIndexTypesSet()
    idx_map[WickIndexTypes.External] = WickIndex.parse_set("pqrsijklmnoabcdefg")

    perm_map = MapPStrIntVectorWickPermutation()
    perm_map[("v", 4)] = WickPermutation.qc_chem()
    perm_map[("t", 2)] = WickPermutation.non_symmetric()
    perm_map[("t", 4)] = WickPermutation.pair_symmetric(2, False)

    defs = MapStrPWickTensorExpr()
    p = lambda x: WickExpr.parse(x, idx_map, perm_map).substitute(defs)
    pt = lambda x: WickTensor.parse(x, idx_map, perm_map)
    pd = lambda x: WickExpr.parse_def(x, idx_map, perm_map)

    return p, pt, pd, defs

P, PT, PD, DEF = init_parsers() # parsers
SP = lambda x: x.simplify() # just simplify
FC = lambda x: x.expand().simplify() # fully contracted
Z = P("") # zero

# definitions

#DEF["h"] = PD("h[pq] = f[pq] \n - 2.0 SUM <j> v[pqjj] \n + SUM <j> v[pjjq]")
f = P("SUM <pq> h[pq] E1[p,q] + 2.0 SUM <j> v[pqjj] E2[pj,qj] - SUM <j> v[pjjq] E2[pj,jq]")
h1 = P("SUM <pq> h[pq] E1[p,q]")
h2 = P("0.5 SUM <pqrs> v[prqs] E2[pq,rs]")
t1 = P("SUM <ai> t[ia] E1[a,i] - SUM <ai> t[ia] E1[i,a]")
#t2 = P("0.5 SUM <abij> t[ijab] E1[a,i] E1[b,j] - 0.5 SUM <abij> t[ijab] E1[i,a] E1[j,b]")
t2 = P("0.5 SUM <abij> t[ijab] E2[ab,ij] - 0.5 SUM <abij> t[ijab] E2[ij, ab]")
ex1 = P("E1[i,a]")
ex2 = P("E1[i,a] E1[j,b]")
ehf = P("2 SUM <i> h[ii] \n + 2 SUM <ij> v[iijj] \n - SUM <ij> v[ijji]")

h = SP(h1)
print("H = ", h)
t = SP(t2)
print("T = ", t)

#HBarTerms = [
#    h, h ^ t,
#    0.5 * ((h ^ t1) ^ t1) + ((h ^ t2) ^ t1) + 0.5 * ((h ^ t2) ^ t2),
#    (1 / 6.0) * (((h ^ t1) ^ t1) ^ t1) + 0.5 * (((h ^ t2) ^ t1) ^ t1),
#    (1 / 24.0) * ((((h ^ t1) ^ t1) ^ t1) ^ t1)
#]

HBarTerms = [h ^ t]

print("HBar = ", h ^ t)
print("[h1, t1] = ", FC(h ^ t))
Hbar = FC(ex1 * sum(HBarTerms, Z))
print("<ai|Hbar|0> = ", Hbar)

def fix_eri_permutations(eq):
    imap = {WickIndexTypes.External: "E",  WickIndexTypes.Inactive: "I"}
    allowed_perms = {"IIII", "IEII", "IIEE", "IEEI", "IEIE", "IEEE", "EEEE"}
    for term in eq.terms:
        for wt in term.tensors:
            if wt.name == "v":
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
