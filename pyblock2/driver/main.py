#! /usr/bin/env python
"""
block2 wrapper.

Author:
    Huanchen Zhai
    Zhi-Hao Cui
"""

from block2 import SZ, SU2, Global, OpNamesSet, NoiseTypes, DecompositionTypes, Threading, ThreadingTypes
from block2 import init_memory, release_memory, set_mkl_num_threads, read_occ, TruncationTypes
from block2 import VectorUInt8, VectorUBond, VectorDouble, PointGroup, DoubleFPCodec
from block2 import Random, FCIDUMP, QCTypes, SeqTypes, TETypes, OpNames, VectorInt, VectorUInt16
from block2 import MatrixFunctions, KuhnMunkres
import numpy as np
import time
import os
import sys

from parser import parse, orbital_reorder, read_integral, format_schedule

DEBUG = True

if len(sys.argv) > 1:
    fin = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "pre":
        pre_run = True
    else:
        pre_run = False
    if len(sys.argv) > 2 and sys.argv[2] == "run":
        no_pre_run = True
    else:
        no_pre_run = False
else:
    raise ValueError("""
        Usage: either:
            (A) python main.py dmrg.conf
            (B) Step 1: python main.py dmrg.conf pre
                Step 2: python main.py dmrg.conf run
    """)

dic = parse(fin)
if "nonspinadapted" in dic:
    from block2 import VectorSZ as VectorSL
    from block2.sz import MultiMPS, MultiMPSInfo
    from block2.sz import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC, MPICommunicator
    from block2.sz import PDM1MPOQC, NPC1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC, NoTransposeRule
    from block2.sz import Expect, DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
    from block2.sz import ParallelRuleQC, ParallelMPO, ParallelMPS, IdentityMPO, VectorMPS, PDM2MPOQC
    from block2.sz import ParallelRulePDM1QC, ParallelRulePDM2QC, ParallelRuleIdentity, ParallelRuleOneBodyQC
    from block2.sz import AntiHermitianRuleQC, TimeEvolution, Linear
    from block2.sz import trans_state_info_to_su2 as trans_si
    from block2.su2 import MPSInfo as TrMPSInfo
    from block2.su2 import trans_mps_info_to_sz as trans_mi, VectorStateInfo as TrVectorStateInfo
    SX = SZ
    TrSX = SU2
else:
    from block2 import VectorSU2 as VectorSL
    from block2.su2 import MultiMPS, MultiMPSInfo
    from block2.su2 import HamiltonianQC, MPS, MPSInfo, ParallelRuleQC, MPICommunicator
    from block2.su2 import PDM1MPOQC, NPC1MPOQC, SimplifiedMPO, Rule, RuleQC, MPOQC, NoTransposeRule
    from block2.su2 import Expect, DMRG, MovingEnvironment, OperatorFunctions, CG, TensorFunctions, MPO
    from block2.su2 import ParallelRuleQC, ParallelMPO, ParallelMPS, IdentityMPO, VectorMPS, PDM2MPOQC
    from block2.su2 import ParallelRulePDM1QC, ParallelRulePDM2QC, ParallelRuleIdentity, ParallelRuleOneBodyQC
    from block2.su2 import AntiHermitianRuleQC, TimeEvolution, Linear
    from block2.su2 import trans_state_info_to_sz as trans_si
    from block2.sz import MPSInfo as TrMPSInfo
    from block2.sz import trans_mps_info_to_su2 as trans_mi, VectorStateInfo as TrVectorStateInfo
    SX = SU2
    TrSX = SZ

# MPI
MPI = MPICommunicator()
from mpi4py import MPI as PYMPI
comm = PYMPI.COMM_WORLD
outputlevel = 2


def _print(*args, **kwargs):
    if MPI.rank == 0 and outputlevel > -1:
        kwargs["flush"] = True
        print(*args, **kwargs)


tx = time.perf_counter()

# input parameters
Random.rand_seed(1234)
outputlevel = int(dic.get("outputlevel", 2))
if DEBUG:
    _print("\n" + "*" * 34 + " INPUT START " + "*" * 34)
    for key, val in dic.items():
        if key == "schedule":
            pval = format_schedule(val)
            for ipv, pv in enumerate(pval):
                _print("%-25s %40s" % (key if ipv == 0 else "", pv))
        else:
            _print("%-25s %40s" % (key, val))
    _print("*" * 34 + " INPUT END   " + "*" * 34 + "\n")

scratch = dic.get("prefix", "./nodex/")
restart_dir = dic.get("restart_dir", None)
restart_dir_per_sweep = dic.get("restart_dir_per_sweep", None)
n_threads = int(dic.get("num_thrds", 28))
mkl_threads = int(dic.get("mkl_thrds", 1))
bond_dims, dav_thrds, noises = dic["schedule"]
sweep_tol = float(dic.get("sweep_tol", 1e-6))

if dic.get("trunc_type", "physical") == "physical":
    trunc_type = TruncationTypes.Physical
else:
    trunc_type = TruncationTypes.Reduced
if dic.get("decomp_type", "density_matrix") == "density_matrix":
    decomp_type = DecompositionTypes.DensityMatrix
else:
    decomp_type = DecompositionTypes.SVD
if dic.get("te_type", "rk4") == "rk4":
    te_type = TETypes.RK4
else:
    te_type = TETypes.TangentSpace

if MPI is not None and MPI.rank == 0:
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
    if restart_dir is not None and not os.path.isdir(restart_dir):
        os.mkdir(restart_dir)
    os.environ['TMPDIR'] = scratch
if MPI is not None:
    MPI.barrier()

# global settings
memory = int(int(dic.get("mem", "40").split()[0]) * 1e9)
fp_cps_cutoff = float(dic.get("fp_cps_cutoff", 1E-16))
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=scratch)
# ZHC NOTE nglobal_threads, nop_threads, MKL_NUM_THREADS
Global.threading = Threading(
    ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global,
    n_threads * mkl_threads, n_threads, mkl_threads)
Global.threading.seq_type = SeqTypes.Tasked
Global.frame.fp_codec = DoubleFPCodec(fp_cps_cutoff, 1024)
Global.frame.load_buffering = False
Global.frame.save_buffering = False
Global.frame.use_main_stack = False
Global.frame.minimal_disk_usage = True
if restart_dir is not None:
    Global.frame.restart_dir = restart_dir
if restart_dir_per_sweep is not None:
    Global.frame.restart_dir_per_sweep = restart_dir_per_sweep
_print(Global.frame)
_print(Global.threading)

if MPI is not None:
    prule = ParallelRuleQC(MPI)
    prule_one_body = ParallelRuleOneBodyQC(MPI)
    prule_pdm1 = ParallelRulePDM1QC(MPI)
    prule_pdm2 = ParallelRulePDM2QC(MPI)
    prule_ident = ParallelRuleIdentity(MPI)


# prepare hamiltonian
if pre_run or not no_pre_run:
    nelec = [int(x) for x in dic["nelec"].split()]
    spin = [int(x) for x in dic.get("spin", "0").split()]
    isym = [int(x) for x in dic.get("irrep", "1").split()]
    if "orbital_rotation" in dic:
        orb_sym = np.load(scratch + "/nat_orb_sym.npy")
        kappa = np.load(scratch + "/nat_kappa.npy")
        kappa = kappa.flatten()
        n_sites = len(orb_sym)
        fcidump = FCIDUMP()
        fcidump.initialize_h1e(n_sites, nelec[0], spin[0], isym[0], 0.0, kappa)
        assert "nofiedler" in dic or "noreorder" in dic
        if "target_t" not in dic:
            dic["target_t"] = "1"
    else:
        orb_sym = None
        fints = dic["orbitals"]
        if open(fints, 'rb').read(4) != b'\x89HDF':
            fcidump = FCIDUMP()
            fcidump.read(fints)
            fcidump.params["nelec"] = str(nelec[0])
            fcidump.params["ms2"] = str(spin[0])
            fcidump.params["isym"] = str(isym[0])
        else:
            fcidump = read_integral(fints, nelec[0], spin[0], isym=isym[0])
    if "nofiedler" in dic or "noreorder" in dic:
        orb_idx = None
    else:
        if "gaopt" in dic:
            orb_idx = orbital_reorder(fcidump, method='gaopt ' + dic["gaopt"])
            _print("using gaopt reorder = ", orb_idx)
        elif "reorder" in dic:
            orb_idx = orbital_reorder(
                fcidump, method='manual ' + dic["reorder"])
            _print("using manual reorder = ", orb_idx)
        elif "irrep_reorder" in dic:
            orb_idx = orbital_reorder(
                fcidump, method='irrep ' + dic.get("sym", "d2h"))
            _print("using irrep reorder = ", orb_idx)
            _print("reordered irrep = ", fcidump.orb_sym)
        else:
            orb_idx = orbital_reorder(fcidump, method='fiedler')
            _print("using fiedler reorder = ", orb_idx)
        np.save(scratch + '/orbital_reorder.npy', orb_idx)

    swap_pg = getattr(PointGroup, "swap_" + dic.get("sym", "d2h"))

    _print("read integral finished", time.perf_counter() - tx)

    vacuum = SX(0)
    target = SX(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
    targets = []
    for inelec in nelec:
        for ispin in spin:
            for iisym in isym:
                targets.append(SX(inelec, ispin, swap_pg(iisym)))
    targets = VectorSL(targets)
    n_sites = fcidump.n_sites
    if orb_sym is None:
        orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
    if "symmetrize_ints" in dic:
        sym_error = fcidump.symmetrize(orb_sym)
        _print("integral sym error = %12.4g" % sym_error)
    hamil = HamiltonianQC(vacuum, n_sites, VectorUInt8(orb_sym), fcidump)
else:
    if "nofiedler" in dic or "noreorder" in dic:
        orb_idx = None
    else:
        orb_idx = np.load(scratch + '/orbital_reorder.npy')
    orb_sym = None
    fcidump = None

# parallelization over sites
# use keyword: conn_centers auto 5      (5 is number of procs)
#          or  conn_centers 10 20 30 40 (list of connection site indices)
if "conn_centers" in dic:
    assert MPI is not None
    cc = dic["conn_centers"].split()
    if cc[0] == "auto":
        ncc = int(cc[1])
        conn_centers = list(
            np.arange(0, n_sites * ncc, n_sites, dtype=int) // ncc)[1:]
        assert len(conn_centers) == ncc - 1
    else:
        conn_centers = [int(xcc) for xcc in cc]
    _print("using connection sites: ", conn_centers)
    assert MPI.size % (len(conn_centers) + 1) == 0
    mps_prule = prule
    prule = prule.split(MPI.size // (len(conn_centers) + 1))
else:
    conn_centers = None

if dic.get("warmup", None) == "occ":
    _print("using occ init")
    assert "occ" in dic
    if len(dic["occ"].split()) == 1:
        with open(dic["occ"], 'r') as ofin:
            dic["occ"] = ofin.readlines()[0]
    occs = VectorDouble([float(occ)
                         for occ in dic["occ"].split() if len(occ) != 0])
    if orb_idx is not None:
        occs = FCIDUMP.array_reorder(occs, VectorUInt16(orb_idx))
        _print("using reordered occ init")
    assert len(occs) == n_sites or len(occs) == n_sites * 2
    cbias = float(dic.get("cbias", 0.0))
    if cbias != 0.0:
        if len(occs) == n_sites:
            occs = VectorDouble(
                [c - cbias if c >= 1 else c + cbias for c in occs])
        else:
            occs = VectorDouble(
                [c - cbias if c >= 0.5 else c + cbias for c in occs])
    bias = float(dic.get("bias", 1.0))
else:
    occs = None

dot = 1 if "onedot" in dic else 2
nroots = int(dic.get("nroots", 1))
mps_tags = dic.get("mps_tags", "KET").split()
read_tags = dic.get("read_mps_tags", "KET").split()
soc = "soc" in dic
overlap = "overlap" in dic

# prepare mps
if len(mps_tags) > 1 or "compression" in dic:
    nroots = len(mps_tags)
    mps = None
    mps_info = None
    forward = False
elif "fullrestart" in dic:
    _print("full restart")
    mps_info = MPSInfo(0) if nroots == 1 and len(
        targets) == 1 else MultiMPSInfo(0)
    mps_info.load_data(scratch + "/mps_info.bin")
    mps_info.tag = mps_tags[0]
    mps_info.load_mutable()
    max_bdim = max([x.n_states_total for x in mps_info.left_dims])
    if mps_info.bond_dim < max_bdim:
        mps_info.bond_dim = max_bdim
    max_bdim = max([x.n_states_total for x in mps_info.right_dims])
    if mps_info.bond_dim < max_bdim:
        mps_info.bond_dim = max_bdim
    mps = MPS(mps_info) if nroots == 1 and len(
        targets) == 1 else MultiMPS(mps_info)
    mps.load_data()
    if mps.dot != dot and nroots == 1:
        if MPI is not None:
            MPI.barrier()
        mps.dot = dot
        mps.save_data()
        if MPI is not None:
            MPI.barrier()
    if nroots != 1:
        mps.nroots = nroots
        mps.wfns = mps.wfns[:nroots]
        mps.weights = mps.weights[:nroots]
    weights = dic.get("weights", None)
    if weights is not None:
        mps.weights = VectorDouble([float(x) for x in weights.split()])
    mps.load_mutable()
    forward = mps.center == 0
    if mps.canonical_form[mps.center] == 'L' and mps.center != mps.n_sites - mps.dot:
        mps.center += 1
        forward = True
    elif mps.canonical_form[mps.center] == 'C' and mps.center != 0:
        mps.center -= 1
        forward = False
    elif mps.center == mps.n_sites - 1 and mps.dot == 2 and nroots == 1:
        if MPI is not None:
            MPI.barrier()
        if mps.canonical_form[mps.center] == 'K':
            cg = CG(200)
            cg.initialize()
            mps.move_left(cg, prule)
        mps.center = mps.n_sites - 2
        mps.save_data()
        forward = False
        if MPI is not None:
            MPI.barrier()
elif pre_run or not no_pre_run:
    if "trans_mps_info" in dic:
        assert nroots == 1 and len(targets) == 1
        tr_vacuum = TrSX(vacuum.n, abs(vacuum.twos), vacuum.pg)
        tr_target = TrSX(target.n, abs(target.twos), target.pg)
        tr_basis = TrVectorStateInfo([trans_si(b) for b in hamil.basis])
        tr_mps_info = TrMPSInfo(n_sites, tr_vacuum, tr_target, tr_basis)
        assert "full_fci_space" not in dic
        tr_mps_info.tag = mps_tags[0]
        if occs is None:
            tr_mps_info.set_bond_dimension(bond_dims[0])
        else:
            tr_mps_info.set_bond_dimension_using_occ(
                bond_dims[0], occs, bias=bias)
        mps_info = trans_mi(tr_mps_info, target)
    else:
        if nroots == 1 and len(targets) == 1:
            mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
        else:
            _print('TARGETS = ', list(targets))
            mps_info = MultiMPSInfo(n_sites, vacuum, targets, hamil.basis)
        if "full_fci_space" in dic:
            mps_info.set_bond_dimension_full_fci()
        mps_info.tag = mps_tags[0]
        if occs is None:
            mps_info.set_bond_dimension(bond_dims[0])
        else:
            mps_info.set_bond_dimension_using_occ(
                bond_dims[0], occs, bias=bias)
    if MPI is None or MPI.rank == 0:
        mps_info.save_data(scratch + '/mps_info.bin')
        mps_info.save_data(scratch + '/%s-mps_info.bin' % mps_tags[0])
    if conn_centers is not None:
        assert nroots == 1
        mps = ParallelMPS(mps_info.n_sites, 0, dot, mps_prule)
    elif nroots != 1 or len(targets) != 1:
        mps = MultiMPS(n_sites, 0, dot, nroots)
        weights = dic.get("weights", None)
        if weights is not None:
            mps.weights = VectorDouble([float(x) for x in weights.split()])
    else:
        mps = MPS(n_sites, 0, dot)
    mps.initialize(mps_info)
    mps.random_canonicalize()
    forward = mps.center == 0
else:
    mps_info = MPSInfo(0) if nroots == 1 and len(
        targets) == 1 else MultiMPSInfo(0)
    mps_info.load_data(scratch + "/mps_info.bin")
    mps_info.tag = mps_tags[0]
    if occs is None:
        mps_info.set_bond_dimension(bond_dims[0])
    else:
        mps_info.set_bond_dimension_using_occ(bond_dims[0], occs, bias=bias)
    if conn_centers is not None:
        assert nroots == 1
        mps = ParallelMPS(mps_info.n_sites, 0, dot, mps_prule)
    elif nroots != 1 or len(targets) != 1:
        mps = MultiMPS(n_sites, 0, dot, nroots)
        weights = dic.get("weights", None)
        if weights is not None:
            mps.weights = VectorDouble([float(x) for x in weights.split()])
    else:
        mps = MPS(mps_info.n_sites, 0, dot)
    mps.initialize(mps_info)
    mps.random_canonicalize()
    forward = mps.center == 0

if mps is not None:
    _print("MPS = ", mps.canonical_form, mps.center, mps.dot)
    _print("GS INIT MPS BOND DIMS = ", ''.join(
        ["%6d" % x.n_states_total for x in mps_info.left_dims]))

if conn_centers is not None and "fullrestart" in dic:
    assert mps.dot == 2
    mps = ParallelMPS(mps, mps_prule)
    if mps.canonical_form[0] == 'C' and mps.canonical_form[1] == 'R':
        mps.canonical_form = 'K' + mps.canonical_form[1:]
    elif mps.canonical_form[-1] == 'C' and mps.canonical_form[-2] == 'L':
        mps.canonical_form = mps.canonical_form[:-1] + 'S'
        mps.center = mps.n_sites - 1

has_tran = "restart_tran_onepdm" in dic or "tran_onepdm" in dic \
    or "restart_tran_twopdm" in dic or "tran_twopdm" in dic \
    or "restart_tran_oh" in dic or "tran_oh" in dic or "compression" in dic
has_2pdm = "restart_tran_twopdm" in dic or "tran_twopdm" in dic \
    or "restart_twopdm" in dic or "twopdm" in dic
anti_herm = "orbital_rotation" in dic
one_body_only = "orbital_rotation" in dic

# prepare mpo
if pre_run or not no_pre_run:
    # mpo for dmrg
    _print("build mpo", time.perf_counter() - tx)
    mpo = MPOQC(hamil, QCTypes.Conventional)
    _print("simpl mpo", time.perf_counter() - tx)
    mpo = SimplifiedMPO(mpo, AntiHermitianRuleQC(RuleQC()) if anti_herm else RuleQC(),
                        True, True, OpNamesSet((OpNames.R, OpNames.RD)))
    _print("simpl mpo finished", time.perf_counter() - tx)

    _print('GS MPO BOND DIMS = ', ''.join(
        ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))

    if MPI is None or MPI.rank == 0:
        mpo.save_data(scratch + '/mpo.bin')

    # mpo for 1pdm
    _print("build 1pdm mpo", time.perf_counter() - tx)
    pmpo = PDM1MPOQC(hamil, 1 if soc else 0)
    pmpo = SimplifiedMPO(pmpo,
                         NoTransposeRule(RuleQC()) if has_tran else RuleQC(),
                         True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    if MPI is None or MPI.rank == 0:
        pmpo.save_data(scratch + '/mpo-1pdm.bin')

    if has_2pdm:
        # mpo for 2pdm
        _print("build 2pdm mpo", time.perf_counter() - tx)
        p2mpo = PDM2MPOQC(hamil)
        p2mpo = SimplifiedMPO(p2mpo,
                              NoTransposeRule(
                                  RuleQC()) if has_tran else RuleQC(),
                              True, True, OpNamesSet((OpNames.R, OpNames.RD)))

        if MPI is None or MPI.rank == 0:
            p2mpo.save_data(scratch + '/mpo-2pdm.bin')

    # mpo for particle number correlation
    _print("build 1npc mpo", time.perf_counter() - tx)
    nmpo = NPC1MPOQC(hamil)
    nmpo = SimplifiedMPO(nmpo, RuleQC(), True, True,
                         OpNamesSet((OpNames.R, OpNames.RD)))

    if MPI is None or MPI.rank == 0:
        nmpo.save_data(scratch + '/mpo-1npc.bin')

    # mpo for identity operator
    _print("build identity mpo", time.perf_counter() - tx)
    impo = IdentityMPO(hamil)
    impo = SimplifiedMPO(impo,
                         NoTransposeRule(RuleQC()) if has_tran else RuleQC(),
                         True, True, OpNamesSet((OpNames.R, OpNames.RD)))

    if MPI is None or MPI.rank == 0:
        impo.save_data(scratch + '/mpo-ident.bin')

else:
    mpo = MPO(0)
    mpo.load_data(scratch + '/mpo.bin')
    cg = CG(200)
    cg.initialize()
    mpo.tf = TensorFunctions(OperatorFunctions(cg))

    _print('GS MPO BOND DIMS = ', ''.join(
        ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))

    pmpo = MPO(0)
    pmpo.load_data(scratch + '/mpo-1pdm.bin')
    pmpo.tf = TensorFunctions(OperatorFunctions(cg))

    _print('1PDM MPO BOND DIMS = ', ''.join(
        ["%6d" % (x.m * x.n) for x in pmpo.left_operator_names]))

    if has_2pdm:
        p2mpo = MPO(0)
        p2mpo.load_data(scratch + '/mpo-2pdm.bin')
        p2mpo.tf = TensorFunctions(OperatorFunctions(cg))

        _print('2PDM MPO BOND DIMS = ', ''.join(
            ["%6d" % (x.m * x.n) for x in p2mpo.left_operator_names]))

    nmpo = MPO(0)
    nmpo.load_data(scratch + '/mpo-1npc.bin')
    nmpo.tf = TensorFunctions(OperatorFunctions(cg))

    _print('1NPC MPO BOND DIMS = ', ''.join(
        ["%6d" % (x.m * x.n) for x in nmpo.left_operator_names]))

    impo = MPO(0)
    impo.load_data(scratch + '/mpo-ident.bin')
    impo.tf = TensorFunctions(OperatorFunctions(cg))

    _print('IDENT MPO BOND DIMS = ', ''.join(
        ["%6d" % (x.m * x.n) for x in impo.left_operator_names]))


def split_mps(iroot, mps, mps_info):
    mps.load_data()  # this will avoid memory sharing
    mps_info.load_mutable()
    mps.load_mutable()

    # break up a MultiMPS to single MPSs
    if len(mps_info.targets) != 1:
        smps_info = MultiMPSInfo(mps_info.n_sites, mps_info.vacuum,
                                 mps_info.targets, mps_info.basis)
        if "full_fci_space" in dic:
            smps_info.set_bond_dimension_full_fci()
        smps_info.tag = mps_info.tag + "-%d" % iroot
        smps_info.bond_dim = mps_info.bond_dim
        for i in range(0, smps_info.n_sites + 1):
            smps_info.left_dims[i] = mps_info.left_dims[i]
            smps_info.right_dims[i] = mps_info.right_dims[i]
        smps_info.save_mutable()
        smps = MultiMPS(smps_info)
        smps.n_sites = mps.n_sites
        smps.center = mps.center
        smps.dot = mps.dot
        smps.canonical_form = '' + mps.canonical_form
        smps.tensors = mps.tensors[:]
        smps.wfns = mps.wfns[iroot:iroot + 1]
        smps.weights = mps.weights[iroot:iroot + 1]
        smps.weights[0] = 1
        smps.nroots = 1
        smps.save_mutable()
    else:
        smps_info = MPSInfo(mps_info.n_sites, mps_info.vacuum,
                            mps_info.targets[0], mps_info.basis)
        if "full_fci_space" in dic:
            smps_info.set_bond_dimension_full_fci()
        smps_info.tag = mps_info.tag + "-%d" % iroot
        smps_info.bond_dim = mps_info.bond_dim
        for i in range(0, smps_info.n_sites + 1):
            smps_info.left_dims[i] = mps_info.left_dims[i]
            smps_info.right_dims[i] = mps_info.right_dims[i]
        smps_info.save_mutable()
        smps = MPS(smps_info)
        smps.n_sites = mps.n_sites
        smps.center = mps.center
        smps.dot = mps.dot
        smps.canonical_form = '' + mps.canonical_form
        smps.tensors = mps.tensors[:]
        if smps.tensors[smps.center] is None:
            smps.tensors[smps.center] = mps.wfns[iroot][0]
        else:
            assert smps.center + 1 < smps.n_sites
            assert smps.tensors[smps.center + 1] is None
            smps.tensors[smps.center + 1] = mps.wfns[iroot][0]
        smps.save_mutable()

    smps.dot = dot
    forward = smps.center == 0
    if smps.canonical_form[smps.center] == 'L' and smps.center != smps.n_sites - smps.dot:
        smps.center += 1
        forward = True
    elif (smps.canonical_form[smps.center] == 'C' or smps.canonical_form[smps.center] == 'M') and smps.center != 0:
        smps.center -= 1
        forward = False
    if smps.canonical_form[smps.center] == 'M' and not isinstance(smps, MultiMPS):
        smps.canonical_form = smps.canonical_form[:smps.center] + \
            'C' + smps.canonical_form[smps.center + 1:]
    if smps.canonical_form[-1] == 'M' and not isinstance(smps, MultiMPS):
        smps.canonical_form = smps.canonical_form[:-1] + 'C'
    if dot == 1:
        if smps.canonical_form[0] == 'C' and smps.canonical_form[1] == 'R':
            smps.canonical_form = 'K' + smps.canonical_form[1:]
        elif smps.canonical_form[-1] == 'C' and smps.canonical_form[-2] == 'L':
            smps.canonical_form = smps.canonical_form[:-1] + 'S'
            smps.center = smps.n_sites - 1
        if smps.canonical_form[0] == 'M' and smps.canonical_form[1] == 'R':
            smps.canonical_form = 'J' + smps.canonical_form[1:]
        elif smps.canonical_form[-1] == 'M' and smps.canonical_form[-2] == 'L':
            smps.canonical_form = smps.canonical_form[:-1] + 'T'
            smps.center = smps.n_sites - 1

    mps.deallocate()
    mps_info.deallocate_mutable()
    smps.save_data()
    return smps, smps_info, forward


def get_mps_from_tags(iroot):
    if iroot >= 0:
        _print('----- root = %3d tag = %s -----' % (iroot, mps_tags[iroot]))
        tag = mps_tags[iroot]
    else:
        _print('----- cps tag = %s -----' % read_tags[0])
        tag = read_tags[0]
    smps_info = MPSInfo(0)
    smps_info.load_data(scratch + "/%s-mps_info.bin" % tag)
    if MPI is not None:
        MPI.barrier()
    smps = MPS(smps_info).deep_copy(smps_info.tag + "-%d" % iroot)
    if MPI is not None:
        MPI.barrier()
    smps_info = smps.info
    smps_info.load_mutable()
    max_bdim = max([x.n_states_total for x in smps_info.left_dims])
    if smps_info.bond_dim < max_bdim:
        smps_info.bond_dim = max_bdim
    max_bdim = max([x.n_states_total for x in smps_info.right_dims])
    if smps_info.bond_dim < max_bdim:
        smps_info.bond_dim = max_bdim
    smps.load_data()
    if smps.dot == 1 and dot == 2:
        if smps.center == 0 and smps.canonical_form[0] == 'S':
            cg = CG(200)
            cg.initialize()
            smps.move_right(cg, prule)
            smps.center = 0
        elif smps.center == smps.n_sites - 1 and smps.canonical_form[smps.center] == 'K':
            cg = CG(200)
            cg.initialize()
            smps.move_left(cg, prule)
            smps.center = smps.n_sites - 2
        smps.dot = dot
        if MPI is not None:
            MPI.barrier()
        smps.save_data()
        if MPI is not None:
            MPI.barrier()
    if smps.center != 0:
        _print('change canonical form ...')
        cf = str(smps.canonical_form)
        ime = MovingEnvironment(impo, smps, smps, "IEX")
        ime.delayed_contraction = OpNamesSet.normal_ops()
        ime.cached_contraction = True
        ime.init_environments(False)
        expect = Expect(ime, smps.info.bond_dim, smps.info.bond_dim)
        expect.iprint = max(min(outputlevel, 3), 0)
        expect.solve(True, smps.center == 0)
        if MPI is not None:
            MPI.barrier()
        smps.save_data()
        if MPI is not None:
            MPI.barrier()
        _print(cf + ' -> ' + smps.canonical_form)
    forward = smps.center == 0
    return smps, smps.info, forward


def get_state_specific_mps(iroot, mps_info):
    smps_info = MPSInfo(0)
    smps_info.load_data(scratch + "/mps_info-ss-%d.bin" % iroot)
    if MPI is not None:
        MPI.barrier()
    smps = MPS(smps_info).deep_copy(mps_info.tag + "-%d" % iroot)
    if MPI is not None:
        MPI.barrier()
    smps_info = smps.info
    smps_info.load_mutable()
    max_bdim = max([x.n_states_total for x in smps_info.left_dims])
    if smps_info.bond_dim < max_bdim:
        smps_info.bond_dim = max_bdim
    max_bdim = max([x.n_states_total for x in smps_info.right_dims])
    if smps_info.bond_dim < max_bdim:
        smps_info.bond_dim = max_bdim
    smps.load_data()
    if smps.dot == 1 and dot == 2:
        if smps.center == 0 and smps.canonical_form[0] == 'S':
            cg = CG(200)
            cg.initialize()
            smps.move_right(cg, prule)
            smps.center = 0
        elif smps.center == smps.n_sites - 1 and smps.canonical_form[smps.center] == 'K':
            cg = CG(200)
            cg.initialize()
            smps.move_left(cg, prule)
            smps.center = smps.n_sites - 2
        smps.dot = dot
        if MPI is not None:
            MPI.barrier()
        smps.save_data()
        if MPI is not None:
            MPI.barrier()
    forward = smps.center == 0
    return smps, smps_info, forward


if not pre_run:
    if MPI is not None:
        if one_body_only:
            mpo = ParallelMPO(mpo, prule_one_body)
        else:
            mpo = ParallelMPO(mpo, prule)
        pmpo = ParallelMPO(pmpo, prule_pdm1)
        if has_2pdm:
            p2mpo = ParallelMPO(p2mpo, prule_pdm2)
        nmpo = ParallelMPO(nmpo, prule_pdm1)
        impo = ParallelMPO(impo, prule_ident)

    _print("para mpo finished", time.perf_counter() - tx)

    if mps is not None:
        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()

    if conn_centers is not None:
        mps.conn_centers = VectorInt(conn_centers)

    # state-specific DMRG
    if "statespecific" in dic and "restart_onepdm" not in dic \
            and "restart_correlation" not in dic and "restart_tran_twopdm" not in dic \
            and "restart_oh" not in dic and "restart_twopdm" not in dic \
            and "restart_tran_onepdm" not in dic and "restart_tran_oh" not in dic:
        assert isinstance(mps, MultiMPS)
        assert nroots != 1

        ext_mpss = []
        for iroot in range(nroots):
            tx = time.perf_counter()
            _print('----- root = %3d / %3d -----' % (iroot, nroots))
            ext_mpss.append(mps.extract(iroot, mps.info.tag + "-%d" % iroot)
                               .make_single(mps.info.tag + "-S%d" % iroot))
            for iex, ext_mps in enumerate(ext_mpss):
                _print(iex, ext_mpss[iex].canonical_form, ext_mpss[iex].center)
                if ext_mps.dot == 1 and dot == 2:
                    if ext_mps.center == 0 and ext_mps.canonical_form[0] == 'S':
                        cg = CG(200)
                        cg.initialize()
                        ext_mps.move_right(cg, prule)
                        ext_mps.center = 0
                    elif ext_mps.center == ext_mps.n_sites - 1 and ext_mps.canonical_form[ext_mps.center] == 'K':
                        cg = CG(200)
                        cg.initialize()
                        ext_mps.move_left(cg, prule)
                        ext_mps.center = ext_mps.n_sites - 2
                    ext_mps.dot = dot
                    ext_mps.save_data()
                _print(iex, ext_mpss[iex].canonical_form, ext_mpss[iex].center)
            if ext_mpss[0].center != ext_mpss[iroot].center:
                _print('change canonical form ...')
                cf = str(ext_mpss[iroot].canonical_form)
                ime = MovingEnvironment(
                    impo, ext_mpss[iroot], ext_mpss[iroot], "IEX")
                ime.delayed_contraction = OpNamesSet.normal_ops()
                ime.cached_contraction = True
                ime.init_environments(False)
                expect = Expect(
                    ime, ext_mpss[iroot].info.bond_dim, ext_mpss[iroot].info.bond_dim)
                expect.iprint = max(min(outputlevel, 3), 0)
                expect.solve(True, ext_mpss[iroot].center == 0)
                ext_mpss[iroot].save_data()
                _print(cf + ' -> ' + ext_mpss[iroot].canonical_form)

            me = MovingEnvironment(
                mpo, ext_mpss[iroot], ext_mpss[iroot], "DMRG")
            me.delayed_contraction = OpNamesSet.normal_ops()
            me.cached_contraction = True
            me.save_partition_info = True
            me.init_environments(outputlevel >= 2)

            _print("env init finished", time.perf_counter() - tx)

            dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
            dmrg.ext_mpss = VectorMPS(ext_mpss[:iroot])
            dmrg.state_specific = True
            dmrg.iprint = max(min(outputlevel, 3), 0)
            for ext_mps in dmrg.ext_mpss:
                ext_me = MovingEnvironment(
                    impo, ext_mpss[iroot], ext_mps, "EX" + ext_mps.info.tag)
                ext_me.delayed_contraction = OpNamesSet.normal_ops()
                ext_me.init_environments(outputlevel >= 2)
                dmrg.ext_mes.append(ext_me)
            if "lowmem_noise" in dic:
                dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
            else:
                dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
            dmrg.cutoff = float(dic.get("cutoff", 1E-14))
            dmrg.decomp_type = decomp_type
            dmrg.trunc_type = trunc_type
            dmrg.davidson_conv_thrds = VectorDouble(dav_thrds)

            sweep_energies = []
            discarded_weights = []
            if "twodot_to_onedot" not in dic:
                E_dmrg = dmrg.solve(len(bond_dims), forward, sweep_tol)
            else:
                tto = int(dic["twodot_to_onedot"])
                assert len(bond_dims) > tto
                dmrg.solve(tto, forward, 0)
                # save the twodot part energies and discarded weights
                sweep_energies.append(np.array(dmrg.energies))
                discarded_weights.append(np.array(dmrg.discarded_weights))
                dmrg.me.dot = 1
                for ext_me in dmrg.ext_mes:
                    ext_me.dot = 1
                dmrg.bond_dims = VectorUBond(bond_dims[tto:])
                dmrg.noises = VectorDouble(noises[tto:])
                dmrg.davidson_conv_thrds = VectorDouble(dav_thrds[tto:])
                E_dmrg = dmrg.solve(len(bond_dims) - tto,
                                    ext_mpss[iroot].center == 0, sweep_tol)
                ext_mpss[iroot].dot = 1
                if MPI is None or MPI.rank == 0:
                    ext_mpss[iroot].save_data()

            if conn_centers is not None:
                me.finalize_environments()

            sweep_energies.append(np.array(dmrg.energies))
            discarded_weights.append(np.array(dmrg.discarded_weights))
            sweep_energies = np.vstack(sweep_energies)
            discarded_weights = np.hstack(discarded_weights)

            if MPI is None or MPI.rank == 0:
                np.save(scratch + "/E_dmrg-%d.npy" % iroot, E_dmrg)
                np.save(scratch + "/bond_dims-%d.npy" %
                        iroot, bond_dims[:len(discarded_weights)])
                np.save(scratch + "/sweep_energies-%d.npy" %
                        iroot, sweep_energies)
                np.save(scratch + "/discarded_weights-%d.npy" %
                        iroot, discarded_weights)
            _print("DMRG Energy for root %4d = %20.15f" % (iroot, E_dmrg))

            if MPI is None or MPI.rank == 0:
                ext_mpss[iroot].info.save_data(
                    scratch + '/mps_info-ss-%d.bin' % iroot)
                ext_mpss[iroot].info.save_data(
                    scratch + '/%s-mps_info-ss-%d.bin' % (mps_tags[0], iroot))

    # GS DMRG
    if "restart_onepdm" not in dic and "restart_twopdm" not in dic \
            and "restart_correlation" not in dic and "restart_tran_twopdm" not in dic \
            and "restart_oh" not in dic and "statespecific" not in dic \
            and "restart_tran_onepdm" not in dic and "restart_tran_oh" not in dic \
            and "delta_t" not in dic and "compression" not in dic:
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(outputlevel >= 2)

        if conn_centers is not None:
            forward = mps.center == 0

        _print("env init finished", time.perf_counter() - tx)

        dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
        dmrg.iprint = max(min(outputlevel, 3), 0)
        if "lowmem_noise" in dic:
            dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
        else:
            dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected
        dmrg.cutoff = float(dic.get("cutoff", 1E-14))
        dmrg.decomp_type = decomp_type
        dmrg.trunc_type = trunc_type
        dmrg.davidson_conv_thrds = VectorDouble(dav_thrds)
        sweep_energies = []
        discarded_weights = []
        if "twodot_to_onedot" not in dic:
            E_dmrg = dmrg.solve(len(bond_dims), forward, sweep_tol)
        else:
            tto = int(dic["twodot_to_onedot"])
            assert len(bond_dims) > tto
            dmrg.solve(tto, forward, 0)
            # save the twodot part energies and discarded weights
            sweep_energies.append(np.array(dmrg.energies))
            discarded_weights.append(np.array(dmrg.discarded_weights))
            dmrg.me.dot = 1
            dmrg.bond_dims = VectorUBond(bond_dims[tto:])
            dmrg.noises = VectorDouble(noises[tto:])
            dmrg.davidson_conv_thrds = VectorDouble(dav_thrds[tto:])
            E_dmrg = dmrg.solve(len(bond_dims) - tto,
                                mps.center == 0, sweep_tol)
            mps.dot = 1
            if MPI is None or MPI.rank == 0:
                mps.save_data()

        if conn_centers is not None:
            me.finalize_environments()

        sweep_energies.append(np.array(dmrg.energies))
        discarded_weights.append(np.array(dmrg.discarded_weights))
        sweep_energies = np.vstack(sweep_energies)
        discarded_weights = np.hstack(discarded_weights)

        if MPI is None or MPI.rank == 0:
            np.save(scratch + "/E_dmrg.npy", E_dmrg)
            np.save(scratch + "/bond_dims.npy",
                    bond_dims[:len(discarded_weights)])
            np.save(scratch + "/sweep_energies.npy", sweep_energies)
            np.save(scratch + "/discarded_weights.npy", discarded_weights)
        _print("DMRG Energy = %20.15f" % E_dmrg)

        if MPI is None or MPI.rank == 0:
            mps_info.save_data(scratch + '/mps_info.bin')
            mps_info.save_data(scratch + '/%s-mps_info.bin' % mps_tags[0])

    # Compression
    if "compression" in dic:
        lmps, lmps_info, _ = get_mps_from_tags(-1)
        mps = lmps.deep_copy(mps_tags[0])
        mps_info = mps.info
        me = MovingEnvironment(impo if overlap else mpo, mps, lmps, "CPS")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(outputlevel >= 2)

        cps = Linear(me, VectorUBond(bond_dims), VectorUBond([lmps.info.bond_dim]))
        cps.iprint = max(min(outputlevel, 3), 0)
        cps.cutoff = float(dic.get("cutoff", 1E-14))
        cps.decomp_type = decomp_type
        cps.trunc_type = trunc_type
        norm = cps.solve(len(bond_dims), mps.center == 0, sweep_tol)

        if MPI is None or MPI.rank == 0:
            np.save(scratch + "/cps_norm.npy", norm)
            mps_info.save_data(scratch + '/mps_info.bin')
            mps_info.save_data(scratch + '/%s-mps_info.bin' % mps_tags[0])

    # Time Evolution (no complex number)
    if "delta_t" in dic:
        dt = float(dic["delta_t"])
        tt = float(dic["target_t"])
        n_steps = int(abs(tt) / abs(dt) + 0.1)
        assert np.abs(abs(n_steps * dt) - abs(tt)) < 1E-10
        _print("Time Evolution NSTEPS = %d" % n_steps)
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(outputlevel >= 2)

        te = TimeEvolution(me, VectorUBond(bond_dims), te_type)
        te.hermitian = not anti_herm
        te.iprint = max(min(outputlevel, 3), 0)
        te.n_sub_sweeps = 1 if te.mode == TETypes.TangentSpace else 2
        te.normalize_mps = False
        te_times = []
        te_energies = []
        te_normsqs = []
        te_discarded_weights = []
        for i in range(n_steps):
            if te.mode == TETypes.TangentSpace:
                te.solve(2, dt / 2, mps.center == 0)
            else:
                te.solve(1, dt, mps.center == 0)
            _print("T = %10.5f <E> = %20.15f <Norm^2> = %20.15f" %
                   ((i + 1) * dt, te.energies[-1], te.normsqs[-1]))
            te_times.append((i + 1) * dt)
            te_energies.append(te.energies[-1])
            te_normsqs.append(te.normsqs[-1])
            te_discarded_weights.append(te.discarded_weights[-1])
        _print("Max Discarded Weight = %9.5g" % max(te_discarded_weights))

        np.save(scratch + "/te_times.npy", np.array(te_times))
        np.save(scratch + "/te_energies.npy", np.array(te_energies))
        np.save(scratch + "/te_normsqs.npy", np.array(te_normsqs))
        np.save(scratch + "/te_discarded_weights.npy",
                np.array(te_discarded_weights))

        if MPI is None or MPI.rank == 0:
            mps_info.save_data(scratch + '/mps_info.bin')
            mps_info.save_data(scratch + '/%s-mps_info.bin' % mps_tags[0])

    def do_onepdm(bmps, kmps):
        me = MovingEnvironment(pmpo, bmps, kmps, "1PDM")
        # currently delayed_contraction is not compatible to
        # ExpectationAlgorithmTypes.Fast
        # me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(outputlevel >= 2)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, bmps.info.bond_dim, kmps.info.bond_dim)
        expect.iprint = max(min(outputlevel, 3), 0)
        expect.solve(True, kmps.center == 0)

        if MPI is None or MPI.rank == 0:
            dmr = expect.get_1pdm(me.n_sites)
            dm = np.array(dmr).copy()
            dmr.deallocate()
            dm = dm.reshape((me.n_sites, 2, me.n_sites, 2))
            dm = np.transpose(dm, (0, 2, 1, 3))
            dm = np.concatenate(
                [dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
            if orb_idx is not None:
                rev_idx = np.argsort(orb_idx)
                dm[:, :, :] = dm[:, rev_idx, :][:, :, rev_idx]
            return dm
        else:
            return None

    # ONEPDM
    if "restart_onepdm" in dic or "onepdm" in dic:

        if nroots == 1:
            dm = do_onepdm(mps, mps)
            if MPI is None or MPI.rank == 0:
                _print("DMRG OCC = ", "".join(
                    ["%6.3f" % x for x in np.diag(dm[0]) + np.diag(dm[1])]))
                np.save(scratch + "/1pdm.npy", dm)
                mps_info.save_data(scratch + '/mps_info.bin')
                mps_info.save_data(scratch + '/%s-mps_info.bin' % mps_tags[0])

                # natural orbital generation
                if "nat_orbs" in dic:
                    spdm = np.sum(dm, axis=0)
                    # need pdm after orbital rotation
                    if orb_idx is not None:
                        spdm[:, :] = spdm[orb_idx, :][:, orb_idx]
                    xpdm = spdm.copy()
                    _print("REORDERED OCC = ", "".join(
                        ["%6.3f" % x for x in np.diag(spdm)]))
                    spdm = spdm.flatten()
                    nat_occs = np.zeros((n_sites, ))
                    if orb_sym is None:
                        raise ValueError(
                            "Need FCIDUMP construction (namely, not a pre run) for 'nat_orbs'!")
                    MatrixFunctions.block_eigs(spdm, nat_occs, orb_sym)
                    _print("NAT OCC = ", "".join(
                        ["%9.6f" % x for x in nat_occs]))
                    # (old, new)
                    rot = np.array(spdm.reshape(
                        (n_sites, n_sites)).T, copy=True)
                    np.save(scratch + "/nat_orb_sym.npy", np.array(orb_sym))
                    for isym in set(orb_sym):
                        mask = np.array(orb_sym) == isym
                        if "nat_km_reorder" in dic:
                            kmidx = np.argsort(KuhnMunkres(1 - rot[mask, :][:, mask] ** 2).solve()[1])
                            _print("init = ", np.sum(np.diag(rot[mask, :][:, mask]) ** 2))
                            rot[:, mask] = rot[:, mask][:, kmidx]
                            nat_occs[mask] = nat_occs[mask][kmidx]
                            _print("final = ", np.sum(np.diag(rot[mask, :][:, mask]) ** 2))
                        if "nat_positive_def" in dic:
                            for j in range(len(nat_occs[mask])):
                                mrot = rot[mask, :][:j + 1, :][:, mask][:, :j + 1]
                                mrot_det = np.linalg.det(mrot)
                                _print("ISYM = %d J = %d MDET = %15.10f" % (isym, j, mrot_det))
                                if mrot_det < 0:
                                    mask0 = np.arange(len(mask), dtype=int)[mask][j]
                                    rot[:, mask0] = -rot[:, mask0]
                        else:
                            mrot = rot[mask, :][:, mask]
                            mrot_det = np.linalg.det(mrot)
                            _print("ISYM = %d MDET = %15.10f" % (isym, mrot_det))
                            if mrot_det < 0:
                                mask0 = np.arange(len(mask), dtype=int)[mask][0]
                                rot[:, mask0] = -rot[:, mask0]
                    if "nat_km_reorder" in dic:
                        _print("REORDERED NAT OCC = ", "".join(["%9.6f" % x for x in nat_occs]))
                    assert np.linalg.norm(rot @ np.diag(nat_occs) @ rot.T - xpdm) < 1E-10
                    np.save(scratch + "/nat_occs.npy", nat_occs)
                    rot_det = np.linalg.det(rot)
                    _print("DET = %15.10f" % rot_det)
                    assert rot_det > 0
                    np.save(scratch + "/nat_rotation.npy", rot)

                    def my_logm(mrot):
                        rs = mrot + mrot.T
                        rl, rv = np.linalg.eigh(rs)
                        assert np.linalg.norm(
                            rs - rv @ np.diag(rl) @ rv.T) < 1E-10
                        rd = rv.T @ mrot @ rv
                        ra, rdet = 1, rd[0, 0]
                        for i in range(1, len(rd)):
                            ra, rdet = rdet, rd[i, i] * rdet - \
                                rd[i - 1, i] * rd[i, i - 1] * ra
                        assert rdet > 0
                        ld = np.zeros_like(rd)
                        for i in range(0, len(rd) // 2 * 2, 2):
                            xcos = (rd[i, i] + rd[i + 1, i + 1]) / 2
                            xsin = (rd[i, i + 1] - rd[i + 1, i]) / 2
                            theta = np.arctan2(xsin, xcos)
                            ld[i, i + 1] = theta
                            ld[i + 1, i] = -theta
                        return rv @ ld @ rv.T

                    import scipy.linalg
                    # kappa = scipy.linalg.logm(rot)
                    kappa = np.zeros_like(rot)
                    for isym in set(orb_sym):
                        mask = np.array(orb_sym) == isym
                        mrot = rot[mask, :][:, mask]
                        mkappa = my_logm(mrot)
                        # mkappa = scipy.linalg.logm(mrot)
                        # assert mkappa.dtype == float
                        gkappa = np.zeros((kappa.shape[0], mkappa.shape[1]))
                        gkappa[mask, :] = mkappa
                        kappa[:, mask] = gkappa
                    assert np.linalg.norm(
                        scipy.linalg.expm(kappa) - rot) < 1E-10
                    assert np.linalg.norm(kappa + kappa.T) < 1E-10

                    # rot is (old, new) => kappa should be minus
                    np.save(scratch + "/nat_kappa.npy", kappa)

                    # integral rotation
                    nat_fname = dic["nat_orbs"].strip()
                    if len(nat_fname) > 0:
                        if fcidump is None:
                            raise ValueError(
                                "Need FCIDUMP construction (namely, not a pre run) for 'nat_orbs'!")
                        # the following code will not check values inside fcidump
                        # since all MPOs are already constructed
                        _print("rotating integrals to natural orbitals ...")
                        # (old, new)
                        fcidump.rotate(VectorDouble(rot.flatten()))
                        _print("finished.")
                        if "symmetrize_ints" in dic:
                            rot_sym_error = fcidump.symmetrize(orb_sym)
                            _print("rotated integral sym error = %12.4g" %
                                   rot_sym_error)
                        _print("writing natural orbital integrals ...")
                        fcidump.write(nat_fname)
                        _print("finished.")

        else:
            for iroot in range(nroots):
                _print('----- root = %3d / %3d -----' % (iroot, nroots))
                if len(mps_tags) > 1:
                    smps, smps_info, forward = get_mps_from_tags(iroot)
                elif "statespecific" in dic:
                    smps, smps_info, forward = get_state_specific_mps(
                        iroot, mps_info)
                else:
                    smps, smps_info, forward = split_mps(iroot, mps, mps_info)
                dm = do_onepdm(smps, smps)
                if MPI is None or MPI.rank == 0:
                    _print("DMRG OCC (state %4d) = " % iroot, "".join(
                        ["%6.3f" % x for x in np.diag(dm[0]) + np.diag(dm[1])]))
                    np.save(scratch + "/1pdm-%d-%d.npy" % (iroot, iroot), dm)
                    smps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)

    # Transition ONEPDM
    # note that there can be a undetermined +1/-1 factor due to the relative phase in two MPSs
    if "restart_tran_onepdm" in dic or "tran_onepdm" in dic:

        assert nroots != 1
        for iroot in range(nroots):
            for jroot in range(nroots):
                _print('----- root = %3d -> %3d / %3d -----' %
                       (jroot, iroot, nroots))
                if len(mps_tags) > 1:
                    simps, simps_info, _ = get_mps_from_tags(iroot)
                    sjmps, sjmps_info, _ = get_mps_from_tags(jroot)
                elif "statespecific" in dic:
                    simps, simps_info, _ = get_state_specific_mps(
                        iroot, mps_info)
                    sjmps, sjmps_info, _ = get_state_specific_mps(
                        jroot, mps_info)
                else:
                    simps, simps_info, _ = split_mps(iroot, mps, mps_info)
                    sjmps, sjmps_info, _ = split_mps(jroot, mps, mps_info)
                dm = do_onepdm(simps, sjmps)
                if SX == SU2:
                    qsbra = simps.info.targets[0].twos
                    # fix different Wigner–Eckart theorem convention
                    dm *= np.sqrt(qsbra + 1)
                dm = dm / np.sqrt(2)
                if MPI is None or MPI.rank == 0:
                    np.save(scratch + "/1pdm-%d-%d.npy" % (iroot, jroot), dm)
            if MPI is None or MPI.rank == 0:
                _print("DMRG OCC (state %4d) = " % iroot, "".join(
                    ["%6.3f" % x for x in np.diag(dm[0]) + np.diag(dm[1])]))
                simps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)

    # Particle Number Correlation
    if "restart_correlation" in dic or "correlation" in dic:
        me = MovingEnvironment(nmpo, mps, mps, "1NPC")
        # me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(outputlevel >= 2)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, mps.info.bond_dim, mps.info.bond_dim)
        expect.iprint = max(min(outputlevel, 3), 0)
        expect.solve(True, mps.center == 0)

        if MPI is None or MPI.rank == 0:
            if SX == SZ:
                dmr = expect.get_1npc(0, me.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()
                dm = dm.reshape((me.n_sites, 2, me.n_sites, 2))
                dm = np.transpose(dm, (0, 2, 1, 3))
                dm = np.concatenate(
                    [dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
                if orb_idx is not None:
                    rev_idx = np.argsort(orb_idx)
                    dm[:, :, :] = dm[:, rev_idx, :][:, :, rev_idx]
            else:
                dmr = expect.get_1npc_spatial(0, me.n_sites)
                dm = np.array(dmr).copy()
                dmr.deallocate()

            np.save(scratch + "/1npc.npy", dm)
            mps_info.save_data(scratch + '/mps_info.bin')
            mps_info.save_data(scratch + '/%s-mps_info.bin' % mps_tags[0])

    def do_twopdm(bmps, kmps):
        me = MovingEnvironment(p2mpo, bmps, kmps, "2PDM")
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(outputlevel >= 2)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, bmps.info.bond_dim, kmps.info.bond_dim)
        expect.iprint = max(min(outputlevel, 3), 0)
        expect.solve(True, kmps.center == 0)

        if MPI is None or MPI.rank == 0:
            dmr = expect.get_2pdm(me.n_sites)
            dm = np.array(dmr, copy=True)
            dm = dm.reshape((me.n_sites, 2, me.n_sites, 2,
                             me.n_sites, 2, me.n_sites, 2))
            dm = np.transpose(dm, (0, 2, 4, 6, 1, 3, 5, 7))
            dm = np.concatenate([dm[None, :, :, :, :, 0, 0, 0, 0], dm[None, :, :, :, :, 0, 1, 1, 0],
                                 dm[None, :, :, :, :, 1, 1, 1, 1]], axis=0)
            if orb_idx is not None:
                rev_idx = np.argsort(orb_idx)
                dm[:, :, :, :, :] = dm[:, rev_idx, :, :, :][:, :, rev_idx,
                                                            :, :][:, :, :, rev_idx, :][:, :, :, :, rev_idx]
            return dm
        else:
            return None

    # TWOPDM
    if "restart_twopdm" in dic or "twopdm" in dic:

        if nroots == 1:
            dm = do_twopdm(mps, mps)
            if MPI is None or MPI.rank == 0:
                np.save(scratch + "/2pdm.npy", dm)
                mps_info.save_data(scratch + '/mps_info.bin')
                mps_info.save_data(scratch + '/%s-mps_info.bin' % mps_tags[0])
        else:
            for iroot in range(nroots):
                _print('----- root = %3d / %3d -----' % (iroot, nroots))
                if len(mps_tags) > 1:
                    smps, smps_info, _ = get_mps_from_tags(iroot)
                elif "statespecific" in dic:
                    smps, smps_info, forward = get_state_specific_mps(
                        iroot, mps_info)
                else:
                    smps, smps_info, forward = split_mps(iroot, mps, mps_info)
                dm = do_twopdm(smps, smps)
                if MPI is None or MPI.rank == 0:
                    np.save(scratch + "/2pdm-%d-%d.npy" % (iroot, iroot), dm)
                    smps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)

    # Transition TWOPDM
    # note that there can be a undetermined +1/-1 factor due to the relative phase in two MPSs
    if "restart_tran_twopdm" in dic or "tran_twopdm" in dic:

        assert nroots != 1
        for iroot in range(nroots):
            for jroot in range(nroots):
                _print('----- root = %3d -> %3d / %3d -----' %
                       (jroot, iroot, nroots))
                if len(mps_tags) > 1:
                    simps, simps_info, _ = get_mps_from_tags(iroot)
                    sjmps, sjmps_info, _ = get_mps_from_tags(jroot)
                elif "statespecific" in dic:
                    simps, simps_info, _ = get_state_specific_mps(
                        iroot, mps_info)
                    sjmps, sjmps_info, _ = get_state_specific_mps(
                        jroot, mps_info)
                else:
                    simps, simps_info, _ = split_mps(iroot, mps, mps_info)
                    sjmps, sjmps_info, _ = split_mps(jroot, mps, mps_info)
                dm = do_twopdm(simps, sjmps)
                if MPI is None or MPI.rank == 0:
                    np.save(scratch + "/2pdm-%d-%d.npy" % (iroot, jroot), dm)
            if MPI is None or MPI.rank == 0:
                simps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)

    def do_oh(bmps, kmps):
        me = MovingEnvironment(impo if overlap else mpo, bmps, kmps, "OH")
        me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.save_partition_info = True
        me.init_environments(outputlevel >= 2)

        _print("env init finished", time.perf_counter() - tx)

        expect = Expect(me, bmps.info.bond_dim, kmps.info.bond_dim)
        expect.iprint = max(min(outputlevel, 3), 0)
        E_oh = expect.solve(False, kmps.center == 0)

        if MPI is None or MPI.rank == 0:
            return E_oh
        else:
            return None

    # OH (Hamiltonian expectation on MPS)
    if "restart_oh" in dic or "oh" in dic:

        if nroots == 1:
            E_oh = do_oh(mps, mps)
            if MPI is None or MPI.rank == 0:
                np.save(scratch + "/E_oh.npy", E_oh)
                print("OH Energy = %20.15f" % E_oh)
                mps_info.save_data(scratch + '/mps_info.bin')
                mps_info.save_data(scratch + '/%s-mps_info.bin' % mps_tags[0])
        else:
            mat_oh = np.zeros((nroots, ))
            for iroot in range(nroots):
                _print('----- root = %3d / %3d -----' % (iroot, nroots))
                if len(mps_tags) > 1:
                    smps, smps_info, _ = get_mps_from_tags(iroot)
                elif "statespecific" in dic:
                    smps, smps_info, forward = get_state_specific_mps(
                        iroot, mps_info)
                else:
                    smps, smps_info, forward = split_mps(iroot, mps, mps_info)
                E_oh = do_oh(smps, smps)
                if MPI is None or MPI.rank == 0:
                    mat_oh[iroot] = E_oh
                    print("OH Energy %4d - %4d = %20.15f" %
                          (iroot, iroot, E_oh))
                    smps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)
            if MPI is None or MPI.rank == 0:
                np.save(scratch + "/E_oh.npy", mat_oh)

    # Transition OH (OH between different MPS roots)
    # note that there can be a undetermined +1/-1 factor due to the relative phase in two MPSs
    # only mat_oh[i, j] with i >= j are filled
    if "restart_tran_oh" in dic or "tran_oh" in dic:

        assert nroots != 1
        mat_oh = np.zeros((nroots, nroots))
        for iroot in range(nroots):
            for jroot in range(iroot + 1):
                _print('----- root = %3d -> %3d / %3d -----' %
                       (jroot, iroot, nroots))
                if len(mps_tags) > 1:
                    simps, simps_info, _ = get_mps_from_tags(iroot)
                    sjmps, sjmps_info, _ = get_mps_from_tags(jroot)
                elif "statespecific" in dic:
                    simps, simps_info, _ = get_state_specific_mps(
                        iroot, mps_info)
                    sjmps, sjmps_info, _ = get_state_specific_mps(
                        jroot, mps_info)
                else:
                    simps, simps_info, _ = split_mps(iroot, mps, mps_info)
                    sjmps, sjmps_info, _ = split_mps(jroot, mps, mps_info)
                E_oh = do_oh(simps, sjmps)
                if MPI is None or MPI.rank == 0:
                    mat_oh[iroot, jroot] = E_oh
                    print("OH Energy %4d - %4d = %20.15f" %
                          (iroot, jroot, E_oh))
            if MPI is None or MPI.rank == 0:
                simps_info.save_data(scratch + '/mps_info-%d.bin' % iroot)
        if MPI is None or MPI.rank == 0:
            np.save(scratch + "/E_oh.npy", mat_oh)

    if mps_info is not None:
        mps_info.deallocate()
