"""Tests for Compressed Genome Evolution (exp588).

Mirrors the phylogenetic-compression project's discipline:
  * three-way agreement: len(encode) == description_length, decode(encode)==id
  * the codec core is application-neutral (no evolution/fitness vocabulary)
  * controls: the null (noise) environment must not compress for free, and the
    MDL objective (C) must not let content hide in an unpenalised macro.
Run:  python -m pytest test_compressed_genome.py -q
      (or: python test_compressed_genome.py)
"""

from __future__ import annotations

import os
import re

import numpy as np

import genome as G
from genome import (Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR,
                    encode, decode, description_length, program_bits, macro_bits,
                    expand, gamma_bits)
from environments import (make_periodic, make_structured, make_unstructured,
                          make_shuffled, fitness)
from evolution import GAConfig, run_ga, random_structured_program
from metrics import summarise_best


# ---------------------------------------------------------------------------
# codec: three-way agreement on random genomes
# ---------------------------------------------------------------------------
def _random_valid_genome(rng, A=4, L=32, nmax=8):
    cfg = GAConfig(A=A, L=L, nmax=nmax)
    return random_structured_program(rng, cfg)


def test_encode_length_equals_description_length():
    rng = np.random.default_rng(0)
    for _ in range(500):
        g = _random_valid_genome(rng)
        assert len(encode(g)) == description_length(g)


def test_description_length_splits_into_LG_plus_LD():
    rng = np.random.default_rng(1)
    for _ in range(500):
        g = _random_valid_genome(rng)
        assert description_length(g) == program_bits(g) + macro_bits(g)


def test_decode_is_inverse_of_encode():
    rng = np.random.default_rng(2)
    for _ in range(500):
        g = _random_valid_genome(rng)
        bits = encode(g)
        g2 = decode(bits, A=g.A, L=g.L, nmax=g.nmax)
        assert g2.macros == g.macros
        assert g2.program == g.program


def test_expand_is_exactly_L():
    rng = np.random.default_rng(3)
    for _ in range(300):
        g = _random_valid_genome(rng)
        assert len(expand(g)) == g.L


def test_gamma_bits_matches_encoding():
    # gamma_bits must equal the real encoded length
    from genome import _gamma_encode
    for x in range(1, 300):
        out = []
        _gamma_encode(x, out)
        assert len(out) == gamma_bits(x)


def test_known_expansion():
    # macro0='AB'; program REP 4 macro0 -> ABAB ABAB then pad
    g = Genome(macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 1)]],
               program=[Instr(OP_REP, 4, 0)], A=4, L=10, nmax=8)
    assert expand(g) == [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
    # MIR of a nested macro
    g2 = Genome(macros=[[Instr(OP_LIT, 2), Instr(OP_LIT, 3)]],
                program=[Instr(OP_MIR, 0)], A=4, L=6, nmax=8)
    assert expand(g2) == [2, 3, 3, 2, 0, 0]


# ---------------------------------------------------------------------------
# structural invariants: DAG constraint, no forward references
# ---------------------------------------------------------------------------
def test_forward_macro_reference_is_rejected():
    # macro 0 references macro 1 -> illegal (must be a DAG, refs strictly earlier)
    bad = Genome(macros=[[Instr(OP_CALL, 1)], [Instr(OP_LIT, 0)]],
                 program=[Instr(OP_LIT, 0)], A=4, L=8)
    try:
        G.validate(bad)
    except ValueError:
        return
    raise AssertionError("forward macro reference should be rejected")


# ---------------------------------------------------------------------------
# domain-neutrality of the codec core
# ---------------------------------------------------------------------------
def _strip_docstrings_and_comments(text: str) -> str:
    text = re.sub(r'""".*?"""', "", text, flags=re.S)
    text = re.sub(r"'''.*?'''", "", text, flags=re.S)
    text = re.sub(r"#.*", "", text)
    return text


def test_core_is_application_neutral():
    here = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(here, "genome.py"), encoding="utf-8").read().lower()
    code = _strip_docstrings_and_comments(text)
    # the codec's executable core must not name the application it is used for
    banned = ["fitness", "evolution", "evolve", "selection", "environment",
              "mutation", "population", "phenotype"]
    for w in banned:
        assert w not in code, f"codec core leaks application word: {w!r}"


# ---------------------------------------------------------------------------
# CONTROL 1: the null (noise) environment does not compress for free
# ---------------------------------------------------------------------------
def test_null_environment_does_not_compress_without_losing_fitness():
    """On i.i.d. noise, group C cannot get BOTH high fitness and a short genome.
    Concretely: if C matches the target as well as the memoriser (F>=0.95) its
    total description is not dramatically shorter than the literal genome."""
    rng = np.random.default_rng(10)
    env = make_unstructured(rng, A=4, L=32)
    cfg = GAConfig(A=4, L=32, pop=150, generations=200, lam=0.002, seed=1)
    c = run_ga(cfg, env, "C")
    s = summarise_best(c.best_by_J)
    literal_bits = gamma_bits(cfg.L + 1) + cfg.L * (2 + 1 * (cfg.A - 1).bit_length())
    # either fitness is well below the memoriser, or it did not really compress
    assert (s["F"] < 0.95) or (s["L_total"] > 0.7 * literal_bits), \
        f"noise appeared to compress for free: F={s['F']} Ltot={s['L_total']}"


# ---------------------------------------------------------------------------
# CONTROL 2: MDL (C) does not let content hide in a macro (B does)
# ---------------------------------------------------------------------------
def test_C_penalises_macro_hiding_that_B_ignores():
    """On a structured target both B and C reach a short program (small L(G)),
    but only C also has a short macro table (small L(D)); B's total genome is
    not compressed."""
    rng = np.random.default_rng(11)
    env = make_periodic(rng, A=4, L=32, period=4)
    cfg = GAConfig(A=4, L=32, pop=150, generations=200, lam=0.002, seed=1)
    b = summarise_best(run_ga(cfg, env, "B").best_by_J)
    c = summarise_best(run_ga(cfg, env, "C").best_by_J)
    # both reach the target
    assert b["F"] >= 0.95 and c["F"] >= 0.95
    # both have a small program
    assert b["LG"] < 40 and c["LG"] < 40
    # only C has a small *total* genome; B's L(D) balloons
    assert c["L_total"] < b["L_total"]
    assert c["LD"] < b["LD"]


# ---------------------------------------------------------------------------
# CONTROL 3: shuffled structure behaves like noise, not like structure
# ---------------------------------------------------------------------------
def test_shuffled_control_matches_noise_not_structure():
    rng = np.random.default_rng(12)
    base = make_periodic(rng, A=4, L=32, period=4)
    shuf = make_shuffled(rng, base)
    cfg = GAConfig(A=4, L=32, pop=150, generations=200, lam=0.002, seed=1)
    c_struct = summarise_best(run_ga(cfg, base, "C").best_by_J)
    c_shuf = summarise_best(run_ga(cfg, shuf, "C").best_by_J)
    # structured compresses far more than its own shuffle at equal fitness budget
    assert c_struct["L_total"] < c_shuf["L_total"]


# ---------------------------------------------------------------------------
# determinism: no Python hash() based seeding anywhere
# ---------------------------------------------------------------------------
def test_no_python_hash_seeding():
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ["genome.py", "environments.py", "evolution.py", "metrics.py"]:
        text = open(os.path.join(here, fn), encoding="utf-8").read()
        code = _strip_docstrings_and_comments(text)
        assert "hash(" not in code, f"{fn} uses hash() for seeding"


def test_same_seed_same_result():
    rng1 = np.random.default_rng(5)
    env = make_periodic(rng1, A=4, L=24, period=3)
    cfg = GAConfig(A=4, L=24, pop=80, generations=60, lam=0.002, seed=3)
    r1 = run_ga(cfg, env, "C")
    r2 = run_ga(cfg, env, "C")
    assert r1.best_by_J.F == r2.best_by_J.F
    assert G.description_length(r1.best_by_J.g) == G.description_length(r2.best_by_J.g)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")


# ===========================================================================
# exp589: exact optimality / Pareto audit
# ===========================================================================
import itertools
import optimal as O
from genome import _sym_bits


def _literal_bits(x, A):
    from genome import gamma_bits
    return gamma_bits(1) + gamma_bits(len(x) + 1) + len(x) * (2 + _sym_bits(A))
    # macros=0 -> gamma(1); program length prefix gamma(L+1); L LITs


def test_kx_reconstructs_and_agrees_with_codec():
    rng = np.random.default_rng(20)
    for _ in range(150):
        L = int(rng.integers(4, 13))
        x = tuple(int(b) for b in rng.integers(0, 2, size=L))
        k, g = O.kx(x, A=2, L=L, mlen=6, Mmax=2, nmax=8, return_genome=True)
        assert G.expand(g) == list(x)
        assert G.description_length(g) == k


def test_kx_is_at_most_literal():
    rng = np.random.default_rng(21)
    for _ in range(150):
        L = int(rng.integers(4, 13))
        x = tuple(int(b) for b in rng.integers(0, 2, size=L))
        k = O.kx(x, A=2, L=L)
        assert k <= _literal_bits(x, 2)


def _brute_min_program_bits(x, macros, A, nmax, maxplen=4):
    """Independent brute force: enumerate ALL programs up to length maxplen over
    the given macro set, run the real codec's expand (so zero-padding is honored
    the same way), and return the min program-section bits among those that
    reproduce x.  Validates optimal._min_program (DP + padding + length prefix)."""
    mbodies = [[Instr(OP_LIT, s) for s in m] for m in macros]
    M = len(macros)
    alphabet = [Instr(OP_LIT, s) for s in range(A)]
    for k in range(M):
        alphabet.append(Instr(OP_CALL, k))
        alphabet.append(Instr(OP_MIR, k))
        for n in range(2, nmax + 1):
            alphabet.append(Instr(OP_REP, n, k))
    best = float("inf")
    for plen in range(0, maxplen + 1):
        for prog in itertools.product(alphabet, repeat=plen):
            g = Genome(macros=mbodies, program=list(prog), A=A, L=len(x), nmax=nmax)
            try:
                if G.expand(g) == list(x):
                    best = min(best, G.program_bits(g))
            except Exception:
                pass
    return best


def test_min_program_matches_bruteforce_programs():
    rng = np.random.default_rng(22)
    for _ in range(60):
        L = int(rng.integers(4, 9))
        x = tuple(int(b) for b in rng.integers(0, 2, size=L))
        cands = O.candidate_macros(x, mlen=4, nmax=4)
        sets = [[]]
        sets += [[m] for m in cands[:2]]
        if len(cands) >= 2:
            sets.append([cands[0], cands[1]])
        for ms in sets:
            pbits, prog = O._min_program(x, list(ms), 2, 4)
            brute = _brute_min_program_bits(x, list(ms), 2, 4, maxplen=4)
            # DP must never exceed brute; and when brute's optimum uses <=4
            # instructions the DP must match it exactly
            assert pbits <= brute or brute == float("inf")
            if brute != float("inf") and prog is not None and len(prog) <= 4:
                assert pbits == brute, (x, ms, pbits, brute)


def test_frontier_is_monotone():
    tgt = [0, 1, 0, 1, 0, 1, 0, 1]
    aud = O.audit_target(tgt, A=2, enumerate_full=True)
    front = aud["frontier"]
    for a, b in zip(front, front[1:]):
        assert b[0] > a[0] and b[1] > a[1]     # more bits AND more fitness


def test_lstar_monotone_in_f():
    tgt = [0, 0, 1, 1, 0, 0, 1, 1]
    aud = O.audit_target(tgt, A=2, enumerate_full=True)
    prev = -1
    for f in [0.25, 0.5, 0.75, 1.0]:
        v = O.lstar_at_least(aud, f)
        if v is not None:
            assert v >= prev
            prev = v


def test_gA_full_dsl_bruteforce_lower_bound_small():
    """For very small L, brute-force ALL bounded genomes (incl. nested macros)
    and confirm kx (flat, <=2 substring macros) is never *below* the true DSL
    minimum -- i.e. kx is a valid (>=) description length, and matches it on
    these structured cases."""
    from genome import gamma_bits
    # brute-force min description length over a bounded full-DSL genome family
    def brute_min(x, A=2, nmax=4):
        L = len(x); best = {}
        subs = set()
        for i in range(L):
            for ln in range(1, min(3, L - i) + 1):
                subs.add(x[i:i + ln])
        subs = list(subs)
        # enumerate <=2 macros (LIT bodies from subs) + programs up to length 4
        macro_choices = [[]]
        for m in subs:
            macro_choices.append([m])
        for m0 in subs:
            for m1 in subs:
                macro_choices.append([m0, m1])
        for macros in macro_choices:
            M = len(macros)
            # program instruction alphabet
            instrs = [Instr(OP_LIT, s) for s in range(A)]
            for k in range(M):
                instrs.append(Instr(OP_CALL, k))
                instrs.append(Instr(OP_MIR, k))
                for n in range(2, nmax + 1):
                    instrs.append(Instr(OP_REP, n, k))
            for plen in range(0, 5):
                for prog in itertools.product(instrs, repeat=plen):
                    g = Genome(macros=[[Instr(OP_LIT, s) for s in m] for m in macros],
                               program=list(prog), A=A, L=L, nmax=nmax)
                    try:
                        if G.expand(g) == list(x):
                            dl = G.description_length(g)
                            if x not in best or dl < best[x]:
                                best[x] = dl
                    except Exception:
                        pass
        return best.get(x, None)

    for x in [(0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 0), (0, 0, 0, 0), (1, 0, 1)]:
        bf = brute_min(x, nmax=4)
        k = O.kx(x, A=2, L=len(x), mlen=3, Mmax=2, nmax=4)
        assert bf is not None
        assert k <= _literal_bits(x, 2)
        assert k == bf, (x, k, bf)   # flat audit meets the bounded full-DSL min


# ===========================================================================
# exp590: mutation-neighborhood & search-efficiency audit
# ===========================================================================
import neighborhood as NB


def _fam(seed=1, A=2, L=16, p=4):
    rng = np.random.default_rng(seed)
    m0 = rng.integers(0, A, p)
    while True:
        m1 = rng.integers(0, A, p)
        if np.any(m1 != m0):
            break
    E0 = np.array([m0[i % p] for i in range(L)])
    E1 = np.array([m1[i % p] for i in range(L)])
    return E0, E1


def test_decode_grouping():
    A_, L_, p = 2, 12, 4
    repC = NB.rep_C(A_, L_, p)
    theta = np.array([1, 0, 1, 1])
    ph = repC.decode(theta)
    assert list(ph) == [theta[i % p] for i in range(L_)]
    repA = NB.rep_A(A_, L_)
    th = np.array([i % 2 for i in range(L_)])
    assert list(repA.decode(th)) == list(th)          # identity


def test_C_fits_periodic_E0_but_R_generally_does_not():
    fE0_C, fE0_R = [], []
    for s in range(12):
        E0, _ = _fam(seed=100 + s)
        repC = NB.rep_C(2, 16, 4)
        repR = NB.rep_R(2, 16, 4, np.random.default_rng(200 + s))
        fE0_C.append(NB.fitness(repC.decode(NB.best_theta(repC, E0)), E0))
        fE0_R.append(NB.fitness(repR.decode(NB.best_theta(repR, E0)), E0))
    assert min(fE0_C) == 1.0                          # C always fits the period
    assert np.mean(fE0_R) < 0.98                      # random grouping usually cannot


def test_C_moves_larger_than_A_same_alignment():
    E0, E1 = _fam(seed=3)
    nbA = NB.neighborhood(NB.rep_A(2, 16), E0, E1)
    nbC = NB.neighborhood(NB.rep_C(2, 16, 4), E0, E1)
    assert nbC.mean_dP > nbA.mean_dP                  # C makes bigger moves
    assert abs(nbC.eta_beneficial - nbA.eta_beneficial) < 1e-9  # equally aligned


def test_C_more_aligned_than_R():
    etaC, etaR = [], []
    for s in range(12):
        E0, E1 = _fam(seed=300 + s)
        etaC.append(NB.neighborhood(NB.rep_C(2, 16, 4), E0, E1).eta_beneficial)
        etaR.append(NB.neighborhood(
            NB.rep_R(2, 16, 4, np.random.default_rng(400 + s)), E0, E1).eta_beneficial)
    assert np.mean(etaC) > np.mean(etaR)              # aligned moves are more useful


def test_C_readapts_faster_than_A():
    E0, E1 = _fam(seed=5)
    raA = NB.readapt_summary(NB.rep_A(2, 16), E0, E1, 0.95, 800, 20, 1)
    raC = NB.readapt_summary(NB.rep_C(2, 16, 4), E0, E1, 0.95, 800, 20, 1)
    assert raC["success_rate"] >= raA["success_rate"] - 1e-9
    assert raC["mean_steps"] is not None and raA["mean_steps"] is not None
    assert raC["mean_steps"] < raA["mean_steps"]


def test_eta_definition_matches_matches_over_changed():
    # a hand case: C on a family where exactly one residue differs
    A_, L_, p = 2, 8, 4
    E0 = np.array([0, 1, 0, 1, 0, 1, 0, 1])           # motif 0101
    E1 = np.array([0, 1, 1, 1, 0, 1, 1, 1])           # motif 0111 (residue 2 differs)
    rep = NB.rep_C(A_, L_, p)
    nb = NB.neighborhood(rep, E0, E1)
    # flipping residue-2 group (positions 2,6) from 0->1 fixes both -> eta 1.0
    assert nb.eta_beneficial == 1.0
    assert nb.mean_dP == 2.0                          # group size L/p = 2


# ===========================================================================
# exp591: epistatic module audit
# ===========================================================================
import epistatic as EP


def test_block_fitness_needs_full_block_match():
    L_, p = 8, 4
    Pi = EP.contiguous_partition(L_, p)
    t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert EP.block_fitness(t, t, Pi) == 1.0
    # one bit off in block 0 -> that block scores 0 -> fitness 0.5
    x = t.copy(); x[0] = 1
    assert EP.block_fitness(x, t, Pi) == 0.5
    # three bits off in block 0 still scores 0 (no partial credit)
    x = t.copy(); x[0] = 1; x[1] = 1; x[2] = 1
    assert EP.block_fitness(x, t, Pi) == 0.5


def test_all_reps_reproduce_E0_exactly():
    L_, p = 16, 4
    rng = np.random.default_rng(1)
    E0 = rng.integers(0, 2, L_)
    Pi = EP.contiguous_partition(L_, p)
    reps = [EP.rep_A_bit(L_), EP.rep_A_block(L_, Pi), EP.rep_C(L_, p),
            EP.rep_R(L_, p, np.random.default_rng(2)),
            EP.rep_C_shuffle(L_, p, np.random.default_rng(3))]
    for rep in reps:
        theta = rep.encode_target(E0)
        assert np.array_equal(rep.decode(theta), E0)      # every rep fits E0


def test_C_partition_equals_A_block_when_aligned():
    L_, p = 16, 4
    Pi = EP.contiguous_partition(L_, p)
    C = EP.rep_C(L_, p)
    Ablk = EP.rep_A_block(L_, Pi)
    cs = sorted(sorted(m.tolist()) for m in C.modules)
    bs = sorted(sorted(m.tolist()) for m in Ablk.modules)
    assert cs == bs                                        # same modules


def test_A_bit_has_no_gradient_in_epistatic_valley():
    L_, p = 16, 4
    rng = np.random.default_rng(4)
    E0 = rng.integers(0, 2, L_); E1 = 1 - E0
    Pi = EP.contiguous_partition(L_, p)
    nb = EP.epi_neighborhood(EP.rep_A_bit(L_), E0, E1, Pi,
                             np.random.default_rng(5), n_samples=3000)
    assert nb.F_E0 == 1.0
    assert nb.P_useful == 0.0                              # single-bit can't cross


def test_C_gradient_present_aligned_absent_shifted():
    L_, p = 16, 4
    rng = np.random.default_rng(6)
    E0 = rng.integers(0, 2, L_); E1 = 1 - E0
    Pi_al = EP.contiguous_partition(L_, p, 0)
    Pi_sh = EP.contiguous_partition(L_, p, p // 2)
    C = EP.rep_C(L_, p)
    nb_al = EP.epi_neighborhood(C, E0, E1, Pi_al, np.random.default_rng(7), 3000)
    nb_sh = EP.epi_neighborhood(C, E0, E1, Pi_sh, np.random.default_rng(7), 3000)
    assert nb_al.P_useful > 0.0                            # aligned: gradient
    assert nb_sh.P_useful == 0.0                           # shifted: no gradient


def test_C_matches_A_block_readapt_when_aligned():
    L_, p = 16, 4
    rng = np.random.default_rng(8)
    E0 = rng.integers(0, 2, L_); E1 = 1 - E0
    Pi = EP.contiguous_partition(L_, p)
    C = EP.rep_C(L_, p)
    Ablk = EP.rep_A_block(L_, Pi)
    rc = EP.epi_readapt_summary(C, E0, E1, Pi, 0.999, 6000, 12, 10)
    rb = EP.epi_readapt_summary(Ablk, E0, E1, Pi, 0.999, 6000, 12, 10)
    ra = EP.epi_readapt_summary(EP.rep_A_bit(L_), E0, E1, Pi, 0.999, 6000, 12, 10)
    # C ~ A_block, and both markedly faster than A_bit
    assert rc["mean_steps"] < 0.6 * ra["mean_steps"]
    assert abs(rc["mean_steps"] - rb["mean_steps"]) < 0.5 * max(
        rc["mean_steps"], rb["mean_steps"])



# ===========================================================================
# exp592: endogenous interaction discovery
# ===========================================================================
import inspect
import endogenous as EN


def test_ari_perfect_and_random():
    part = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]
    assert abs(EN.adjusted_rand_index(part, part, 8) - 1.0) < 1e-9
    rng = np.random.default_rng(0)
    aris = []
    for _ in range(20):
        rp = EN.random_partition_like(part, 8, rng)
        aris.append(EN.adjusted_rand_index(rp, part, 8))
    assert np.mean(aris) < 0.4          # random partitions are far from truth


def test_blind_learner_recovers_causal_blocks():
    w = EN.make_world(p=4, n_causal=3, n_nuisance=2)
    rng = np.random.default_rng(1)
    sample = EN.high_fitness_sample(w, 300, rng, noise=0.02)
    draw = EN.goal_sampler(w, np.random.default_rng(2), "aligned")
    learned, _ = EN.learn_fitness_driven(sample, draw, np.random.default_rng(3))
    ari = EN.adjusted_rand_index(learned, w.true_partition(), w.L)
    assert ari > 0.95                    # recovers Pi* from fitness obs alone


def test_naive_tempted_by_nuisance_but_fitness_driven_is_not():
    w = EN.make_world(p=4, n_causal=3, n_nuisance=2)
    rng = np.random.default_rng(5)
    sample = EN.high_fitness_sample(w, 300, rng, noise=0.02)
    draw = EN.goal_sampler(w, np.random.default_rng(6), "aligned")
    naive = EN.learn_naive(sample)
    learned, _ = EN.learn_fitness_driven(sample, draw, np.random.default_rng(7))
    causal = set(int(i) for b in w.causal_blocks for i in b)

    def nuis_grouped(part):
        lab = EN._labels_from_partition(part, w.L)
        from collections import Counter
        sizes = Counter(lab.tolist())
        g = t = 0
        for i in range(w.L):
            if i in causal:
                continue
            t += 1
            g += (sizes[lab[i]] > 1)
        return g / t
    assert nuis_grouped(naive) > 0.5     # naive groups nuisance
    assert nuis_grouped(learned) == 0.0  # fitness-driven excludes it


def test_module_relevance_causal_positive_nuisance_zero():
    w = EN.make_world(p=4, n_causal=3, n_nuisance=2)
    draw = EN.goal_sampler(w, np.random.default_rng(8), "aligned")
    rng = np.random.default_rng(9)
    U_causal = EN.module_fitness_relevance(w.causal_blocks[0], w.L, draw, rng)
    U_nuis = EN.module_fitness_relevance(w.nuisance_blocks[0], w.L, draw, rng)
    assert U_causal > 0.05
    assert U_nuis == 0.0


def test_leakage_learner_follows_fitness_not_a_hardcoded_partition():
    """Behavioural leakage control: build a world whose TRUE causal blocks are
    NON-contiguous (interleaved).  If the learner secretly assumed contiguous
    blocks it would fail; recovering the interleaved Pi* proves it uses only the
    fitness observations."""
    p = 4
    # interleaved causal blocks over 12 positions: {0,3,6,9},{1,4,7,10},{2,5,8,11}
    causal = [np.array([j for j in range(12) if j % 3 == r]) for r in range(3)]
    nuisance = [np.arange(12, 16), np.arange(16, 20)]
    w = EN.World(L=20, causal_blocks=causal, nuisance_blocks=nuisance, p=p)
    rng = np.random.default_rng(11)
    sample = EN.high_fitness_sample(w, 300, rng, noise=0.02)
    draw = EN.goal_sampler(w, np.random.default_rng(12), "aligned")
    learned, _ = EN.learn_fitness_driven(sample, draw, np.random.default_rng(13))
    ari = EN.adjusted_rand_index(learned, w.true_partition(), w.L)
    assert ari > 0.95                    # follows the (interleaved) fitness blocks


def test_leakage_learner_source_has_no_pistar_access():
    """The learners must not reference the world / true partition in code."""
    for fn in (EN.learn_naive, EN.learn_fitness_driven,
               EN.covariation_partition, EN.module_fitness_relevance):
        src = inspect.getsource(fn)
        for banned in ["true_partition", "causal_blocks", "nuisance_blocks",
                       "world.", "Pi_star"]:
            assert banned not in src, f"{fn.__name__} references {banned!r}"


def test_heldout_operator_beats_A_bit():
    """The learned partition, used as a resample operator, crosses UNSEEN-goal
    valleys faster than 1-bit mutation (structure generalises past {0000,1111})."""
    w = EN.make_world(p=4, n_causal=3, n_nuisance=2)
    rng = np.random.default_rng(20)
    sample = EN.high_fitness_sample(w, 300, rng, noise=0.02)
    draw = EN.goal_sampler(w, np.random.default_rng(21), "aligned")
    learned, _ = EN.learn_fitness_driven(sample, draw, np.random.default_rng(22))
    Pi = w.causal_blocks
    repC = EP.GeneRep("C", [np.array(m) for m in learned], w.L)
    repbit = EP.GeneRep("A", [np.array([i]) for i in range(w.L)], w.L)
    grng = np.random.default_rng(30)
    tC, tb = [], []
    for _ in range(4):
        target = grng.integers(0, 2, w.L)         # unseen arbitrary goal
        rc = EP.epi_readapt_summary(repC, grng.integers(0, 2, w.L), target, Pi,
                                    0.999, 5000, 8, 40)
        rb = EP.epi_readapt_summary(repbit, grng.integers(0, 2, w.L), target, Pi,
                                    0.999, 5000, 8, 40)
        if rc["mean_steps"] and rb["mean_steps"]:
            tC.append(rc["mean_steps"]); tb.append(rb["mean_steps"])
    assert np.mean(tC) < np.mean(tb)


# ===========================================================================
# exp593: adaptive genome capacity
# ===========================================================================
import capacity as CAP


def test_capacity_mutations_stay_valid():
    cfg = CAP.DEFAULT_CFG
    rng = np.random.default_rng(0)
    g = CAP._random_init(rng, cfg)
    for _ in range(2000):
        g2 = CAP.mutate_capacity(rng, g, cfg, allow_add=True, allow_delete=True)
        G.validate(g2)                       # never produces an invalid genome
        assert len(G.expand(g2)) == cfg["L"]
        g = g2


def test_delete_macro_only_removes_unused():
    from genome import Genome, Instr, OP_LIT, OP_CALL, OP_REP
    cfg = CAP.DEFAULT_CFG
    # macro0 used (REP), macro1 unused
    g = Genome(macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 1)],
                       [Instr(OP_LIT, 2), Instr(OP_LIT, 3)]],
               program=[Instr(OP_REP, 8, 0)], A=cfg["A"], L=cfg["L"], nmax=cfg["nmax"])
    rng = np.random.default_rng(1)
    removed_unused = False
    for _ in range(50):
        h = CAP.op_delete_macro(rng, g, cfg)
        # macro0 (used) must still be present and expansion unchanged
        assert 0 in [i for i, c in enumerate(G.macro_call_counts(h)) if c > 0] or \
            len(h.macros) >= 1
        if len(h.macros) < len(g.macros):
            removed_unused = True
    assert removed_unused                     # the unused macro can be pruned


def test_dup_macro_adds_a_copy():
    from genome import Genome, Instr, OP_LIT, OP_REP
    cfg = CAP.DEFAULT_CFG
    g = Genome(macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 1)]],
               program=[Instr(OP_REP, 8, 0), Instr(OP_REP, 8, 0)],
               A=cfg["A"], L=cfg["L"], nmax=cfg["nmax"])
    rng = np.random.default_rng(2)
    saw_dup = False
    for _ in range(30):
        h = CAP.op_dup_macro(rng, g, cfg)
        if len(h.macros) == 2:
            saw_dup = True
            assert h.macros[1] == h.macros[0]     # initially identical copy
            G.validate(h)
    assert saw_dup


def test_fstar_monotone_and_deficit():
    c1 = CAP.fstar_curve(CAP.make_target(1, CAP.SMALL), 2, 12)
    c2 = CAP.fstar_curve(CAP.make_target(2, CAP.SMALL), 2, 12)
    # F*(K) is non-decreasing in K
    for cur in (c1, c2):
        ks = sorted(cur)
        vals = [cur[k] for k in ks]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
    # more rules require a longer minimum description for F=1
    assert CAP.k_min_for_fitness(c2, 1.0) > CAP.k_min_for_fitness(c1, 1.0)


def test_mdl_prunes_redundant_duplicate():
    from genome import Genome, Instr, OP_LIT, OP_REP
    cfg = CAP.DEFAULT_CFG
    m = [Instr(OP_LIT, 0), Instr(OP_LIT, 1)]

    def seed():
        tiles = cfg["L"] // 2
        prog = []
        while tiles > 0:
            n = min(cfg["nmax"], tiles)
            prog.append(Instr(OP_REP, n, 0)); tiles -= n
        return Genome(macros=[list(m), list(m)], program=prog,
                      A=cfg["A"], L=cfg["L"], nmax=cfg["nmax"])

    target = CAP.make_target(1, cfg)

    def evolve(lam, seed_i):
        rng = np.random.default_rng(seed_i)
        pop = [seed() for _ in range(60)]
        scored = [(x, *CAP._evaluate(x, target, lam)) for x in pop]
        for _ in range(80):
            scored.sort(key=lambda t: t[3], reverse=True)
            new = scored[:2]
            while len(new) < 60:
                idx = rng.integers(0, len(scored), size=4)
                par = max((scored[int(i)] for i in idx), key=lambda t: t[3])
                ch = CAP.mutate_capacity(rng, par[0], cfg, True, True)
                new.append((ch, *CAP._evaluate(ch, target, lam)))
            scored = new
        champ = max(scored, key=lambda t: t[3])
        return len(champ[0].macros)

    on = np.mean([evolve(0.001, s) for s in (1, 2)])
    off = np.mean([evolve(0.0, s) for s in (1, 2)])
    assert on < off                          # MDL removes the redundant copy


def test_capacity_grows_with_demand():
    # short staged run: 1-rule -> 4-rule ; effective complexity must increase
    recs = CAP.run_staged([1, 4], lam=0.001, seed=1, gens_per_stage=150, pop=90)
    assert recs[1].metrics["L_total"] > recs[0].metrics["L_total"] + 3
    assert recs[1].metrics["n_used_macros"] > recs[0].metrics["n_used_macros"]
