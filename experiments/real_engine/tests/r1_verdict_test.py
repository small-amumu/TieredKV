# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""synthetic test of r1_verdict.py v2 (writes only into a temporary directory given as argv[1]).
Cases: equal arms -> Part 1 PASS (A0 included), P1 / P2 hold, 60 matched; engine NLL +0.02 -> A2 FAIL; engine KL +1.5e-3 -> A3
FAIL; KL +4e-4 -> A3 PASS; vt missing -> FAIL and P1 / P2 not holding; rc != 0 -> FAIL; lengths differ on 2 tasks -> P1 not
evaluable and A0 FAIL; a baseline as good as the engine -> P1 does not hold; a wrong data md5 -> FAIL; all vt items excluded ->
FAIL; the reference 2e-3 KL above X-0139 -> A0 FAIL; h33 rows +0.03 NLL (X-0139 too) -> P2 does not hold.
Prints one line per case and R1_VERDICT_TEST PASS / FAIL."""
import json, os, random, subprocess, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(HERE, "r1_verdict.py")
SRC = open(V).read()
MD5 = [l.split('"')[1] for l in SRC.splitlines() if l.startswith("RP2_MD5")][0]
EMD5 = [l.split('"')[1] for l in SRC.splitlines() if l.startswith("ENG_MD5")][0]
EXP = json.load(open(os.path.join(HERE, "EXPECTED_MD5.json")))
DMD5 = {"niah_single_3": "744ac2c5be94f6e7a9c68605e6a6cfef", "niah_multikey_2": "8fdd6127ce098f53a7b5f3892de734f7",
        "niah_multikey_3": "9661e424b48c1ce753f3c9408a0b0632", "niah_multivalue": "20c23af3ce3ac4f1f35d464e0352255b",
        "niah_multiquery": "5e1ac06d17c776603d791b3fc991588b", "vt": "6f76ebd82ff04bc18864060f6b3d3f11"}
T = ("niah_single_3", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt")
PFX = "stkm8v8128k64_x2s128_p50_"


def arm(kl, nll, ref=False):
    x = dict(status="ok", tf_kl_mean=kl, gold_nll=nll, tf_finite=True, gold_finite=True, tf_steps=50, tf_flags=0, gold_flags=0,
             tf_kl_steps=[kl * 3] * 50)
    if ref:
        x.update(tf_agree_ref=0.995, tf_kl_ref=1e-5, tf_kl_ref_max=1e-3)
    return x


def make(d, dn=0.0, dk=0.0, drop=None, rc_bad=None, len_off_tasks=(), base_kl=0.17, bad_md5=None, excl=None, ref_off=0.0,
         h33_nll=0.0, tfnll_off_tasks=()):
    os.makedirs(d, exist_ok=True)
    rnd = random.Random(1)
    xs = []
    for t in T:
        rows = []
        for i in range(10):
            b = rnd.gauss(0.5, 0.1); tn = rnd.gauss(0.1, 0.02)
            k33 = 0.0156 + rnd.gauss(0, 2e-3); k54 = 0.0153 + rnd.gauss(0, 2e-3)
            r = dict(task=t, i=i, status="ok", tf_status="ok", gold_status="ok", L0=32000 + i, tf_len=51, gold_len=20,
                     fp=dict(gold_nll=b, tf_nll=tn), ref33=arm(k33 + ref_off, b + 0.023 + h33_nll), ref54=arm(k54 + ref_off, b + 0.025),
                     eng33=arm(k33 + ref_off + dk + rnd.gauss(0, 1e-4), b + 0.023 + h33_nll + dn + rnd.gauss(0, 0.003), True),
                     eng54=arm(k54 + ref_off + dk + rnd.gauss(0, 1e-4), b + 0.025 + dn + rnd.gauss(0, 0.003), True))
            if t == excl:
                r["gold_status"] = "gold of 200 tokens outside [2, 30]"
            rows.append(r)
            task = f"ruler:{t}@32768"
            L = 32000 + i + (7 if t in len_off_tasks else 0)
            xs.append(dict(arm="fp", task=task, i=i, len=L, tf_status="ok", gold_status="ok", tf_steps=50, gold_steps=19, gold_nll=b,
                           tf_nll=tn + (0.01 if t in tfnll_off_tasks else 0.0)))
            for a_, kl, dnl in (("kivi128", 0.288, 0.82), ("kvq2n", base_kl, 0.50 if base_kl > 0.1 else 0.024), ("kvq3", 0.0184, 0.071),
                                (PFX + "h33", k33, 0.023 + h33_nll), (PFX + "h54", k54, 0.025)):
                xs.append(dict(arm=a_, task=task, i=i, len=L, tf_status="ok", gold_status="ok", tf_steps=50, gold_steps=19,
                               tf_kl_mean=kl + (rnd.gauss(0, 1e-3) if a_[:3] != PFX[:3] else 0.0), gold_nll=b + dnl))
        if t == drop:
            continue
        with open(os.path.join(d, f"r1_{t}.jsonl"), "w") as f:
            md5 = dict(EXP, **{"engine/rp2_tf.py": MD5, "engine/rp_engine.py": EMD5})
            f.write(json.dumps(dict(meta=True, md5=md5, task=t, data_md5=("0" * 32 if t == bad_md5 else DMD5[t]), items="0:10",
                                    maxgen=30 if t == "vt" else 128, model="/home/user/models/LLaMA-2-7B-32K")) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps(dict(meta=True, end=True, n_errors=0)) + "\n")
    with open(os.path.join(d, "xa.jsonl"), "w") as f:
        for x in xs[: len(xs) // 2]:
            f.write(json.dumps(x) + "\n")
    with open(os.path.join(d, "xb.jsonl"), "w") as f:
        for x in xs[len(xs) // 2:]:
            f.write(json.dumps(x) + "\n")
    for g, ts in (("a", T[:3]), ("b", T[3:])):
        with open(os.path.join(d, f"rc_{g}.log"), "w") as f:
            f.write(f"2026-11-10T00:00:00Z chunk_test_{g} rc=0 CHUNK_TEST PASS\n")
            for t in ts:
                f.write(f"2026-11-10T00:00:00Z r1_{t} rc={1 if t == rc_bad else 0} secs=1 RP2_TF_DONE\n")
    o = subprocess.run([sys.executable, "-I", V, d, "--rc", os.path.join(d, "rc_a.log"), os.path.join(d, "rc_b.log"),
                        "--x0139", os.path.join(d, "xa.jsonl"), os.path.join(d, "xb.jsonl")], capture_output=True, text=True)
    if o.returncode:
        print(o.stderr[-2000:])
    return json.loads(o.stdout)


base = sys.argv[1]
ok = True


def case(name, c):
    global ok
    print(name + ":", c)
    ok &= bool(c)


o = make(os.path.join(base, "eq"))
case("equal -> PASS (A0), P1, P2, 60 matched, judgement inputs hold",
     o["part1"]["PASS"] and all(o["part1"]["widths"][w]["A0"] for w in ("h33", "h54")) and o["part2"]["P1"]["holds"]
     and o["part2"]["P2"]["holds"] and o["part2"]["matching"]["n_matched"] == 60 and o["judgement_inputs"]["all_hold"]
     and abs(o["part2"]["table"]["eng54"]["gold_dnll"] - 0.025) < 0.005)
o = make(os.path.join(base, "nll"), dn=0.02)
case("NLL +0.02 -> A2 FAIL", not o["part1"]["PASS"] and not o["part1"]["widths"]["h33"]["A2"] and not o["judgement_inputs"]["crit_i"])
o = make(os.path.join(base, "kl"), dk=1.5e-3)
case("KL +1.5e-3 -> A3 FAIL", not o["part1"]["PASS"] and not o["part1"]["widths"]["h54"]["A3"])
o = make(os.path.join(base, "kl_small"), dk=4e-4)
case("KL +4e-4 -> A3 PASS (margin 1e-3)", o["part1"]["widths"]["h54"]["A3"] and o["part1"]["PASS"])
o = make(os.path.join(base, "miss"), drop="vt")
case("vt missing -> FAIL, P1 / P2 not holding", not o["part1"]["PASS"] and any("vt" in p for p in o["completeness_problems"])
     and not o["part2"]["P1"]["holds"] and not o["part2"]["P2"]["holds"])
o = make(os.path.join(base, "rc"), rc_bad="vt")
case("rc!=0 -> FAIL", not o["part1"]["PASS"] and any("rc logs" in p for p in o["completeness_problems"]))
o = make(os.path.join(base, "len"), len_off_tasks=T[:2])
case("lengths differ on 2 tasks -> P1 not evaluable, A0 FAIL", o["part2"]["matching"]["n_matched"] == 40
     and not o["part2"]["matching"]["evaluable"] and not o["part2"]["P1"]["holds"] and not o["part1"]["PASS"])
o = make(os.path.join(base, "base"), base_kl=0.0150)
case("baseline as good as engine -> P1 does not hold", o["part1"]["PASS"] and not o["part2"]["P1"]["holds"])
o = make(os.path.join(base, "md5"), bad_md5="vt")
case("wrong data md5 -> FAIL", not o["part1"]["PASS"] and any("data_md5" in p for p in o["completeness_problems"]))
o = make(os.path.join(base, "excl"), excl="vt")
case("all vt excluded -> FAIL", not o["part1"]["PASS"] and any("usable items" in p for p in o["completeness_problems"]))
o = make(os.path.join(base, "a0"), ref_off=2e-3)
case("reference 2e-3 KL above X-0139 -> A0 FAIL", not o["part1"]["widths"]["h54"]["A0"] and not o["part1"]["PASS"]
     and o["part1"]["widths"]["h54"]["A3"])
o = make(os.path.join(base, "p2"), h33_nll=0.03)
case("h33 +0.03 NLL -> P2 does not hold", o["part1"]["PASS"] and not o["part2"]["P2"]["holds"] and not o["judgement_inputs"]["crit_ii"])
o = make(os.path.join(base, "tfnll"), tfnll_off_tasks=T[:1])
case("fp trajectory NLL off on 1 task -> 50 matched, evaluable False, A0 FAIL, judgement (iv) false",
     o["part2"]["matching"]["n_matched"] == 50 and not o["part2"]["matching"]["evaluable"] and not o["judgement_inputs"]["crit_iv_A0"]
     and not o["judgement_inputs"]["all_hold"])
print("R1_VERDICT_TEST", "PASS" if ok else "FAIL")
