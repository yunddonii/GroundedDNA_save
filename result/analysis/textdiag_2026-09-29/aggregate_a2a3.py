"""Aggregate the per-run A1/A2/A3 JSONs into 3-seed means (± sample SD) per dataset x arm."""
import glob, json, os, sys, collections
import numpy as np

AXES = ["C_primary_object", "C_secondary_object", "C_activity_or_relation", "C_color_texture"]


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return f"{np.mean(xs):.3f}" + (f" ± {np.std(xs, ddof=1):.3f}" if len(xs) > 1 else "")


def main(d, out_md):
    groups = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        arm = "notext" if r["text_supervision_disabled"] else ("anchors" if r["axis_center"] == "anchors" else "base")
        groups[(r["dataset"], arm)].append(r)
    lines = ["# A1/A2/A3 on exploratory anchor runs (Gumbel ON; own N), 3 seeds, first 500 validation rows, CPU deployment forward", ""]
    lines += ["## A2 code->own-axis (chance .25) and geometry", "", "| dataset | arm | n | pre-quant token | codeword | eff-rank z (mean over slots) | cos(z,q) | codewords used / K | proto~codeword cos |", "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for (ds, arm), rs in sorted(groups.items()):
        a2 = [r["A2"] for r in rs]
        lines.append(f"| {ds} | {arm} | {len(rs)} | {ms([x['pre_quantisation_slot_token']['code_picks_own_axis'] for x in a2])} | "
                     f"{ms([x['quantised_codeword']['code_picks_own_axis'] for x in a2])} | "
                     f"{ms([np.mean([x['per_slot'][ax]['eff_rank_z'] for ax in AXES]) for x in a2])} | "
                     f"{ms([np.mean([x['per_slot'][ax]['cos_z_q_mean'] for ax in AXES]) for x in a2])} | "
                     f"{np.mean([np.mean([x['per_slot'][ax]['codewords_used'] for ax in AXES]) for x in a2]):.1f} / {a2[0]['per_slot'][AXES[0]]['K']} | "
                     f"{ms([np.mean([x['per_slot'][ax]['prototype_vs_codeword_cos_mean'] for ax in AXES]) for x in a2])} |")
    lines += ["", "## A1 the text_code_kl target on the validation rows (mean over the 4 local slots)", "", "| dataset | arm | mean confidence | share excluded (<= 0.2) | text argmax = visual codeword | text top-1 mass |", "|---|---|---:|---:|---:|---:|"]
    for (ds, arm), rs in sorted(groups.items()):
        a1 = [r["A1_text_code_kl_target"] for r in rs]
        f = lambda key: ms([np.mean([x[ax][key] for ax in AXES]) for x in a1])
        lines.append(f"| {ds} | {arm} | {f('mean_confidence')} | {f('share_below_threshold')} | {f('text_argmax_equals_visual_codeword')} | {f('text_top1_mass_mean')} |")
    for rule_label, rule_key in (("caption word Jaccard >= 0.25", "jaccard>=0.25"), ("caption CLIP cosine top 2 %", "caption_cos_top2pct")):
        lines += ["", f"## A3 cross-image slot consistency, pairing rule: {rule_label}", "",
                  "P(same codeword) lift over all pairs: OWN slot vs the OTHER three slots (same pairs). Base-Hamming delta of the own-slot codon vs all pairs.", "",
                  "| dataset | arm | axis | pairs | own-slot lift | other-slots lift | own − other | Δ base-Hamming (own) |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for (ds, arm), rs in sorted(groups.items()):
            for ax in AXES:
                vals = []
                for r in rs:
                    rules = r["A3"]["rules"][ax]
                    key = next((k for k in rules if k.startswith(rule_key)), None)
                    v = rules.get(key) if key else None
                    vals.append(v)
                vals = [v for v in vals if v]
                if not vals:
                    continue
                own = [v["lift_same_codeword"] for v in vals]; oth = [v["other_slots_lift_mean"] for v in vals]
                lines.append(f"| {ds} | {arm} | {ax.replace('C_', '')} | {np.mean([v['pairs'] for v in vals]):.0f} | {ms(own)} | {ms(oth)} | "
                             f"{ms([o - t for o, t in zip(own, oth)])} | {ms([v['hamming_delta_vs_all_pairs'] for v in vals])} |")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
