"""Preparation preview (metadata/stat only): would the full-R command's carried admission pass now?"""
import json, sys, time
sys.path.insert(0, '/data/yschoi/gdna_anchor_refit_v9r6')
import scripts.phase3_selection_matrix as M
import scripts.anchor_refit_stage as RT
snap = '/data/yschoi/gdna_anchor_refit_v9r6/artifacts/anchor_confirmation/ancRsmk9r6_snapshot_aa7476e79f15a159.json'
pin = M._file_pin(snap)
carried = M.load_admission_authority(snap)
refusal = M.anchor_carried_admission_refusal(snap, carried, expected_sha256=pin['sha256'])
specs = M.parse_input_seal_specs([f'{ds}:refit=/data/yschoi/gdna_p3exec_seals/{ds}.refit.input-seal.json'
                                   for ds in ('cifar10', 'flickr25k', 'nuswide', 'mscoco')])
cells = RT.refit_cells(RT.anchor_freeze_authority(), M.anchor_incumbent())
t0 = time.time(); evidence = {}
seals = M.verify_campaign_input_seals(specs, cells, full=M.admission_is_full(carried), expected=carried,
                                      historical=True, evidence=evidence)
print(json.dumps({'snapshot_file_pin': pin, 'carried_keys': sorted(carried), 'carried_refusal': refusal,
                  'full_admission': M.admission_is_full(carried), 'stats_only_seconds': round(time.time() - t0, 2),
                  'cells': len(cells),
                  'seals': {k: {kk: v.get(kk) for kk in ('seal_file_sha256', 'aggregate_sha256', 'mode')}
                            for k, v in seals.items()},
                  'historical_evidence_keys': sorted(evidence)}, default=str, indent=1))

# Bound-mode recipe mapping, as execution re-runs it after input admission (rendered with the admitted refit seals).
report = RT.refit_admission(cells, namespace='ancR9', incumbent=M.anchor_incumbent(), freeze=RT.anchor_freeze_authority(),
                            epochs=None, input_seals=seals)
allowed = set(RT.REFIT_PROTOCOL_FIELDS) | set(RT.REFIT_SEAL_FIELDS)
out = {k: {'recipe_digest': v['digest'], 'differs_from_f': v['differs_from_f'],
           'input_authority_compared': v['input_authority_compared'], 'admitted_overrides': v['overrides'],
           'outside_contract': sorted(set(v['differs_from_f']) - allowed)} for k, v in report.items()}
json.dump(out, open('/tmp/claude-1003/-home-yschoi-GroundedDNA/dad53253-2628-45a2-849e-bb4f82a11b67/scratchpad/v9r6/r_run_render/bound_recipe_mapping.json', 'w'), indent=1)
print('BOUND', len(out), 'cells; any outside contract:', any(v['outside_contract'] for v in out.values()),
      '; distinct differing-field lists:', sorted({tuple(v['differs_from_f']) for v in out.values()}, key=len))
