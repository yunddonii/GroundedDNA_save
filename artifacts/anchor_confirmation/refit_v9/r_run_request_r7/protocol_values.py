import json, sys
sys.path.insert(0, '/data/yschoi/gdna_anchor_refit_v9r6')
import scripts.phase3_selection_matrix as M
import scripts.anchor_refit_stage as RT
inc = M.anchor_incumbent(); fz = RT.anchor_freeze_authority()
out = {}
for ds in ('cifar10', 'flickr25k', 'nuswide', 'mscoco'):
    n = fz['datasets'][ds]['N']
    out[ds] = {'N': n, 'lambdas': fz['datasets'][ds].get('lambdas'), 'topp': fz['datasets'][ds].get('routing_adaptive_topp'),
               'joint': fz['datasets'][ds].get('lambda_codon_joint'),
               'protocol_fields_seed42': RT.refit_protocol_fields(ds, n, seed=42, arm='anchors', incumbent=inc, epochs=None)}
print(json.dumps(out, indent=1, default=str))
