"""Reconstruct catalogue objects and conservative split units; no split assignment."""
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
E = ROOT / 'rebuild/dataset_v2e_author_quality_freeze'
C = ROOT / 'rebuild/dataset_v2c_acquisition_qc'
A = ROOT / 'rebuild/dataset_v2a_catalog_manifest'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name, data):
    with (OUT / name).open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


class Union:
    def __init__(self, ids):
        self.parent = {cid: cid for cid in ids}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)
    def groups(self):
        groups = collections.defaultdict(list)
        for cid in self.parent:
            groups[self.find(cid)].append(cid)
        return sorted([sorted(v) for v in groups.values()])


def normalize(identifier):
    text = unicodedata.normalize('NFKC', identifier).strip()
    for prefix in ['SIMBAD:', 'HEASARC-GLOBCLUST:']:
        if text.upper().startswith(prefix):
            text = text[len(prefix):]
            break
    # Preserve catalogue signs, decimals, brackets and component suffixes.
    text = text.replace('\u2212', '-').replace('\u2013', '-')
    return re.sub(r'\s+', '', text).casefold()


def main():
    accepted = read(E / 'accepted_inventory.json')
    eligible = accepted['carried_forward_D_deterministic_records'] + accepted['author_accepted_records']
    assert len(eligible) == len({r['candidate_id'] for r in eligible}) == 370
    assert read(E / 'validation.json')['passed']
    by_eligible = {r['candidate_id']: r for r in eligible}
    rejected = {r['candidate_id'] for r in read(E / 'rejected_product_queue.json')['records']}
    deferred = {r['candidate_id'] for r in read(E / 'deferred_review_queue.json')['records']}
    assert len(rejected) == 13 and len(deferred) == 17 and not (set(by_eligible) & (rejected | deferred))
    # A/C used only for identity/ownership/geometry, never as quality authorities.
    sources = {r['candidate_id']: r for r in read(A / 'candidate_manifest.json')['records']}
    geometries = {r['candidate_id']: r for r in read(C / 'candidate_geometry.json')['records']}
    universe = read(E / 'prospective_inventory_effect.json')['records']
    assert len(universe) == len(sources) == 473
    object_union = Union(sources)
    split_union = Union(sources)
    canonical = {cid: normalize(r['canonical_identifier']) for cid, r in sources.items()}
    tokens = collections.defaultdict(set)
    for cid, row in sources.items():
        for name in [row['canonical_identifier'], *row['aliases']]:
            token = normalize(name)
            if token:
                tokens[token].add(cid)
    identity_relations = []
    unresolved_alias_relations = []
    for token, ids in sorted(tokens.items()):
        if len(ids) < 2:
            continue
        members = sorted(ids)
        # Canonical identity shared with a preserved official alias supports identity.
        # Merely sharing a noncanonical field alias is a conservative dependence edge,
        # not certification that two targets are the same physical object.
        confirmed = any(canonical[cid] == token for cid in members)
        relation = {'normalized_token': token, 'candidate_ids': members,
                    'evidence': 'Frozen catalogue main identifier / alias records; no appearance, coordinates-only merge or model outcome',
                    'physical_identity_supported': confirmed}
        for cid in members[1:]:
            split_union.join(members[0], cid)
            if confirmed:
                object_union.join(members[0], cid)
        (identity_relations if confirmed else unresolved_alias_relations).append(relation)
    constraints = read(E / 'frozen_inputs/overlap_grouping_constraints.json')
    assert constraints['nontrivial_components'] == 17
    assert len(constraints['components']) == 454
    for comp in constraints['components']:
        for cid in comp['candidate_ids'][1:]:
            split_union.join(comp['candidate_ids'][0], cid)
    provenance = read(C / 'provenance_ledger.json')['products']
    exact = collections.defaultdict(set)
    for product in provenance:
        exact[product['sha256']].add(product['candidate_id'])
    exact_relations = []
    for digest, ids in sorted(exact.items()):
        if len(ids) > 1:
            members = sorted(ids)
            exact_relations.append({'sha256': digest, 'candidate_ids': members,
                                    'scope': 'Exact complete FITS bytes; conservative split dependence, not a new physical identity claim'})
            for cid in members[1:]:
                split_union.join(members[0], cid)
    for members in object_union.groups():
        for cid in members[1:]:
            split_union.join(members[0], cid)
    physical_groups = []
    physical_by_cid = {}
    identity_class_conflicts = []
    for members in object_union.groups():
        object_id = 'object_' + hashlib.sha256('\n'.join(members).encode()).hexdigest()[:20]
        active = [cid for cid in members if cid in by_eligible]
        classes = sorted({sources[cid]['target_class'] for cid in active})
        if len(classes) > 1:
            identity_class_conflicts.append({'physical_object_id': object_id, 'candidate_ids': active, 'classes': classes})
        physical_groups.append({'physical_object_id': object_id, 'all_source_candidate_ids': members,
                                'eligible_candidate_ids': active, 'canonical_identifiers': [sources[cid]['canonical_identifier'] for cid in members],
                                'eligible_frozen_classes': classes,
                                'identity_basis': 'Unambiguous frozen catalogue main ID and supported aliases; no visual identity inference.',
                                'new_official_identity_certification_performed': False})
        for cid in members:
            physical_by_cid[cid] = object_id
    units = []
    unit_by_cid = {}
    for members in split_union.groups():
        unit_id = 'unit_' + hashlib.sha256('\n'.join(members).encode()).hexdigest()[:20]
        active = [cid for cid in members if cid in by_eligible]
        object_ids = sorted({physical_by_cid[cid] for cid in active})
        class_object_ids = collections.defaultdict(set)
        for cid in active:
            class_object_ids[sources[cid]['target_class']].add(physical_by_cid[cid])
        unit = {'split_unit_id': unit_id, 'full_473_universe_members': members, 'eligible_candidate_ids': active,
                'eligible_physical_object_ids': object_ids,
                'physical_object_counts_by_class': {cl: len(v) for cl, v in sorted(class_object_ids.items())},
                'original_nontrivial_overlap_components': [c['component_id'] for c in constraints['components'] if c['object_count'] > 1 and set(c['candidate_ids']) <= set(members)],
                'must_be_indivisible': True, 'excluded_or_deferred_bridges_retained': True, 'future_split': None}
        units.append(unit)
        for cid in members:
            unit_by_cid[cid] = unit_id
    # Standalone previews are lineage derivatives, never input images. Mixed-object
    # contact sheets and review montages are explicitly forbidden as model inputs.
    standalone_assets = []
    for folder in [ROOT / 'rebuild/dataset_v2b_pilot', C]:
        for cid in sources:
            for suffix in ['blue', 'red', 'nir', 'pseudo_rgb']:
                path = folder / 'previews' / f'{cid}_{suffix}.png'
                if path.exists():
                    standalone_assets.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'size_bytes': path.stat().st_size,
                                              'candidate_id': cid, 'physical_object_id': physical_by_cid[cid],
                                              'split_unit_id': unit_by_cid[cid], 'derivative_of_frozen_candidate_bundle': True,
                                              'parent_eligible_in_E_primary_pool': cid in by_eligible,
                                              'model_input_authorized': False, 'new_independent_observation': False})
    # Review panels are excluded as inputs but still have explicit object lineage.
    with (E / 'frozen_inputs/review_blinding_key.csv').open(newline='') as stream:
        review_keys = {r['review_id']: r['candidate_id'] for r in csv.DictReader(stream)}
    for subfolder in ['blinded_review/assets', 'audit_detail']:
        folder = ROOT / 'rebuild/dataset_v2d_curation_shortcut_gate' / subfolder
        for path in sorted(folder.glob('*.png')):
            review_id = path.name.split('_', 1)[0]
            assert review_id in review_keys
            cid = review_keys[review_id]
            standalone_assets.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'size_bytes': path.stat().st_size,
                                      'candidate_id': cid, 'physical_object_id': physical_by_cid[cid],
                                      'split_unit_id': unit_by_cid[cid], 'derivative_of_frozen_candidate_bundle': True,
                                      'parent_eligible_in_E_primary_pool': cid in by_eligible,
                                      'review_id': review_id, 'ownership_evidence': 'Frozen E snapshot of original D review key plus D asset filename convention',
                                      'model_input_authorized': False, 'new_independent_observation': False})
    candidate_records = []
    for cid, row in sorted(by_eligible.items()):
        geometry = geometries[cid]
        assert geometry['canonical_identifier'] == row['canonical_identifier'] and geometry['target_class'] == row['frozen_class']
        assert row['aliases'] == sources[cid]['aliases']
        assert len(row['products']) == 3 and {p['band'] for p in row['products']} == {'blue', 'red', 'nir'}
        for product in row['products']:
            path = ROOT / product['repository_relative_path']
            assert path.stat().st_size == product['size_bytes'] and sha(path) == product['sha256']
        candidate_records.append({'candidate_id': cid, 'canonical_identifier': row['canonical_identifier'],
                                  'frozen_class': row['frozen_class'], 'frozen_subtype': row['frozen_subtype'],
                                  'quality_authority': 'Dataset-v2E accepted_inventory.json', 'quality_evidence_level': row['quality_evidence_level'],
                                  'physical_object_id': physical_by_cid[cid], 'split_unit_id': unit_by_cid[cid],
                                  'aliases': row['aliases'], 'products': row['products'],
                                  'FOV_arcmin': geometry['pilot_fov_arcmin'], 'catalogue_extent_arcmin': geometry['pilot_extent_arcmin'],
                                  'ra_deg': geometry['ra_deg'], 'dec_deg': geometry['dec_deg'],
                                  'split': None, 'benchmark_activated': False})
    counts = []
    for cl in sorted({r['frozen_class'] for r in eligible}):
        subset = [r for r in candidate_records if r['frozen_class'] == cl]
        object_ids = {r['physical_object_id'] for r in subset}
        counts.append({'frozen_class': cl, 'quality_ready_candidates': len(subset), 'FITS_band_images': 3 * len(subset),
                       'independent_multiband_observations': len(subset), 'distinct_catalogue_supported_physical_objects': len(object_ids),
                       'split_units_touching_class': len({r['split_unit_id'] for r in subset}),
                       'E_explicit_author_accept_quality': sum(r['quality_evidence_level'] == 'E_explicit_author_accept_quality' for r in subset),
                       'E_carried_D_deterministic_quality': sum(r['quality_evidence_level'] == 'D_deterministic_quality_ready_not_author_adjudication' for r in subset)})
    original_membership_check = []
    for comp in constraints['components']:
        if comp['object_count'] > 1:
            target_units = {unit_by_cid[cid] for cid in comp['candidate_ids']}
            assert len(target_units) == 1
            original_membership_check.append({**comp, 'resulting_split_unit_id': next(iter(target_units)),
                                               'eligible_members': [cid for cid in comp['candidate_ids'] if cid in by_eligible],
                                               'all_original_members_preserved_as_constraints': True})
    assert len(candidate_records) == 370 and sum(c['FITS_band_images'] for c in counts) == 1110
    save('object_group_inventory.json', {'created_utc_before_split_protocol': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                         'quality_authority_path': str((E / 'accepted_inventory.json').relative_to(ROOT)),
                                         'quality_authority_sha256': sha(E / 'accepted_inventory.json'),
                                         'quality_authority_inventory_sha256': sha(E / 'file_inventory.json'),
                                         'primary_eligibility_candidates': candidate_records, 'primary_eligibility_count': 370,
                                         'eligible_FITS_band_images': 1110, 'counts_by_class_before_split_proposal': counts,
                                         'physical_object_groups_all_473': physical_groups,
                                         'eligible_distinct_physical_object_groups': sum(bool(g['eligible_candidate_ids']) for g in physical_groups),
                                         'split_units_full_universe': units, 'active_split_units': sum(bool(u['eligible_candidate_ids']) for u in units),
                                         'original17_nontrivial_components': original_membership_check,
                                         'supported_identity_relations': identity_relations,
                                         'unresolved_shared_alias_relations': unresolved_alias_relations,
                                         'physical_identity_class_conflicts': identity_class_conflicts,
                                         'cross_candidate_exact_FITS_relations': exact_relations,
                                         'standalone_derivative_lineage': standalone_assets,
                                         'standalone_derivative_count_all_source_candidates': len(standalone_assets),
                                         'standalone_derivative_count_eligible_parent': sum(r['parent_eligible_in_E_primary_pool'] for r in standalone_assets),
                                         'counts_do_not_treat_bands_or_previews_as_independent_objects': True,
                                         'input_scope': 'Only E quality-ready FITS bundles may later produce tensors. Existing previews/contact sheets/audit panels/author-review assets are never model inputs. Standalone derivatives inherit the parent unit; multi-object review montages are not scientific observations and receive no split.',
                                         'normalization_rule': 'NFKC/case/whitespace normalization and removal of known source prefixes; preserve plus/minus, decimal points, brackets and component suffixes. No sign-stripping, nearest-coordinate or appearance-only identity merge.',
                                         'identity_limitation': 'Catalogue-supported grouping reconstructed from frozen provenance, not a new online object-identity or substructure adjudication. Shared noncanonical aliases are unresolved and conservatively join split units without forcing physical identity.',
                                         'quality_rejections_13_excluded': sorted(rejected), 'quality_deferred17_excluded': sorted(deferred),
                                         'other_E_noneligible_candidates_not_admitted': sorted(set(sources) - set(by_eligible) - rejected - deferred),
                                         'no_split_assigned': True})
    print(json.dumps({'counts': counts, 'distinct_eligible_objects': sum(bool(g['eligible_candidate_ids']) for g in physical_groups),
                      'active_split_units': sum(bool(u['eligible_candidate_ids']) for u in units),
                      'identity_alias_relations': len(identity_relations), 'unresolved_alias_relations': len(unresolved_alias_relations),
                      'identity_class_conflicts': len(identity_class_conflicts), 'derivatives_with_parent_group': len(standalone_assets)}))


def boundary(event, args):
    if event.startswith('socket.') or event == 'urllib.Request':
        raise RuntimeError('Dataset-v2F is offline')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        if write and not path.is_relative_to(OUT):
            raise RuntimeError('Write outside Dataset-v2F')
        if path.suffix.lower() in {'.pt', '.pth', '.h5', '.hdf5', '.safetensors'} or 'baseline_comparison_v1' in path.parts:
            raise RuntimeError('No checkpoint/historical-result access in protocol build')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    sys.addaudithook(boundary)
    main()
