"""Offline deterministic quality triage and descriptive shortcut analysis; no fitting."""
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
C = ROOT / 'rebuild/dataset_v2c_acquisition_qc'
sys.path.insert(0, str(C))
from fits_qc import load_fits, pix_to_sky, sky_to_pix


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, data):
    with (OUT / name).open('x') as f:
        json.dump(data, f, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def table(name, rows, fields=None):
    with (OUT / name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def stats(values):
    x = np.array([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(x):
        return {'n': 0}
    return {'n': len(x), 'minimum': float(x.min()), 'p05': float(np.quantile(x, .05)),
            'p25': float(np.quantile(x, .25)), 'median': float(np.median(x)),
            'p75': float(np.quantile(x, .75)), 'p95': float(np.quantile(x, .95)),
            'maximum': float(x.max()), 'mean': float(x.mean()),
            'sample_SD': float(x.std(ddof=1)) if len(x) > 1 else None}


def categorical_association(classes, categories):
    cs, xs = sorted(set(classes)), sorted(set(categories))
    count = np.array([[sum(c == a and x == b for c, x in zip(classes, categories)) for b in xs] for a in cs], float)
    n = count.sum()
    p = count / n
    pc, px = p.sum(axis=1), p.sum(axis=0)
    expected = np.outer(pc, px)
    good = p > 0
    mi = float(np.sum(p[good] * np.log2(p[good] / expected[good])))
    h = float(-np.sum(pc * np.log2(pc)))
    chi = float(np.sum((count - n * expected) ** 2 / (n * expected)))
    denom = min(len(cs) - 1, len(xs) - 1)
    return {'n': int(n), 'classes': cs, 'categories': xs, 'counts': count.astype(int).tolist(),
            'mutual_information_bits': mi, 'class_entropy_bits': h,
            'fraction_of_class_entropy_associated': mi / h if h else 0,
            'Cramer_V': math.sqrt(chi / (n * denom)) if denom else 0,
            'interpretation': 'Descriptive in-collection association; no trained classifier, accuracy claim, significance test or causal attribution.'}


def eta_squared(rows, key, log=False):
    selected = [(r['target_class'], r[key]) for r in rows if r.get(key) is not None and (not log or r[key] > 0)]
    if not selected:
        return {'n': 0, 'eta_squared': None}
    labels = [c for c, x in selected]
    values = np.array([math.log(x) if log else x for c, x in selected])
    total = float(np.sum((values - values.mean()) ** 2))
    between = sum(int(np.sum(np.array(labels) == c)) * (values[np.array(labels) == c].mean() - values.mean()) ** 2 for c in set(labels))
    return {'n': len(values), 'eta_squared': float(between / total) if total else 0,
            'transformation': 'natural_log' if log else 'identity', 'missing_excluded': len(rows) - len(values)}


def pooled_quartile_bins(values):
    good = np.array([x for x in values if x is not None and np.isfinite(x)])
    edges = sorted(set(float(x) for x in np.quantile(good, [.25, .5, .75]))) if len(good) else []
    labels = ['missing' if x is None else 'Q' + str(int(np.searchsorted(edges, x, side='left')) + 1) for x in values]
    return labels, edges


def marker(image, x, y):
    pen = ImageDraw.Draw(image)
    for dx1, dy1, dx2, dy2 in [(-8, 0, -3, 0), (3, 0, 8, 0), (0, -8, 0, -3), (0, 3, 0, 8)]:
        pen.line((x + dx1, y + dy1, x + dx2, y + dy2), fill='#00ffff', width=1)


def main():
    begin = time.monotonic()
    policy = read(OUT / 'qc_issue_policy.json')
    rows = read(C / 'candidate_geometry.json')['records']
    objects = {o['candidate_id']: o for o in read(C / 'object_qc_summary.json')['objects']}
    products = read(C / 'technical_qc.json')['products']
    mapping = policy['exact_existing_issue_mapping']
    assert {f for o in objects.values() for f in o['issues']} == set(mapping)
    product_metrics = []
    by_id = collections.defaultdict(list)
    for index, q in enumerate(products):
        path = ROOT / q['repository_relative_path']
        assert sha(path) == q['sha256']
        data, header, cards, integrity = load_fits(path)
        finite = np.isfinite(data)
        central = data[224:288, 224:288]
        values = data[finite]
        hard = []
        if not q['technical_checks_passed']:
            hard.append('missing_corrupt_or_wrong_grid_required_band')
        if not len(values) or float(values.min()) == float(values.max()):
            hard.append('constant_or_allzero_required_band')
        if 1 - finite.mean() > .10 or 1 - np.isfinite(central).mean() > .01:
            hard.append('declared_invalid_global_above10percent_or_central64_above1percent')
        p1, p99 = np.percentile(values, [1, 99]) if len(values) else [0., 0.]
        dynamic = float(p99 - p1)
        background_mask = np.ones((512, 512), bool)
        background_mask[192:320, 192:320] = False
        bg = data[background_mask & finite]
        tiles = []
        for y in range(0, 512, 32):
            for x in range(0, 512, 32):
                if 192 <= x < 320 and 192 <= y < 320:
                    continue
                v = data[y:y + 32, x:x + 32]
                if np.isfinite(v).any():
                    tiles.append(float(np.nanmedian(v)))
        def ratio(v):
            return float(v / dynamic) if dynamic > 0 and math.isfinite(v) else None
        medrow, medcol = np.nanmedian(data, axis=1), np.nanmedian(data, axis=0)
        cfinite = central[np.isfinite(central)]
        metrics = {'candidate_id': q['candidate_id'], 'band': q['band'], 'fits_sha256': q['sha256'],
                   'blocking_reasons': hard, 'all_zero': bool(len(values) and np.all(values == 0)),
                   'finite_fraction': float(finite.mean()), 'central64_invalid_fraction': float(1 - np.isfinite(central).mean()),
                   'zero_fraction': float(np.count_nonzero(data == 0) / data.size),
                   'raw_p01': float(p1), 'raw_p99': float(p99), 'raw_dynamic_range': dynamic,
                   'raw_max_plateau_fraction': q['pixel_screening']['empirical_max_fraction'],
                   'preview_white_fraction_fixed_p99': float(np.count_nonzero(data >= p99) / data.size) if dynamic > 0 else None,
                   'background_tile_p90_p10_over_global_dynamic': ratio(np.percentile(tiles, 90) - np.percentile(tiles, 10)) if tiles else None,
                   'background_IQR_over_global_dynamic': ratio(np.percentile(bg, 75) - np.percentile(bg, 25)) if len(bg) else None,
                   'row_median_step_max_normalized': ratio(float(np.nanmax(np.abs(np.diff(medrow))))),
                   'column_median_step_max_normalized': ratio(float(np.nanmax(np.abs(np.diff(medcol))))),
                   'background_centre_median_difference_normalized': ratio(abs(float(np.median(bg)) - float(np.median(cfinite)))) if len(bg) and len(cfinite) else None,
                   'source_path': q['repository_relative_path']}
        product_metrics.append(metrics)
        by_id[q['candidate_id']].append((q, metrics))
        if (index + 1) % 240 == 0:
            print(json.dumps({'raw_products_checked': index + 1, 'total': 1419}), flush=True)
    features = []
    for row in rows:
        cid = row['candidate_id']
        original = objects[cid]
        bands = by_id[cid]
        categories = {category: [f for f in original['issues'] if mapping[f]['category'] == category] for category in ['a', 'b', 'c', 'd', 'e']}
        hard = sorted({f for q, m in bands for f in m['blocking_reasons']})
        quality = 'deterministically_unusable' if hard else 'pending_genuine_author_review' if categories['b'] else 'quality_usable'
        extent = row['pilot_extent_arcmin']
        feature = {'candidate_id': cid, 'canonical_identifier': row['canonical_identifier'], 'target_class': row['target_class'],
                   'target_subclass': row['target_subclass'], 'selection_role': row['selection_role'],
                   'selection_index_within_class': row['selection_index_within_class'], 'v2c_disposition': original['qc_label'],
                   'quality_triage': quality, 'blocking_reasons': hard, 'issues_by_category': categories,
                   'FOV_arcmin': row['pilot_fov_arcmin'], 'pixel_scale_arcsec': bands[0][0]['WCS']['pixel_scale_arcsec'],
                   'catalogue_extent_arcmin': extent, 'extent_to_FOV_ratio': extent / row['pilot_fov_arcmin'] if extent is not None else None,
                   'extent_in_current_pixels': extent * 60 / bands[0][0]['WCS']['pixel_scale_arcsec'] if extent is not None else None,
                   'extent_in_B15arcsec_pixels': extent * 4 if extent is not None else None,
                   'extent_exceeds_B4arcmin_field': extent > 4 if extent is not None else None,
                   'allzero_bands': [q['band'] for q, m in bands if m['all_zero']],
                   'max_zero_fraction': max(m['zero_fraction'] for q, m in bands),
                   'max_raw_plateau_fraction': max(m['raw_max_plateau_fraction'] for q, m in bands),
                   'catalogue_extent_missing': extent is None, 'catalogue_FOV_capped': row['pilot_fov_was_capped'],
                   'explicit_artifact_issue': any(any(s in f.lower() for s in ['streak', 'stripe', 'step', 'seam', 'discontinuity', 'plate_mark', 'partial_blank', 'partial_coverage']) for f in categories['b']),
                   'ordinary_weak_contrast_or_nebulosity_caveat': any(any(s in f for s in ['weak_', 'nebulosity', 'nebulous']) for f in categories['e']),
                   'catalogue_type_unchanged': row['catalogue_object_type']}
        for key in ['background_tile_p90_p10_over_global_dynamic', 'background_IQR_over_global_dynamic', 'row_median_step_max_normalized', 'column_median_step_max_normalized', 'background_centre_median_difference_normalized']:
            usable = [m[key] for q, m in bands if m[key] is not None]
            feature[key] = max(usable) if usable else None
        # Feasibility test only: proposed4arcmin common support,16x16 pixels at15arcsec.
        target_header = {'CRPIX1': 8.5, 'CRPIX2': 8.5, 'CRVAL1': row['ra_deg'], 'CRVAL2': row['dec_deg'], 'CDELT1': -15 / 3600, 'CDELT2': 15 / 3600}
        support = []
        for q, m in bands:
            positions = [sky_to_pix(q['header'], *pix_to_sky(target_header, x, y)) for x, y in [(.5, .5), (16.5, .5), (16.5, 16.5), (.5, 16.5)]]
            support.append(all(bool(np.all(xy >= .5) and np.all(xy <= 512.5)) for xy in positions))
        feature['B4arcmin_support_available_all_bands'] = all(support)
        features.append(feature)
    save('raw_pixel_audit.json', {'products': product_metrics, 'thresholds_from': 'qc_issue_policy.json', 'training_or_inference': False})
    table('candidate_features.csv', [{k: v for k, v in r.items() if k != 'issues_by_category'} for r in features])
    save('deterministic_triage.json', {'records': features, 'interpretation': 'Quality proposal only; metadata and context alone do not reject, serious review kept pending.'})
    classes = [r['target_class'] for r in features]
    risk = {'no_model_fitted_or_evaluated': True, 'population': 'All473 frozen candidates, before any proposed selection or author decisions',
            'class_distributions': {}, 'categorical_associations': {}, 'continuous_associations': {},
            'background_caveat': 'Outer-field pixels can contain real target emission and neighbours; DN values are not calibrated across photographic plates. These descriptors expose risk rather than identifying true sky background or proving causal leakage.'}
    fov_rows = []
    scalar_keys = ['FOV_arcmin', 'pixel_scale_arcsec', 'catalogue_extent_arcmin', 'extent_in_current_pixels', 'extent_in_B15arcsec_pixels', 'max_zero_fraction', 'max_raw_plateau_fraction', 'background_tile_p90_p10_over_global_dynamic', 'background_IQR_over_global_dynamic', 'row_median_step_max_normalized', 'column_median_step_max_normalized', 'background_centre_median_difference_normalized']
    for cl in sorted(set(classes)):
        subset = [r for r in features if r['target_class'] == cl]
        distribution = {key: stats([r[key] for r in subset]) for key in scalar_keys}
        distribution['FOV_all_values_sorted_arcmin'] = sorted(r['FOV_arcmin'] for r in subset)
        counter = collections.Counter(r['FOV_arcmin'] for r in subset)
        distribution['FOV_value_counts'] = [{'arcmin': v, 'count': n} for v, n in sorted(counter.items())]
        distribution['v2c_screening_counts'] = dict(collections.Counter(r['v2c_disposition'] for r in subset))
        distribution['v2c_review_fraction'] = sum(r['v2c_disposition'] == 'review_needed' for r in subset) / len(subset)
        distribution['allzero_objects'] = sum(bool(r['allzero_bands']) for r in subset)
        distribution['extent_unknown_objects'] = sum(r['catalogue_extent_arcmin'] is None for r in subset)
        distribution['B_extent_above4arcmin_objects'] = sum(r['extent_exceeds_B4arcmin_field'] is True for r in subset)
        distribution['B_extent_below2pixels_objects'] = sum(r['extent_in_B15arcsec_pixels'] is not None and r['extent_in_B15arcsec_pixels'] < 2 for r in subset)
        distribution['B_common_support_available'] = sum(r['B4arcmin_support_available_all_bands'] for r in subset)
        risk['class_distributions'][cl] = distribution
        for rank, r in enumerate(sorted(subset, key=lambda r: (r['FOV_arcmin'], r['candidate_id'])), 1):
            fov_rows.append({'class': cl, 'candidate_id': r['candidate_id'], 'FOV_arcmin': r['FOV_arcmin'], 'pixel_scale_arcsec': r['pixel_scale_arcsec'], 'sorted_rank': rank, 'empirical_CDF': rank / len(subset)})
    fov_bins = ['5 exactly' if r['FOV_arcmin'] == 5 else '(5,10]' if r['FOV_arcmin'] <= 10 else '(10,20]' if r['FOV_arcmin'] <= 20 else '(20,40]' if r['FOV_arcmin'] <= 40 else '(40,80]' if r['FOV_arcmin'] <= 80 else '(80,120]' for r in features]
    risk['categorical_associations']['FOV_predeclared_bins'] = categorical_association(classes, fov_bins)
    for key in ['v2c_disposition', 'catalogue_extent_missing', 'catalogue_FOV_capped', 'explicit_artifact_issue', 'ordinary_weak_contrast_or_nebulosity_caveat']:
        risk['categorical_associations'][key] = categorical_association(classes, [str(r[key]) for r in features])
    risk['categorical_associations']['raw_plateau_above0.5percent'] = categorical_association(classes, [str(r['max_raw_plateau_fraction'] > .005) for r in features])
    for key in scalar_keys:
        bins, edges = pooled_quartile_bins([r[key] for r in features])
        risk['categorical_associations'][key + '_pooled_quartiles'] = {**categorical_association(classes, bins), 'pooled_quartile_edges': edges}
        risk['continuous_associations'][key] = eta_squared(features, key, key in ['FOV_arcmin', 'pixel_scale_arcsec', 'catalogue_extent_arcmin', 'extent_in_current_pixels'])
    risk['representation_gate'] = {'choice_made': False, 'A': 'Existing catalogue-adaptive512x512 inputs; scale and context shortcuts remain measurable.',
                                   'B': 'Feasibility only: common4arcmin TAN support at15arcsec/pixel ->16x16; nested1/2/4arcmin ->4/8/16 pixels. All source bytes unchanged, no new representation rendered.',
                                   'common_support_candidates': sum(r['B4arcmin_support_available_all_bands'] for r in features),
                                   'limits': 'No new data or resampling can restore missing large-field context or native resolution. Equal pixel scale is not equal measured PSF. Candidate-level quality usability is conditional on the chosen representation and controls.'}
    save('shortcut_risk_analysis.json', risk)
    table('FOV_distributions.csv', fov_rows)
    # Deterministic audit sample includes false-pass controls plus review dispositions.
    rng = np.random.default_rng(policy['audit']['seed'])
    sample = []
    for cl in sorted(set(classes)):
        for disposition, n in [('technical_pass', 10), ('review_needed', 8)]:
            candidates = sorted([r for r in features if r['target_class'] == cl and r['v2c_disposition'] == disposition], key=lambda r: (r['FOV_arcmin'], r['selection_index_within_class'], r['candidate_id']))
            for stratum, indexes in enumerate(np.array_split(np.arange(len(candidates)), min(n, len(candidates)))):
                item = candidates[int(rng.choice(indexes))]
                sample.append({'candidate_id': item['candidate_id'], 'target_class': cl, 'source_disposition': disposition, 'FOV_rank_stratum': stratum, 'FOV_arcmin': item['FOV_arcmin']})
    blind_rng = np.random.default_rng(policy['author_review']['blinding_seed'])
    shuffled = sorted(features, key=lambda r: r['candidate_id'])
    shuffled = [shuffled[int(i)] for i in blind_rng.permutation(len(shuffled))]
    keys = []
    for item in shuffled:
        token = 'Q-' + format(int(blind_rng.integers(0, 2**63)), '016X')
        keys.append({'review_id': token, 'candidate_id': item['candidate_id'], 'target_class': item['target_class'], 'target_subclass': item['target_subclass'], 'canonical_identifier': item['canonical_identifier'], 'selection_role': item['selection_role']})
    assert len({r['review_id'] for r in keys}) == 473
    table('review_blinding_key.csv', keys)
    blind = {r['candidate_id']: r['review_id'] for r in keys}
    sample.sort(key=lambda r: blind[r['candidate_id']])
    for item in sample:
        item['review_id'] = blind[item['candidate_id']]
    save('audit_plan.json', {'policy_sha256': sha(OUT / 'qc_issue_policy.json'), 'records': sample, 'sample_size': len(sample), 'labels_hidden_in_mosaics': True})
    (OUT / 'audit_mosaics').mkdir()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    pages = []
    for start in range(0, len(sample), 6):
        group = sample[start:start + 6]
        canvas = Image.new('RGB', (1280, len(group) * 290 + 45), '#eee')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), 'Blinded quality audit: blue | red | NIR | pseudo-RGB | central128 native-pixel RGB crop', font=font, fill='black')
        for i, item in enumerate(group):
            cid = item['candidate_id']
            y = 45 + i * 290
            draw.text((8, y), item['review_id'], font=font, fill='black')
            xy = by_id[cid][0][0]['WCS']['catalogue_position_fits_1based_xy']
            mx, my = xy[0] - 1, 512 - xy[1]
            for col, band in enumerate(['blue', 'red', 'nir', 'pseudo_rgb']):
                with Image.open(C / 'previews' / f'{cid}_{band}.png') as source:
                    image = source.convert('RGB').resize((238, 238), Image.Resampling.LANCZOS)
                marker(image, mx * 238 / 512, my * 238 / 512)
                canvas.paste(image, (col * 256 + 8, y + 25))
            with Image.open(C / 'previews' / f'{cid}_pseudo_rgb.png') as source:
                crop = source.convert('RGB').crop((192, 192, 320, 320))
            marker(crop, mx - 192, my - 192)
            canvas.paste(crop, (1030, y + 65))
        name = f'audit_mosaics/audit_{start // 6 + 1:02d}.png'
        canvas.save(OUT / name)
        pages.append({'page': name, 'review_ids': [r['review_id'] for r in group], 'sha256': sha(OUT / name)})
    save('audit_mosaic_index.json', {'pages': pages, 'screening_scale': '238-pixel panels plus unresized128-pixel crop; no class or subtype labels'})
    print(json.dumps({'all_candidates_triaged': len(features), 'quality_counts': dict(collections.Counter(r['quality_triage'] for r in features)), 'audit_size': len(sample), 'audit_pages': len(pages), 'elapsed_seconds': time.monotonic() - begin}))


def boundary(event, args):
    if event.startswith('socket.') or event == 'urllib.Request':
        raise RuntimeError('Dataset-v2D is offline')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        if write and not p.is_relative_to(OUT):
            raise RuntimeError('Write outside Dataset-v2D')
        if 'baseline_comparison_v1' in p.parts or p.suffix.lower() in {'.pt', '.pth', '.h5', '.hdf5', '.safetensors'}:
            raise RuntimeError('No v1 or checkpoint access')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    sys.addaudithook(boundary)
    main()
