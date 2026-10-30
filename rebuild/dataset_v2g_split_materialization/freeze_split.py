"""Materialize and validate manifests from the single saved optimizer result."""
import collections
import csv
import datetime
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
F=ROOT/'rebuild/dataset_v2f_representation_split_protocol'
E=ROOT/'rebuild/dataset_v2e_author_quality_freeze'
ROLES=('train','validation','test')


def read(path):return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def table(name,rows,fields=None):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as stream:
        columns=fields or list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def preserved():
    entries=read(OUT/'preservation_input_hashes.json')['files']
    for entry in entries:
        p=ROOT/entry['path'];assert p.stat().st_size==entry['size_bytes'] and sha(p)==entry['sha256'],entry['path']
    actual={str(p.relative_to(ROOT)) for name in ['dataset_v2a_catalog_manifest','dataset_v2b_pilot','dataset_v2c_acquisition_qc','dataset_v2d_curation_shortcut_gate','dataset_v2e_author_quality_freeze','dataset_v2f_representation_split_protocol'] for p in (ROOT/'rebuild'/name).rglob('*') if p.is_file()}
    assert actual=={r['path'] for r in entries}
    return len(entries)


def normalize(name):
    # Validation of already frozen alias relations only, no new identity decision.
    import re,unicodedata
    name=unicodedata.normalize('NFKC',name).strip()
    for prefix in ['SIMBAD:','HEASARC-GLOBCLUST:']:
        if name.upper().startswith(prefix):name=name[len(prefix):];break
    return re.sub(r'\s+','',name.replace('\u2212','-').replace('\u2013','-')).casefold()


def integer_optimum_counts(total):
    # Independent brute-force count-only bound, never assigns objects or invokes F.
    target=[Fraction(3,5),Fraction(1,5),Fraction(1,5)]
    feasible=[]
    for train in range(21,total+1):
        for validation in range(6,total-train+1):
            test=total-train-validation
            if test<10:continue
            counts=(train,validation,test)
            value=sum((Fraction(counts[i],total)-target[i])**2 for i in range(3))
            feasible.append((value,counts))
    best=min(cost for cost,counts in feasible)
    return best,[list(counts) for cost,counts in feasible if cost==best]


def main():
    assert (OUT/'optimizer_completion.json').exists(),'Frozen optimizer still running or failed; never invoke a second optimizer'
    assert not (OUT/'split_manifest.json').exists(),'A split is already frozen; use read-only validation'
    nfiles=preserved()
    completion=read(OUT/'optimizer_completion.json');invocation=read(OUT/'optimizer_invocation.json')
    assert completion['optimizer_invocations']==invocation['invocation_number']==1
    result_hash_before=sha(OUT/'optimizer_result.json')
    correction=read(OUT/'export_correction_record.json')
    assert correction['optimizer_result_sha256_before_correction']==result_hash_before
    for archived in correction['files']:
        path=OUT/archived['archived_path']
        assert path.stat().st_size==archived['size_bytes'] and sha(path)==archived['sha256']
    assert completion['result_sha256']==result_hash_before
    assert completion['source_algorithm_sha256']==sha(F/'split_algorithm_reference.py')==sha(OUT/'frozen_inputs/split_algorithm_reference.py')
    author=read(OUT/'author_resolution.json')
    f_split=read(OUT/'frozen_inputs/split_protocol.json');f_rep=read(OUT/'frozen_inputs/representation_protocol.json');f_controls=read(OUT/'frozen_inputs/shortcut_control_protocol.json')
    groups=read(OUT/'frozen_inputs/object_group_inventory.json')
    profile=f_split['alternative_profiles_require_author_decision']['phase4f2']
    assert author['adopted_split_profile']=='phase4f2' and profile['fractions_exact']==author['fractions_exact']==['3/5','1/5','1/5']
    assert profile['minimum_distinct_objects_per_class']==author['class_object_floors']==dict(zip(ROLES,[21,6,10]))
    assert author['split_seed']==profile['split_seed']==3908274 and author['primary_representation']=='A'
    assert author['mandatory_image_control']=='B'
    assert set(author['mandatory_controls'])=={'B_common_field','metadata_FOV_only','outer_field','FOV_stratified_reporting'}
    adopted={'phase':'Dataset-v2G','source_F_inventory_sha256':sha(F/'file_inventory.json'),
             'source_F_bundle_sha256':read(OUT/'frozen_inputs/protocol_bundle_manifest.json')['canonical_protocol_bundle_sha256'],
             'author_resolution_sha256':sha(OUT/'author_resolution.json'),'adopted_profile':'phase4f2',
             'fractions_exact':profile['fractions_exact'],'object_floors_per_class':profile['minimum_distinct_objects_per_class'],
             'split_seed':3908274,'algorithm_source_sha256':sha(F/'split_algorithm_reference.py'),
             'primary_representation':'A','mandatory_common_field_control':'B','mandatory_controls':author['mandatory_controls'],
             'representation_A_recipe_sha256':canonical({'common_contract':f_rep['common_contract'],'A':f_rep['A']}),
             'representation_B_recipe_sha256':canonical({'common_contract':f_rep['common_contract'],'B':f_rep['B']}),
             'unchanged_frozen_representation_protocol_sha256':sha(OUT/'frozen_inputs/representation_protocol.json'),
             'unchanged_frozen_shortcut_protocol_sha256':sha(OUT/'frozen_inputs/shortcut_control_protocol.json'),
             'scientific_recipes_changed':False,'primary_may_change_after_model_accuracy':False,
             'F_historical_pending_records_modified':False,'tensors_or_training_authorized_in_G':False}
    save('adopted_protocol.json',adopted)
    result=read(OUT/'optimizer_result.json');assignment=result['assignments']
    active=[u for u in groups['split_units_full_universe'] if u['eligible_candidate_ids']]
    assert set(assignment)=={u['split_unit_id'] for u in active} and len(assignment)==358
    assert set(assignment.values())==set(ROLES) and result['enumerations']==177147
    candidates=groups['primary_eligibility_candidates'];by_candidate={r['candidate_id']:r for r in candidates}
    e_accepted=read(E/'accepted_inventory.json');e_items=e_accepted['carried_forward_D_deterministic_records']+e_accepted['author_accepted_records']
    e_by_id={r['candidate_id']:r for r in e_items}
    assert set(by_candidate)==set(e_by_id) and len(candidates)==370
    exclusions={r['candidate_id'] for name in ['rejected_product_queue.json','deferred_review_queue.json'] for r in read(E/name)['records']}
    assert len(exclusions)==30 and not exclusions&set(by_candidate)
    unit_by_cid={cid:u['split_unit_id'] for u in groups['split_units_full_universe'] for cid in u['full_473_universe_members']}
    object_by_cid={cid:g['physical_object_id'] for g in groups['physical_object_groups_all_473'] for cid in g['all_source_candidate_ids']}
    assert len(unit_by_cid)==len(object_by_cid)==473
    object_rows=[];image_rows=[]
    for source in sorted(candidates,key=lambda r:(r['physical_object_id'],r['candidate_id'])):
        cid=source['candidate_id'];role=assignment[source['split_unit_id']]
        assert source['products']==e_by_id[cid]['products']
        row={**source,'split':role,'membership_scope':'primary_quality_ready_multiband_observation',
             'source_master_count':3,'statistical_unit':'physical_astronomical_object',
             'primary_representation':'A','mandatory_common_field_control':'B','tensor_materialized':False,
             'source_E_accepted_inventory_sha256':sha(E/'accepted_inventory.json')}
        object_rows.append(row)
        for product in source['products']:
            path=ROOT/product['repository_relative_path']
            assert path.stat().st_size==product['size_bytes'] and sha(path)==product['sha256']
            image_rows.append({'candidate_id':cid,'physical_object_id':source['physical_object_id'],
                               'split_unit_id':source['split_unit_id'],'split':role,
                               'frozen_class':source['frozen_class'],'frozen_subtype':source['frozen_subtype'],
                               **product,'quality_evidence_level':source['quality_evidence_level'],
                               'role':'unchanged_FITS_master_channel_of_one_object','new_independent_statistical_observation':False})
    assert len(object_rows)==len({r['physical_object_id'] for r in object_rows})==370
    assert len(image_rows)==len({r['repository_relative_path'] for r in image_rows})==1110
    unit_rows=[]
    for unit in groups['split_units_full_universe']:
        uid=unit['split_unit_id'];role=assignment.get(uid)
        assert bool(role)==bool(unit['eligible_candidate_ids'])
        unit_rows.append({**unit,'future_split':role,'materialized_constraint_split':role,
                          'primary_pool_unit':bool(unit['eligible_candidate_ids']),
                          'noneligible_members_inherit_constraint_without_admission':True})
    source_status={r['candidate_id']:r for r in read(E/'prospective_inventory_effect.json')['records']}
    full_rows=[]
    for cid in sorted(unit_by_cid):
        eligible=cid in by_candidate;status=source_status[cid]
        full_rows.append({'candidate_id':cid,'physical_object_id':object_by_cid[cid],
                          'canonical_identifier':status['canonical_identifier'],'frozen_class':status['frozen_class'],
                          'frozen_subtype':status['frozen_subtype'],'source_E_status':status['E_status'],
                          'split_unit_id':unit_by_cid[cid],'eligible_for_primary_pool':eligible,
                          'primary_split':assignment[unit_by_cid[cid]] if eligible else None,
                          'inherited_constraint_split':assignment.get(unit_by_cid[cid]),
                          'noneligible_member_not_admitted':not eligible})
    derivative_rows=[]
    for d in groups['standalone_derivative_lineage']:
        assert d['physical_object_id']==object_by_cid[d['candidate_id']]
        assert d['split_unit_id']==unit_by_cid[d['candidate_id']]
        derivative_rows.append({**d,'inherited_constraint_split':assignment.get(d['split_unit_id']),
                                'primary_parent_split':assignment[d['split_unit_id']] if d['parent_eligible_in_E_primary_pool'] else None,
                                'assigned_as_scientific_observation':False,'model_input_authorized':False})
    assert len(derivative_rows)==2534 and sum(r['parent_eligible_in_E_primary_pool'] for r in derivative_rows)==1872
    save('object_membership.json',{'records':object_rows,'physical_objects':370,'one_membership_per_object':True})
    table('object_membership.csv',object_rows)
    save('image_membership.json',{'records':image_rows,'FITS_master_images':1110,'bands_per_object':3,'separate_band_samples_forbidden':True})
    table('image_membership.csv',image_rows)
    save('assignment_unit_membership.json',{'records':unit_rows,'active_units':358,'all_full_universe_units':len(unit_rows)})
    table('assignment_unit_membership.csv',unit_rows)
    save('full_candidate_membership.json',{'records':full_rows,'source_universe_candidates':473,'primary_pool_candidates':370,
                                         'noneligible_candidates':103,'noneligible_constraints_are_not_admissions':True})
    table('full_candidate_membership.csv',full_rows)
    save('derivative_membership.json',{'records':derivative_rows,'lineage_only_not_model_inputs':True,
                                     'count':2534,'eligible_parent_count':1872})
    table('derivative_membership.csv',derivative_rows)
    classes=sorted({r['frozen_class'] for r in object_rows});statistics={'by_split':{},'by_class':{},'totals':{}}
    count_table=[];class_units=[]
    for role in ROLES:
        objects=[r for r in object_rows if r['split']==role]
        images=[r for r in image_rows if r['split']==role]
        units=[r for r in unit_rows if r['materialized_constraint_split']==role]
        per_class={}
        for cl in classes:
            oo=[r for r in objects if r['frozen_class']==cl]
            ii=[r for r in images if r['frozen_class']==cl]
            uu={r['split_unit_id'] for r in oo}
            assert len(ii)==3*len(oo)
            row={'split':role,'frozen_class':cl,'object_observations':len(oo),
                 'distinct_physical_objects':len({r['physical_object_id'] for r in oo}),
                 'FITS_band_images':len(ii),'assignment_units_touching_class':len(uu),
                 'D_deterministic_quality':sum(r['quality_evidence_level']=='D_deterministic_quality_ready_not_author_adjudication' for r in oo),
                 'E_author_accept_quality':sum(r['quality_evidence_level']=='E_explicit_author_accept_quality' for r in oo)}
            assert row['distinct_physical_objects']>=author['class_object_floors'][role]
            per_class[cl]=row;count_table.append(row)
        summary={'objects':len(objects),'distinct_physical_objects':len({r['physical_object_id'] for r in objects}),
                 'FITS_master_images':len(images),'active_assignment_units':len(units),
                 'object_fraction_exact':str(Fraction(len(objects),370)),
                 'object_fraction_decimal':len(objects)/370,
                 'target_fraction_exact':str(Fraction(3,5) if role=='train' else Fraction(1,5)),
                 'target_object_count':222 if role=='train' else 74,
                 'integer_deviation_from_global_target':len(objects)-(222 if role=='train' else 74),
                 'standalone_derivatives_of_eligible_parents':sum(r['primary_parent_split']==role for r in derivative_rows),
                 'all_lineage_derivatives_inheriting_constraint_role':sum(r['inherited_constraint_split']==role for r in derivative_rows),
                 'per_class':per_class}
        statistics['by_split'][role]=summary
        save(f'splits/{role}.json',{'role':role,'adopted_profile':'phase4f2','split_seed':3908274,
                                   'primary_representation':'A','mandatory_common_field_control':'B',
                                   'objects':objects,'master_images':images,'assignment_unit_ids':sorted(r['split_unit_id'] for r in units),
                                   'counts':summary,'tensors_created':False})
        table(f'splits/{role}.csv',[r for r in image_rows if r['split']==role])
    targets=[Fraction(3,5),Fraction(1,5),Fraction(1,5)];observed_objective=Fraction(0);lower_bound=Fraction(0);objective_proof={}
    for cl in classes:
        counts=[statistics['by_split'][role]['per_class'][cl]['distinct_physical_objects'] for role in ROLES]
        n=sum(counts);bound,optimal_counts=integer_optimum_counts(n)
        cost=sum((Fraction(counts[i],n)-targets[i])**2 for i in range(3))
        observed_objective+=cost;lower_bound+=bound
        assert counts==result['counts'][cl]
        objective_proof[cl]={'objects':n,'actual_train_validation_test':counts,'exact_class_objective':str(cost),
                             'unconstrained_integer_lower_bound':str(bound),'all_optimal_count_vectors':optimal_counts,
                             'floors_satisfied':True,'actual_reaches_integer_lower_bound':cost==bound,
                             'actual_object_fractions_exact':[str(Fraction(c,n)) for c in counts]}
        statistics['by_class'][cl]=objective_proof[cl]
    assert observed_objective==Fraction(result['objective_exact'])
    assert observed_objective==lower_bound,'Frozen solution failed independent count-only optimality lower bound'
    # Independent tie/order verification without a second optimization. Here every
    # class has enough singleton objects to realize each unconstrained optimal
    # count vector for every coupled assignment. The first seeded assignment is
    # therefore globally optimal under the frozen enumeration rule.
    optimizer_inputs=read(OUT/'optimizer_input.json')
    coupled_input=[u for u in optimizer_inputs['units'] if sum(u['weights_by_class'].values())>1]
    coupled_class_totals={cl:sum(u['weights_by_class'].get(cl,0) for u in coupled_input) for cl in classes}
    independent_tie_condition=all(coupled_class_totals[cl]<=min(min(v) for v in objective_proof[cl]['all_optimal_count_vectors']) for cl in classes)
    role_priority=sorted(ROLES,key=lambda role:(hashlib.sha256(f'Dataset-v2F|3908274|role|{role}'.encode()).hexdigest(),ROLES.index(role)))
    assert independent_tie_condition,'Cannot certify first-tie analytically; do not produce another split'
    assert all(assignment[u['unit_id']]==role_priority[0] for u in coupled_input)
    for cl in classes:
        best_counts=min(objective_proof[cl]['all_optimal_count_vectors'],key=lambda v:(-v[0],-v[1]))
        assert objective_proof[cl]['actual_train_validation_test']==best_counts
        free=sorted([u for u in optimizer_inputs['units'] if u['weights_by_class']=={cl:1}],
                    key=lambda u:(hashlib.sha256(f"Dataset-v2F|3908274|singleton|{cl}|{u['unit_id']}".encode()).hexdigest(),u['unit_id']))
        coupled_per_role=[sum(u['weights_by_class'].get(cl,0) for u in coupled_input if assignment[u['unit_id']]==role) for role in ROLES]
        singleton_counts=[best_counts[i]-coupled_per_role[i] for i in range(3)]
        position=0
        for role,count in zip(ROLES,singleton_counts):
            assert all(assignment[u['unit_id']]==role for u in free[position:position+count])
            position+=count
        assert position==len(free)
    save('seeded_order_validation.json',{'passed':True,'optimizer_invocations_for_validation':0,
                                        'all_coupled_assignments_can_reach_integer_lower_bound':independent_tie_condition,
                                        'coupled_object_counts_by_class':coupled_class_totals,'seeded_role_priority':role_priority,
                                        'first_seeded_coupled_assignment_verified':True,
                                        'train_then_validation_integer_count_ties_verified':True,
                                        'class_singleton_SHA_order_and_role_boundaries_verified':True,
                                        'source_algorithm_sha256':completion['source_algorithm_sha256']})
    statistics['totals']={'objects':370,'distinct_physical_objects':370,'FITS_master_images':1110,
                          'active_assignment_units':358,'all_universe_assignment_units':454,'all_source_candidates':473,
                          'standalone_derivative_lineages':2534,'derivatives_with_primary_eligible_parent':1872,
                          'class_touching_unit_counts_are_nonadditive_for_cross_class_unit':True}
    statistics['fraction_interpretation']='60/20/20 is the target under exact per-class rational squared-error minimization, floors and frozen tie rules. Integer counts may deviate from global74/74; no post-hoc rebalancing or second split is permitted.'
    save('split_statistics.json',statistics);table('counts_by_split_class.csv',count_table)
    table('counts_by_split.csv',[{k:v for k,v in statistics['by_split'][role].items() if k!='per_class'}|{'split':role} for role in ROLES])
    save('objective_validation.json',{'source_result_sha256':result_hash_before,'class_count_bruteforce_only':True,
                                      'real_object_assignment_optimizer_invoked_for_validation':False,
                                      'by_class':objective_proof,'actual_exact_objective':str(observed_objective),
                                      'unconstrained_integer_lower_bound':str(lower_bound),
                                      'global_optimality_proven_by_achieved_lower_bound':True,
                                      'seeded_assignment_ties':'Chosen once by unchanged frozen F enumeration and role/unit ordering; not reoptimized.'})
    # Independence checks use membership maps only; no source pixel decoding.
    sets={role:{r['candidate_id'] for r in object_rows if r['split']==role} for role in ROLES}
    assert len(set.union(*sets.values()))==370 and sum(len(s) for s in sets.values())==370
    for i,role in enumerate(ROLES):
        for other in ROLES[i+1:]:assert not sets[role]&sets[other]
    physical_roles=collections.defaultdict(set);alias_roles=collections.defaultdict(set);content_roles=collections.defaultdict(set)
    for row in object_rows:
        physical_roles[row['physical_object_id']].add(row['split'])
        for name in [row['canonical_identifier'],*row['aliases']]:alias_roles[normalize(name)].add(row['split'])
    for row in image_rows:content_roles[row['sha256']].add(row['split'])
    assert all(len(s)==1 for lookup in [physical_roles,alias_roles,content_roles] for s in lookup.values())
    overlap_checks=[]
    for comp in groups['original17_nontrivial_components']:
        role_set={assignment.get(unit_by_cid[cid]) for cid in comp['candidate_ids']}
        assert len(role_set)==1
        overlap_checks.append({'original_component_id':comp['component_id'],'all_source_candidate_ids':comp['candidate_ids'],
                               'eligible_members':comp['eligible_members'],'constraint_split':next(iter(role_set)),
                               'all_members_indivisible_including_ineligible_bridges':True})
    assert len(overlap_checks)==17
    for derivative in derivative_rows:
        assert derivative['inherited_constraint_split']==assignment.get(unit_by_cid[derivative['candidate_id']])
        if derivative['parent_eligible_in_E_primary_pool']:
            assert derivative['primary_parent_split']==assignment[by_candidate[derivative['candidate_id']]['split_unit_id']]
    save('grouping_validation.json',{'passed':True,'physical_object_cross_split_overlap':0,'normalized_alias_cross_split_overlap':0,
                                     'exact_eligible_master_SHA_cross_split_overlap':0,'derivative_parent_role_violations':0,
                                     'original_overlap_components':overlap_checks,'original_nontrivial_components_preserved':17,
                                     'inactive_units_not_assigned_or_admitted':96,
                                     'review_montages_not_model_inputs_and_not_observations':True,
                                     'noneligible_constraint_role_does_not_admit_object':True})
    assert sha(OUT/'optimizer_result.json')==result_hash_before
    save('result_only_validation_receipt.json',{'passed':True,'optimizer_calls_in_this_validation':0,
                                               'total_real_optimizer_invocations':1,'saved_result_sha256_before_and_after':result_hash_before,
                                               'no_new_seed_or_alternative_split':True,'tensors_training_or_model_inference':False})
    membership_files=['object_membership.json','object_membership.csv','image_membership.json','image_membership.csv',
                      'assignment_unit_membership.json','assignment_unit_membership.csv','full_candidate_membership.json','full_candidate_membership.csv',
                      'derivative_membership.json','derivative_membership.csv','split_statistics.json','counts_by_split_class.csv',
                      'grouping_validation.json','objective_validation.json','seeded_order_validation.json']
    membership_files += [f'splits/{role}.{ext}' for role in ROLES for ext in ['json','csv']]
    binding={'schema':'Dataset-v2G-object-grouped-split-v1','quality_authority_E_inventory_sha256':sha(E/'file_inventory.json'),
             'source_F_inventory_sha256':sha(F/'file_inventory.json'),
             'source_F_protocol_bundle_sha256':adopted['source_F_bundle_sha256'],
             'author_resolution_sha256':sha(OUT/'author_resolution.json'),'adopted_protocol_sha256':sha(OUT/'adopted_protocol.json'),
             'optimizer_input_sha256':sha(OUT/'optimizer_input.json'),'optimizer_result_sha256':result_hash_before,
             'source_algorithm_sha256':sha(F/'split_algorithm_reference.py'),
             'membership_and_role_files':{name:{'sha256':sha(OUT/name),'size_bytes':(OUT/name).stat().st_size} for name in membership_files},
             'primary_representation':'A','mandatory_controls':author['mandatory_controls'],
             'fractions_exact':['3/5','1/5','1/5'],'floors_train_validation_test':[21,6,10],'split_seed':3908274,
             'statistical_unit':'physical_astronomical_object','materialized_splits':1,'tensors_created':False}
    split_digest=canonical(binding)
    save('split_manifest.json',{'binding':binding,'canonical_split_sha256':split_digest,
                                'canonicalization':'UTF8 JSON of binding, sorted keys, separators comma/colon, ensure_ascii=False, allow_nan=False; SHA256.',
                                'materialized_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'exactly_one_split':True,'counts':statistics['by_split']})
    (OUT/'split_digest.txt').write_text(split_digest+'\n')
    save('split_freeze_receipt.json',{'canonical_split_sha256':split_digest,'split_manifest_file_sha256':sha(OUT/'split_manifest.json'),
                                     'memberships_and_policy_bound':True,'primary_representation_cannot_change_from_accuracy':True,
                                     'frozen_for_author_review':True,'tensor_or_training_next_phase_authorized':False})
    report=['# Dataset-v2G — one frozen object-grouped split','',
            '**Exactly one split was materialized and validated.** The author explicitly adopted the complete phase4f2 numerical profile and representationA as primary. Dataset-v2A–v2F remain byte-for-byte unchanged. No tensor, model, inference, prediction or benchmark result was generated or inspected.','',
            '## Author resolution and frozen method','',
            'The exact author message is preserved in author_resolution.json. Adopted train/validation/test target fractions are60/20/20, distinct-object class floors21/6/10, split seed3908274. No parts were imported from phase4f or other profiles. A remains the primary adaptive full-field scientific view; B remains the mandatory common4arcmin/16×16-effective-sample shortcut control. Metadata/FOV-only, outer-field and FOV-stratified controls remain mandatory regardless of outcome. No future accuracy can change the primary representation.','',
            'All numerical FITS-to-tensor and control recipes are the unchanged frozen F recipes; no tensor is materialized now. F’s historical pending-state documents remain intact; their two gates are resolved only in this separately recorded G phase. The source F inventory SHA-256 is `'+sha(F/'file_inventory.json')+'`.','',
            'The original split_algorithm_reference.py was imported and optimize was invoked exactly once on358 active assignment units. It enumerated177147 coupled-unit assignments using the unchanged seed/tie rules and rational objective. No implementation acceleration, seed search, manual adjustment, alternate split or profile mixing occurred. Runtime and source/result hashes are retained in optimizer_invocation.json and optimizer_completion.json.','',
            '## Exact primary split counts','',
            '| Split | Physical objects / multiband observations | FITS band images | Assignment units | Actual object fraction |','|---|---:|---:|---:|---:|']
    for role in ROLES:
        s=statistics['by_split'][role]
        report.append(f"| {role} | {s['distinct_physical_objects']} | {s['FITS_master_images']} | {s['active_assignment_units']} | {s['object_fraction_decimal']:.6%} |")
    report+=['| **Total** | **370** | **1110** | **358** | **100%** |','',
             '| Frozen class | Train objects / FITS / units | Validation objects / FITS / units | Test objects / FITS / units | Floors |','|---|---|---|---|---|']
    for cl in classes:
        cells=[]
        for role in ROLES:
            x=statistics['by_split'][role]['per_class'][cl]
            cells.append(f"{x['distinct_physical_objects']} / {x['FITS_band_images']} / {x['assignment_units_touching_class']}")
        report.append('| '+cl+' | '+' | '.join(cells)+' | 21/6/10 satisfied |')
    report+=['',
             'The statistical unit is the physical astronomical object, with three jointly used bands. Class-touching assignment-unit counts are not additive where one original unit spans two classes; the overall358 units are unique. All370 E quality-ready objects are represented exactly once, preserving330 deterministic and40 explicit author-quality evidence levels.13 rejected products and17 deferred cases remain excluded, with no new reserve promotion or replacement.','',
             '## Fractions, integer allocation and optimality','',
             'The60/20/20 fractions are the frozen per-class object-count optimization targets, not a requirement to override exact integer/tie/group constraints. Global ideal counts are222/74/74. Actual counts and integer deviations are recorded in split_statistics.json. No post-hoc rebalance was performed to force global fractions: that would change the frozen algorithm. Every class satisfies21 train,6 validation and10 test distinct-object floors.','',
             'The independently recomputed exact squared-deviation objective is `'+str(observed_objective)+'`. Brute-force integer **count-only** lower bounds were calculated separately for each class; their sum equals the saved optimizer objective. The result therefore reaches the global lower bound even with the indivisible group constraints. This validation creates no second membership assignment and does not call optimize again. Frozen source/input hashes and the one invocation record establish the seeded tie/order procedure.','',
             '## Complete memberships and grouping','',
             '- object_membership.json/.csv:370 primary objects, exact class/subtype/evidence, aliases, source master hashes and role.','- image_membership.json/.csv:1110 unchanged FITS masters with repository-relative path, byte size, SHA-256, band, survey and owning object/unit/role.','- splits/train.json, validation.json and test.json plus CSV companions are the three roles of this single split. They reference masters; no images are moved or copied into tensor directories.','- assignment_unit_membership.json/.csv: all454 full-universe units, including358 active and96 inactive units.','- full_candidate_membership.json/.csv: all473 source candidates. Noneligible members may inherit a constraint role from an eligible neighbour without becoming primary observations.','- derivative_membership.json/.csv:2534 standalone derivative lineages, including1872 with eligible parents. Every derivative inherits its owning object/unit constraint. QC previews, native review crops and panels remain prohibited model inputs. Multi-object review montages are not independent astronomical observations.','',
             'All17 original nontrivial overlap components remain indivisible, including rejected/deferred/blocked/reserve bridges and the cross-class unit. Original physical identities, normalized aliases, exact eligible FITS hashes and known derivatives have zero cross-split violations. The original full473-universe closure is preserved before eligibility projection. No grouped relation was dissolved to improve class balance.','',
             '## Frozen split digest','',
             '`'+split_digest+'`','',
             'split_manifest.json defines the canonical binding and SHA-256. It binds the explicit author resolution, complete adopted profile, immutable F algorithm/protocol/grouping, E quality authority, all object/image/unit/full-candidate/derivative memberships, role manifests, grouping/optimality evidence and mandatory representation/control roles. Every membership file has a byte size and SHA-256 in that binding. split_digest.txt and split_freeze_receipt.json repeat the digest. The broader file_inventory.json additionally covers this report, execution and preservation evidence.','',
             '## Preservation, validation and stop','',
             f'All{nfiles:,} A–F files were hashed at start and verified unchanged at completion. All1110 referenced master image hashes/sizes were independently verified without decoding pixels. Initial/final restricted Git status is recorded and outside this G directory is unchanged. The known expected preexisting completed-author-copy change in D was already bound by E; no old inventory/report/template is rewritten.','',
             'Result-only validation reuses the one immutable optimizer result and creates no alternative split. All memberships are unique, exhaustive and group-isolated; all adopted floors and exact rational targets/objective are verified. No tensors, FITS-to-tensor processing, new data, model training/inference, checkpoint access, model outcome inspection, network, manuscript edit, staging, commit or push occurred.','',
             'The first manifest-export attempt stopped because D review-asset derivative rows contain additional review_id/ownership provenance columns. Its exact code and partial outputs are preserved under implementation_attempts/export01_heterogeneous_derivative_columns/ and bound by export_correction_record.json. The correction forms a deterministic union of CSV columns. Both export attempts use the identical saved optimizer result; optimize was never invoked again and no role, scientific policy, source artifact or membership value was changed. Only one canonical split manifest was created and frozen.','',
             'Stop after this split freeze and author review. Tensor materialization, training and evaluation require separate authorization. The final inventory SHA-256 is in inventory_receipt.json and the final response; the inventory and its receipt exclude themselves to avoid circular hashes.','']
    (OUT/'REPORT.md').write_text('\n'.join(report))
    validation={'passed':True,'exactly_one_materialized_split':True,'optimizer_invocations':1,'optimizer_invocations_during_validation':0,
                'frozen_F_algorithm_source_unmodified':True,'complete_phase4f2_profile_adopted':True,'fraction_targets_exact':['3/5','1/5','1/5'],
                'class_object_floors':{'train':21,'validation':6,'test':10},'split_seed':3908274,
                'class_floor_violations':0,'exact_optimizer_objective_independently_recomputed':str(observed_objective),
                'global_objective_lower_bound_achieved':True,'integer_global_fraction_deviations_explicit':True,
                'seeded_coupled_tie_and_singleton_order_independently_verified':True,
                'manifest_export_attempts':2,'export_correction_preserved':True,'same_optimizer_result_used_for_both_exports':True,
                'primary_representation':'A','B_and_all_other_shortcut_controls_mandatory':True,
                'eligible_physical_objects':370,'FITS_master_images':1110,'active_assignment_units':358,
                'all_source_candidates_accounted_for':473,'all_full_universe_units':454,
                'original_nontrivial_overlap_components_preserved':17,'standalone_derivative_lineages_preserved':2534,
                'physical_object_cross_split_overlap':0,'normalized_alias_cross_split_overlap':0,'exact_eligible_master_SHA_cross_split_overlap':0,
                'derivative_parent_role_violations':0,'all_primary_memberships_unique_and_exhaustive':True,
                'rejected13_and_deferred17_excluded':True,'reserves_or_replacements_newly_admitted':0,
                'previous_phase_files_preserved_byte_for_byte':nfiles,'source_F_inventory_sha256':sha(F/'file_inventory.json'),
                'canonical_split_sha256':split_digest,'tensors_materialized':0,'training_or_model_inference':False,
                'checkpoint_or_model_outcome_access':False,'network_or_downloads':False,'manuscript_or_Git_mutations':False,
                'counts':statistics['by_split'],'training_authorized_by_G':False}
    save('validation.json',validation)
    assert sha(OUT/'optimizer_result.json')==result_hash_before
    assert preserved()==nfiles
    paths=[f'rebuild/{n}' for n in ['dataset_v2a_catalog_manifest','dataset_v2b_pilot','dataset_v2c_acquisition_qc',
                                    'dataset_v2d_curation_shortcut_gate','dataset_v2e_author_quality_freeze','dataset_v2f_representation_split_protocol',OUT.name,'baseline_comparison_v1']]
    for name in ['final_git_status.txt','file_inventory.json','inventory_receipt.json']:(OUT/name).touch(exist_ok=False)
    def status():return subprocess.check_output(['git','-c','core.fsmonitor=false','status','--porcelain=v1','--untracked-files=all','--',*paths],cwd=ROOT,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'})
    final=status()
    def outside(raw):return [line for line in raw.decode().splitlines() if f'rebuild/{OUT.name}/' not in line]
    assert outside(final)==outside((OUT/'initial_git_status.txt').read_bytes())
    (OUT/'final_git_status.txt').write_bytes(final)
    excluded={'file_inventory.json','inventory_receipt.json'}
    files=[{'path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and str(p.relative_to(OUT)) not in excluded]
    inventory={'files':files,'envelope_exclusions':sorted(excluded),'canonical_split_sha256':split_digest,
               'source_preservation_evidence':'preservation_input_hashes.json','exactly_one_materialized_split':True}
    (OUT/'file_inventory.json').write_text(json.dumps(inventory,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    for e in files:assert sha(ROOT/e['path'])==e['sha256']
    receipt={'inventory_sha256':sha(OUT/'file_inventory.json'),'canonical_split_sha256':split_digest,
             'report_sha256':sha(OUT/'REPORT.md'),'inventory_files':len(files),'inventoried_bytes':sum(e['size_bytes'] for e in files),
             'all_listed_hashes_reverified':True,'previous_phase_files_preserved':nfiles,
             'restricted_Git_status_unchanged_outside_G':True,'optimizer_invocations':1}
    (OUT/'inventory_receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    assert status()==final
    for p in OUT.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print(json.dumps({'passed':True,'exactly_one_split':True,'split_counts':{r:{k:v for k,v in statistics['by_split'][r].items() if k!='per_class'} for r in ROLES},
                      'class_counts':result['counts'],'split_digest':split_digest,'inventory_sha256':receipt['inventory_sha256']}))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    def boundary(event,args):
        if event.startswith('socket.') or event=='urllib.Request':raise RuntimeError('Dataset-v2G validation is offline')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
            if write and not p.is_relative_to(OUT):raise RuntimeError('Write outside Dataset-v2G')
            if 'baseline_comparison_v1' in p.parts or p.suffix.lower() in {'.pt','.pth','.h5','.hdf5','.safetensors'}:raise RuntimeError('No model/checkpoint/historical-result access')
    sys.addaudithook(boundary)
    main()
