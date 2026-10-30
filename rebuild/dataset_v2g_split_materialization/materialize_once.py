"""One invocation of the exact frozen F optimizer; no tensor or model code."""
import datetime
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
F = ROOT / 'rebuild/dataset_v2f_representation_split_protocol'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, data):
    with (OUT / name).open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    (OUT / name).chmod(0o444)


def main():
    # The existence of the immutable invocation record forbids rerunning this worker.
    assert not (OUT / 'optimizer_invocation.json').exists(), 'Exactly one optimizer invocation; use result-only validation'
    source = F / 'split_algorithm_reference.py'
    assert sha(source) == sha(OUT / 'frozen_inputs/split_algorithm_reference.py')
    group = read(OUT / 'frozen_inputs/object_group_inventory.json')
    adopted = read(OUT / 'author_resolution.json')
    profile = read(OUT / 'frozen_inputs/split_protocol.json')['alternative_profiles_require_author_decision']['phase4f2']
    assert adopted['fractions_exact'] == profile['fractions_exact'] == ['3/5','1/5','1/5']
    assert adopted['class_object_floors'] == profile['minimum_distinct_objects_per_class'] == {'train':21,'validation':6,'test':10}
    assert adopted['split_seed'] == profile['split_seed'] == 3908274
    assert adopted['primary_representation'] == 'A' and adopted['mandatory_image_control'] == 'B'
    units = [{'unit_id': u['split_unit_id'], 'weights_by_class': u['physical_object_counts_by_class']}
             for u in group['split_units_full_universe'] if u['eligible_candidate_ids']]
    assert len(units) == 358 and len(group['primary_eligibility_candidates']) == 370
    expected = 3 ** sum(sum(u['weights_by_class'].values()) > 1 for u in units)
    assert expected == 177147
    inputs = {'units': units, 'fractions_exact': adopted['fractions_exact'], 'floors_train_validation_test':[21,6,10],
              'split_seed':3908274, 'maximum_enumerations':2000000, 'expected_coupled_enumerations':expected,
              'source_algorithm_path':str(source.relative_to(ROOT)), 'source_algorithm_sha256':sha(source),
              'source_F_group_inventory_sha256':sha(OUT/'frozen_inputs/object_group_inventory.json'),
              'author_resolution_sha256':sha(OUT/'author_resolution.json')}
    inputs['canonical_input_sha256'] = hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    save('optimizer_input.json',inputs)
    save('optimizer_invocation.json',{'started_utc':now(),'optimizer_input_sha256':sha(OUT/'optimizer_input.json'),
                                      'canonical_input_sha256':inputs['canonical_input_sha256'],
                                      'source_algorithm_sha256':sha(source),'invocation_number':1,
                                      'implementation_modified_or_reimplemented':False,
                                      'seed_search_or_profile_mixing':False,'optimization_uses_model_or_quality_outcomes':False})
    spec = importlib.util.spec_from_file_location('frozen_dataset_v2f_split_algorithm',source)
    algorithm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(algorithm)
    begin = time.monotonic()
    stopped = threading.Event()
    def progress():
        while not stopped.wait(30):
            state = {'utc':now(),'stage':'unchanged_frozen_exact_optimizer_running','elapsed_seconds':round(time.monotonic()-begin,1),
                     'CPU_seconds':round(resource.getrusage(resource.RUSAGE_SELF).ru_utime,1),'expected_assignments_in_frozen_enumeration':expected}
            with (OUT/'optimization_progress.jsonl').open('a') as stream:
                stream.write(json.dumps(state,sort_keys=True)+'\n')
            print(json.dumps(state),flush=True)
    monitor = threading.Thread(target=progress,daemon=True)
    monitor.start()
    try:
        result = algorithm.optimize(units,[Fraction(x) for x in adopted['fractions_exact']],(21,6,10),3908274,max_enumerations=2000000)
    except BaseException as exc:
        stopped.set();monitor.join()
        save('optimizer_failure.json',{'failed_utc':now(),'error':type(exc).__name__+': '+str(exc),
                                       'elapsed_seconds':time.monotonic()-begin,'automatic_retry':False})
        raise
    stopped.set();monitor.join()
    assert len(result['assignments']) == 358 and result['enumerations'] == expected
    save('optimizer_result.json',result)
    save('optimizer_completion.json',{'completed_utc':now(),'elapsed_seconds':time.monotonic()-begin,
                                      'CPU_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                                      'maximum_RSS_KiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                      'result_sha256':sha(OUT/'optimizer_result.json'),'source_algorithm_sha256':sha(source),
                                      'source_unchanged_after_execution':sha(source)==inputs['source_algorithm_sha256'],
                                      'optimizer_invocations':1,'model_training_or_inference':False})
    print(json.dumps({'completed':True,'optimizer_invocations':1,'class_counts':result['counts'],
                      'exact_objective':result['objective_exact'],'elapsed_seconds':time.monotonic()-begin}),flush=True)


if __name__=='__main__':
    sys.dont_write_bytecode=True
    def boundary(event,args):
        if event.startswith('socket.') or event=='urllib.Request':raise RuntimeError('Dataset-v2G is offline')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
            if write and not path.is_relative_to(OUT):raise RuntimeError('Write outside Dataset-v2G')
            if 'baseline_comparison_v1' in path.parts or path.suffix.lower() in {'.pt','.pth','.h5','.hdf5','.safetensors'}:raise RuntimeError('No model/checkpoint/historical-outcome access')
    sys.addaudithook(boundary)
    main()
