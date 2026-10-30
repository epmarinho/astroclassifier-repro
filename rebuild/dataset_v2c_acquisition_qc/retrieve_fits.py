"""Exact predeclared DSS2 requests; bounded, throttled and immutable resume ledger."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PLAN = json.loads((OUT / 'retrieval_plan.json').read_text())
REQUESTS = [r for r in PLAN['requests'] if r['acquisition_origin'] == 'new_v2c']
URLS = {r['url'] for r in REQUESTS}
assert len(REQUESTS) == len(URLS) == 1359
assert hashlib.sha256((OUT / 'acquisition_protocol.json').read_bytes()).hexdigest() == PLAN['protocol_sha256']
assert hashlib.sha256((OUT / 'candidate_geometry.json').read_bytes()).hexdigest() == PLAN['geometry_sha256']
STOP = threading.Event()
START_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
LAST_START = 0.0


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def retrieve(index, row):
    global LAST_START
    meta = OUT / 'request_records' / (row['candidate_id'] + '_' + row['band'] + '.json')
    if meta.exists():
        old = json.loads(meta.read_text())
        assert old['url'] == row['url']
        if old.get('response_path'):
            assert sha(OUT / old['response_path']) == old['response_sha256']
        return {'cached': True, 'retained': bool(old.get('fits_master_retained'))}
    with START_LOCK:
        if STOP.is_set():
            return {'stopped': True}
        delay = max(0, 1 - (time.monotonic() - LAST_START))
        if delay:
            time.sleep(delay)
        if STOP.is_set():
            return {'stopped': True}
        LAST_START = time.monotonic()
    begin = time.monotonic()
    record = {**row, 'request_index': index, 'started_utc': now(), 'method': 'GET',
              'http_status': None, 'response_path': None, 'response_sha256': None,
              'size_bytes': None, 'request_headers': {
                  'User-Agent': 'Astroclassifier-Dataset-v2C-fixed-473-object-research-QC/1.0',
                  'Accept': 'application/fits,application/octet-stream'},
              'redirects_followed': 0, 'automatic_retries': 0, 'error': None}
    response = None
    data = None
    try:
        opener = urllib.request.build_opener(NoRedirect())
        response = opener.open(urllib.request.Request(row['url'], headers=record['request_headers']), timeout=60)
    except urllib.error.HTTPError as exc:
        response = exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
    if response is not None:
        with response:
            record.update(http_status=response.status, response_headers=list(response.headers.items()),
                          content_type=response.headers.get('Content-Type', ''))
            try:
                data = response.read(20000001)
            except (TimeoutError, OSError) as exc:
                record['error'] = 'read_failed: ' + str(exc)
            if record['http_status'] in {301, 302, 303, 307, 308}:
                record['location_not_followed'] = response.headers.get('Location')
    if data is not None:
        fits = record['http_status'] == 200 and data.startswith(b'SIMPLE  =') and len(data) <= 20000000
        path = OUT / row['output'] if fits else OUT / 'failed_responses' / (meta.stem + '.response.bin')
        with path.open('xb') as stream:
            stream.write(data)
        path.chmod(0o444)
        record.update(response_path=str(path.relative_to(OUT)), response_sha256=hashlib.sha256(data).hexdigest(),
                      size_bytes=len(data), fits_master_retained=fits)
        if len(data) > 20000000:
            record['error'] = 'response_exceeds_frozen_limit'
    record.update(finished_utc=now(), elapsed_seconds=time.monotonic() - begin)
    with meta.open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write('\n')
    meta.chmod(0o444)
    with LOG_LOCK:
        with (OUT / 'network_request_log.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    if record['http_status'] in {401, 403, 429}:
        STOP.set()
    return {'cached': False, 'retained': bool(record.get('fits_master_retained')),
            'http_status': record['http_status'], 'error': record['error']}


def main():
    for name in ['fits', 'request_records', 'failed_responses']:
        (OUT / name).mkdir(exist_ok=True)
    done = cached = retained = 0
    begin = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(retrieve, i, r) for i, r in enumerate(REQUESTS, 1)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            done += 1
            cached += result.get('cached', False)
            retained += result.get('retained', False)
            if done % 15 == 0 or (not result.get('retained') and not result.get('stopped')):
                print(json.dumps({'completed_new_slots': done, 'total': 1359, 'cached': cached,
                                  'retained_new_FITS': retained, 'elapsed_seconds': round(time.monotonic() - begin, 1),
                                  'last': result}), flush=True)
    print(json.dumps({'finished': True, 'completed_slots': done, 'cached': cached,
                      'retained_new_FITS': retained, 'access_restriction': STOP.is_set()}), flush=True)
    if STOP.is_set():
        raise SystemExit('Access/rate restriction: stopped; no bypass')


if __name__ == '__main__':
    sys.dont_write_bytecode = True

    def boundary(event, args):
        if event == 'urllib.Request' and args[0] not in URLS:
            raise RuntimeError('URL outside the 1359 new frozen requests; no pilot redownload')
        if event.startswith('socket.') and '--offline-resume' in sys.argv:
            raise RuntimeError('Resume validation must perform zero network requests')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1:3]
            write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if write and not p.is_relative_to(OUT):
                raise RuntimeError('Write outside Dataset-v2C')
            if 'baseline_comparison_v1' in p.parts or p.suffix.lower() in {'.pt', '.pth', '.h5', '.hdf5', '.safetensors'}:
                raise RuntimeError('No v1/model/checkpoint access')

    sys.addaudithook(boundary)
    main()
