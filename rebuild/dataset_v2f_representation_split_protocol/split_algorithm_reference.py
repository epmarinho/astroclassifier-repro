"""Exact group-count optimizer; executable entry point uses synthetic cases only."""
from fractions import Fraction
import hashlib
import itertools

ROLES = ('train', 'validation', 'test')


def seed_order(seed, purpose, token):
    return hashlib.sha256(f'Dataset-v2F|{seed}|{purpose}|{token}'.encode()).hexdigest()


def singleton_allocation(total_objects, coupled_counts, free_count, fractions, floors):
    """Exact separable integer squared-error solution with lower bounds."""
    allocation = [max(0, floors[i] - coupled_counts[i]) for i in range(3)]
    if sum(allocation) > free_count:
        return None
    target = [fractions[i] * total_objects - coupled_counts[i] for i in range(3)]
    for _ in range(free_count - sum(allocation)):
        marginal = [2 * (allocation[i] - target[i]) + 1 for i in range(3)]
        # Equal costs favor higher train count, then validation, then test.
        chosen = min(range(3), key=lambda i: (marginal[i], i))
        allocation[chosen] += 1
    assert sum(allocation) == free_count
    return tuple(allocation)


def optimize(units, fractions, floors, seed, max_enumerations=2000000):
    """No seed search or floors relaxation; fail if exact enumeration exceeds cap."""
    assert sum(fractions) == 1 and all(x > 0 for x in fractions)
    assert all(isinstance(n, int) and n >= 0 for n in floors)
    assert len({u['unit_id'] for u in units}) == len(units)
    classes = sorted({cl for unit in units for cl, count in unit['weights_by_class'].items() if count})
    totals = {cl: sum(u['weights_by_class'].get(cl, 0) for u in units) for cl in classes}
    if any(totals[cl] < sum(floors) for cl in classes):
        raise ValueError('Object floor arithmetically infeasible; no automatic weakening')
    coupled = [u for u in units if sum(u['weights_by_class'].values()) > 1]
    coupled.sort(key=lambda u: (seed_order(seed, 'coupled', u['unit_id']), u['unit_id']))
    free = {cl: sorted([u for u in units if u['weights_by_class'] == {cl: 1}], key=lambda u: (seed_order(seed, 'singleton|' + cl, u['unit_id']), u['unit_id'])) for cl in classes}
    assert len(coupled) + sum(len(v) for v in free.values()) == len(units)
    combinations = 3 ** len(coupled)
    if combinations > max_enumerations:
        raise ValueError('Exact optimizer enumeration cap exceeded; no greedy fallback or new seed')
    role_order = sorted(range(3), key=lambda i: (seed_order(seed, 'role', ROLES[i]), i))
    best = None
    for assignment in itertools.product(role_order, repeat=len(coupled)):
        counts = {cl: [0, 0, 0] for cl in classes}
        for unit, role in zip(coupled, assignment):
            for cl, n in unit['weights_by_class'].items():
                counts[cl][role] += n
        allocations = {}
        objective = Fraction(0)
        for cl in classes:
            a = singleton_allocation(totals[cl], counts[cl], len(free[cl]), fractions, floors)
            if a is None:
                break
            allocations[cl] = a
            objective += sum((Fraction(counts[cl][i] + a[i], totals[cl]) - fractions[i]) ** 2 for i in range(3))
        else:
            # Enumeration order supplies a seeded deterministic tie breaker.
            if best is None or objective < best[0]:
                best = objective, assignment, allocations, counts
    if best is None:
        raise ValueError('No floor-compliant group assignment; stop for author review')
    objective, assignments, allocations, coupled_counts = best
    result = {u['unit_id']: ROLES[role] for u, role in zip(coupled, assignments)}
    final_counts = {}
    for cl in classes:
        offset = 0
        for i, n in enumerate(allocations[cl]):
            for unit in free[cl][offset:offset + n]:
                result[unit['unit_id']] = ROLES[i]
            offset += n
        final_counts[cl] = [coupled_counts[cl][i] + allocations[cl][i] for i in range(3)]
        assert offset == len(free[cl]) and all(final_counts[cl][i] >= floors[i] for i in range(3))
    assert len(result) == len(units)
    return {'assignments': result, 'counts': final_counts, 'objective_exact': str(objective), 'enumerations': combinations}


def toy_tests():
    fractions = [Fraction(3, 5), Fraction(1, 5), Fraction(1, 5)]
    # Independently compare the convex count allocator against brute-force counts.
    allocation_checks = 0
    for n in range(3, 13):
        for coupled in [(0, 0, 0), (1, 0, 0), (0, 1, 1)]:
            free = n - sum(coupled)
            if free < 0:
                continue
            actual = singleton_allocation(n, coupled, free, fractions, (1, 1, 1))
            candidates = []
            for train in range(free + 1):
                for val in range(free - train + 1):
                    test = free - train - val
                    counts = [coupled[0] + train, coupled[1] + val, coupled[2] + test]
                    if min(counts) < 1:
                        continue
                    cost = sum((Fraction(counts[i]) - n * fractions[i]) ** 2 for i in range(3))
                    candidates.append((cost, -train, -val, (train, val, test)))
            expected = min(candidates)[3] if candidates else None
            assert actual == expected
            allocation_checks += 1
    units = [{'unit_id': 'SYNTHETIC-cross-class', 'weights_by_class': {'A': 1, 'B': 1}}]
    units += [{'unit_id': f'SYNTHETIC-{cl}-{i}', 'weights_by_class': {cl: 1}} for cl in ['A', 'B'] for i in range(5)]
    result = optimize(units, fractions, (2, 1, 1), 12345)
    assert result == optimize(list(reversed(units)), fractions, (2, 1, 1), 12345)
    infeasible_stopped = False
    try:
        optimize(units, fractions, (20, 5, 10), 12345)
    except ValueError:
        infeasible_stopped = True
    assert infeasible_stopped
    return {'synthetic_only': True, 'primary_eligibility_assignments_generated': 0,
            'count_allocator_bruteforce_checks': allocation_checks,
            'cross_class_component_atomic': True, 'input_order_independence': True,
            'infeasible_floor_stops': True, 'toy_exact_objective': result['objective_exact']}


if __name__ == '__main__':
    import json
    print(json.dumps(toy_tests()))
