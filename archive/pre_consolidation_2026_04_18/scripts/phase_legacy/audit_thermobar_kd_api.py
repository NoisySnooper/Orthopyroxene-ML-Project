"""Discover Thermobar's opx-liq equilibrium filter API."""
import inspect
import Thermobar as pt

print("Thermobar version:", pt.__version__)
print()
print("=" * 78)
print("STEP 1: callables matching equilibrium/Kd/Mgno/Fe_Mg patterns")
print("=" * 78)
keywords = ['kd', 'equilib', 'eq_test', 'eq_check', 'mgno', 'fe_mg']
matches = []
for name in sorted(dir(pt)):
    if name.startswith('_'):
        continue
    if any(k in name.lower() for k in keywords):
        obj = getattr(pt, name)
        if callable(obj):
            matches.append(name)
            print(f'\n{name}:')
            try:
                sig = inspect.signature(obj)
                print(f'  signature: {sig}')
            except (ValueError, TypeError) as e:
                print(f'  signature: (unavailable: {e})')
            try:
                doc = inspect.getdoc(obj) or '(no docstring)'
                print(f'  doc: {doc[:400]}')
            except Exception as e:
                print(f'  doc fail: {e}')

print()
print("=" * 78)
print("STEP 2: opx-specific equilibrium / Kd / test routines")
print("=" * 78)
opx_matches = []
for name in sorted(dir(pt)):
    if name.startswith('_'):
        continue
    if 'opx' in name.lower() and any(k in name.lower() for k in ['liq', 'eq', 'kd', 'test', 'filter', 'mgno', 'fe_mg']):
        obj = getattr(pt, name)
        if callable(obj):
            opx_matches.append(name)
            print(f'\n--- {name} ---')
            try:
                sig = inspect.signature(obj)
                print(f'  signature: {sig}')
            except (ValueError, TypeError):
                pass
            try:
                doc = inspect.getdoc(obj) or '(no docstring)'
                print(f'  doc: {doc[:600]}')
            except Exception:
                pass
print()
print("=" * 78)
print("STEP 3: source for top candidates")
print("=" * 78)
candidates = [m for m in matches + opx_matches
              if any(k in m.lower() for k in ['opx', 'mgno', 'eq_px', 'px_liq', 'kd_opx', 'kd_px'])]
candidates = sorted(set(candidates))
print('Candidates:', candidates)
for name in candidates[:10]:
    obj = getattr(pt, name)
    print(f'\n=== {name} ===')
    try:
        print(inspect.getsource(obj)[:3500])
    except (OSError, TypeError) as e:
        print(f'  getsource failed: {e}')
    print('---')
