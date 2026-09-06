from pathlib import Path
root=Path('/workspace/cuvarbase/docs/build/html')
images=sorted(p.relative_to(root).as_posix() for p in root.rglob('*.png'))
print('\n'.join(images))
# Seven directives: five execute GPU computations, two draw transit geometry.
expected=['bls_example','bls_example_transit','ce_example','bls_transit_diagram','planet_transit_diagram']
for stem in expected:
    assert any(Path(p).stem == stem for p in images), stem
lomb={Path(p).stem for p in images if Path(p).stem.startswith('lomb-')}
assert {'lomb-1', 'lomb-2'} <= lomb, lomb
assert not any(s in (root/'index.html').read_text() for s in ['Exception occurred in plotting'])
print('All five GPU figures plus two geometry diagrams rendered')
