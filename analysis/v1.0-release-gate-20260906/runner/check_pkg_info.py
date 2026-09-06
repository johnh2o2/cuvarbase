import tarfile
with tarfile.open('/workspace/cuvarbase/dist/cuvarbase-1.0.0.tar.gz') as tar:
    text=tar.extractfile('cuvarbase-1.0.0/PKG-INFO').read().decode()
assert 'Until v1.0.0' not in text and 'git+https' not in text
assert not any('0.2.5' in line and 'since 0.2.5' not in line for line in text.splitlines())
assert 'pip install cuvarbase' in text
assert '1,785 passed + 1 xfailed of 1,786 collected' in text
print('PKG-INFO: flipped README, measured count, no forbidden banner/git+/0.2.5 text')
