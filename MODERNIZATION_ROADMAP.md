# cuvarbase Modernization Roadmap

This document outlines concrete steps to modernize cuvarbase while maintaining its PyCUDA foundation. These improvements address compatibility, maintainability, and user experience without requiring a risky framework migration.

## Phase 1: Python Version Support (Priority: HIGH)

### Objective
Update Python version support to drop legacy Python 2.7 and add support for modern Python versions.

### Actions

1. **Drop Python 2.7 Support**
   - Remove `future` package dependency
   - Remove `from __future__ import` statements
   - Update setup.py classifiers
   - Clean up Python 2/3 compatibility code

2. **Add Modern Python Support**
   - Test with Python 3.7, 3.8, 3.9, 3.10, 3.11
   - Update CI to test multiple Python versions
   - Update installation documentation

3. **Code Modernization**
   - Use f-strings instead of .format()
   - Add type hints to public APIs
   - Use pathlib for path operations
   - Leverage modern dictionary features

**Estimated Effort**: 2-3 weeks  
**Breaking Changes**: Yes (drops Python 2.7)  
**Benefits**: Cleaner code, better IDE support, easier maintenance

## Phase 2: Dependency and Version Management (Priority: HIGH)

### Objective
Resolve version pinning issues and improve dependency management.

### Actions

1. **Investigate PyCUDA 2024.1.2 Issue**
   - Document the specific issue with this version
   - Test with latest PyCUDA versions
   - Update version constraints based on findings

2. **CUDA Version Testing**
   - Test with CUDA 11.x series
   - Test with CUDA 12.x series
   - Create compatibility matrix

3. **Create pyproject.toml**
   ```toml
   [build-system]
   requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
   
   [project]
   name = "cuvarbase"
   dynamic = ["version"]
   dependencies = [
       "numpy>=1.17",
       "scipy>=1.3",
       "pycuda>=2021.1",
       "scikit-cuda>=0.5.3",
   ]
   requires-python = ">=3.7"
   ```

4. **Dependency Audit**
   - Update NumPy minimum version (1.6 is very old)
   - Update SciPy minimum version
   - Consider removing scikit-cuda for direct cuFFT usage

**Estimated Effort**: 2-4 weeks  
**Breaking Changes**: Minor (version requirements)  
**Benefits**: Better compatibility, easier installation

## Phase 3: Installation and Documentation (Priority: HIGH)

### Objective
Simplify installation and improve user experience.

### Actions

1. **Docker Support**
   Create Dockerfile:
   ```dockerfile
   FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
   RUN apt-get update && apt-get install -y python3 python3-pip
   RUN pip3 install cuvarbase
   ```

2. **Conda Package**
   - Create conda-forge recipe
   - Enables: `conda install -c conda-forge cuvarbase`
   - Handles CUDA dependencies automatically

3. **Installation Documentation**
   - Platform-specific quick-start guides
   - Troubleshooting common issues
   - Video tutorial for first-time users
   - Pre-built binary wheels for pip (if possible)

4. **Example Notebooks**
   - Update existing notebooks to Python 3
   - Add Google Colab compatibility
   - Create "getting started" notebook

**Estimated Effort**: 3-4 weeks  
**Breaking Changes**: None  
**Benefits**: Easier onboarding, fewer support requests

## Phase 4: Testing and CI/CD (Priority: MEDIUM)

### Objective
Improve code quality and catch regressions early.

### Actions

1. **GitHub Actions CI**
   ```yaml
   name: Tests
   on: [push, pull_request]
   jobs:
     test:
       strategy:
         matrix:
           python-version: [3.7, 3.8, 3.9, 3.10, 3.11]
           cuda-version: [11.8, 12.0]
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Install dependencies
         - name: Run tests
   ```

2. **Expand Test Coverage**
   - Add tests for edge cases
   - Add performance benchmarks
   - Add regression tests

3. **Code Quality Tools**
   - Add black for formatting
   - Add ruff/flake8 for linting
   - Add mypy for type checking

4. **Documentation Build**
   - Automate Sphinx documentation builds
   - Deploy documentation on commits to main

**Estimated Effort**: 3-4 weeks  
**Breaking Changes**: None  
**Benefits**: Catch bugs early, maintain quality

## Phase 5: Optional CPU Fallback (Priority: LOW)

### Objective
Add CPU-based implementations for systems without CUDA.

### Actions

1. **Numba Integration**
   ```python
   # cuvarbase/cpu_fallback.py
   import numba
   
   @numba.jit
   def lombscargle_cpu(t, y, freqs):
       # CPU implementation
       pass
   ```

2. **Automatic Fallback**
   ```python
   # cuvarbase/__init__.py
   try:
       import pycuda.driver as cuda
       GPU_AVAILABLE = True
   except ImportError:
       GPU_AVAILABLE = False
       warnings.warn("CUDA not available, using CPU fallback")
   ```

3. **Selective Implementation**
   - Start with Lomb-Scargle (most commonly used)
   - Add BLS as second priority
   - Other algorithms as needed

**Estimated Effort**: 6-8 weeks (per algorithm)  
**Breaking Changes**: None  
**Benefits**: Broader accessibility, easier development/debugging

## Phase 6: Performance Optimization (Priority: LOW)

### Objective
Improve performance without changing the framework.

### Actions

1. **Profile Current Performance**
   - Identify bottlenecks
   - Measure kernel execution times
   - Analyze memory transfer patterns

2. **Kernel Optimization**
   - Review for newer CUDA features
   - Optimize memory access patterns
   - Improve occupancy

3. **Multi-GPU Support**
   - Add automatic GPU detection
   - Load balancing across GPUs
   - Unified interface

**Estimated Effort**: 8-12 weeks  
**Breaking Changes**: None  
**Benefits**: Better performance, multi-GPU utilization

## Phase 7: API Improvements (Priority: LOW)

### Objective
Modernize the API while maintaining backward compatibility.

### Actions

1. **Consistent API**
   - Standardize parameter names
   - Consistent return types
   - Better error messages

2. **Context Managers**
   ```python
   with cuvarbase.GPU() as gpu:
       results = gpu.lombscargle(t, y, freqs)
   ```

3. **Batch Processing API**
   ```python
   # Process multiple light curves
   results = cuvarbase.batch_process(
       lightcurves,
       method='lombscargle',
       freqs=freqs
   )
   ```

**Estimated Effort**: 4-6 weeks  
**Breaking Changes**: None (add alongside existing)  
**Benefits**: Better user experience, more pythonic

## Implementation Timeline

### Year 1 (Immediate)
- Q1: Phase 1 (Python version support)
- Q2: Phase 2 (Dependency management)
- Q3: Phase 3 (Installation/documentation)
- Q4: Phase 4 (Testing/CI)

### Year 2 (Future)
- Q1-Q2: Phase 5 (CPU fallback - if resources available)
- Q3-Q4: Phase 6 (Performance optimization - if resources available)

### Year 3+ (Optional)
- Phase 7 (API improvements - community-driven)

## Resource Requirements

### Minimum Viable Improvements (Phases 1-3)
- **Developer Time**: 1 person, 2-3 months
- **Infrastructure**: GitHub Actions (free), Read the Docs (free)
- **Budget**: $0

### Full Roadmap (Phases 1-7)
- **Developer Time**: 1-2 people, 6-12 months
- **Infrastructure**: Same as above
- **Budget**: $0 (volunteer) or $50k-100k (paid development)

## Success Metrics

### Technical Metrics
- [ ] Support Python 3.7-3.11
- [ ] Zero known compatibility issues with latest PyCUDA
- [ ] Test coverage > 80%
- [ ] Documentation coverage = 100% of public API
- [ ] Installation success rate > 95% (from user surveys)

### Community Metrics
- [ ] Reduce installation-related issues by 50%
- [ ] Increase GitHub stars by 25%
- [ ] Active community contributions (PRs, issues)
- [ ] Positive user feedback

## Risk Mitigation

### Risk: Breaking Existing User Code
**Mitigation**: 
- Maintain backward compatibility where possible
- Provide deprecation warnings for 1 year before removal
- Document migration path for breaking changes
- Semantic versioning (major.minor.patch)

### Risk: Resource Constraints
**Mitigation**:
- Prioritize high-impact, low-effort improvements
- Seek community contributions
- Apply for NumFOCUS or similar grants
- Incremental progress is acceptable

### Risk: CUDA/PyCUDA Ecosystem Changes
**Mitigation**:
- Monitor PyCUDA development
- Maintain communication with PyCUDA maintainers
- Have contingency plan for framework change (this document)
- Regular testing with new versions

## Community Involvement

### How to Contribute
1. **Code Contributions**: Pull requests welcome
2. **Testing**: Test on different platforms
3. **Documentation**: Improve docs and examples
4. **Funding**: Sponsor development via GitHub Sponsors

### Maintainer Responsibilities
- Review PRs within 2 weeks
- Monthly status updates
- Clear contributor guidelines
- Responsive to security issues

## Alternative Scenarios

### If PyCUDA Becomes Unmaintained
- Revisit TECHNOLOGY_ASSESSMENT.md recommendations
- Consider CuPy as primary alternative
- Budget 6-12 months for migration
- Maintain PyCUDA version as legacy branch

### If Major Algorithm Redesign Needed
- Consider modern frameworks at design stage
- Prototype with multiple frameworks
- Choose based on performance data
- Learn from this migration experience

## Conclusion

This roadmap provides a practical path forward that:
1. **Improves user experience** without risky migrations
2. **Modernizes the codebase** while preserving core assets
3. **Maintains scientific rigor** and performance
4. **Enables future growth** with optional enhancements

The key insight: **incremental improvements beat risky rewrites**.

---

**Document Version**: 1.0  
**Date**: 2025-10-14  
**Last Updated**: 2025-10-14  
**Status**: Draft - Ready for Review
