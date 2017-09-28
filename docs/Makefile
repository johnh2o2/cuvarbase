# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -E -a
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = cuvarbase
SOURCEDIR     = source
BUILDDIR      = build
GH_PAGES_SOURCES = source Makefile
DOC_BRANCH = devel

gh-pages:
	git checkout gh-pages
	rm -rf build _sources _static
	git checkout $(DOC_BRANCH) $(GH_PAGES_SOURCES)
	git reset HEAD
	make html
	mv -fv build/html/* ./
	rm -rf $(GH_PAGES_SOURCES) build
	git add -A
	git ci -m "Generated gh-pages for `git log $(DOC_BRANCH) -1 --pretty=short --abbrev-commit`" && git push origin gh-pages ; git checkout $(DOC_BRANCH)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
