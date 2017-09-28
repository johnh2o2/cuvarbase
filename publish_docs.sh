set -x

DOC_BRANCH=devel
NEEDED="cuvarbase docs/Makefile docs/source README.rst INSTALL.rst CHANGELOG.rst"

# We need to grab hidden files with mv...
shopt -s dotglob nullglob


HAS_GH_BRANCH=`git branch | grep gh-pages`
if [ "$HAS_GH_BRANCH" == "" ]; then
    echo "Did not detect gh-pages branch. Creating now."
    git checkout -b gh-pages || exit 1
else 
    git checkout gh-pages || exit 1
fi

# update
git pull origin gh-pages

# clean out
git rm -rf .

# checkout the files we need for the documentation
git checkout $DOC_BRANCH $NEEDED
git reset HEAD

# make docs
cd docs
make html || exit 1
cd ..

# move content to parent directory
mv docs/build/html/* ./

# remove unneeded files
rm -rf $NEEDED docs

# update the repo
git add --all
git commit -m "Updating docs"
git push -u origin gh-pages

git checkout $DOC_BRANCH
