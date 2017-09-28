set -x

DOC_BRANCH=devel
NEEDED="cuvarbase docs/Makefile docs/source README.rst INSTALL.rst CHANGELOG.rst"

HAS_GH_BRANCH=`git branch | grep gh-pages`
if [ "$HAS_GH_BRANCH" == "" ]; then
    echo "Did not detect gh-pages branch. Creating now."
    git checkout -b gh-pages || exit 1
else 
    git checkout gh-pages || exit 1
fi

# update
git pull origin gh-pages
git rm -rf .

git checkout $DOC_BRANCH $NEEDED
git reset HEAD
cd docs
make html || exit 1
cd ..
mv docs/build/html/* ./
rm -rf $NEEDED docs

git add --all
git commit -m "Updating docs"
git push -u origin gh-pages

git checkout $DOC_BRANCH
