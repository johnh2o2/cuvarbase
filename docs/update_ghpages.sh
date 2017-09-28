#!/usr/bin/env bash

# build the docs
make clean
make html
cd ..

# commit and push
git add --all
git commit -m "building and pushing docs"
git push

# switch branches and pull the data we want
git checkout gh-pages
rm -rf .
touch .nojekyll
git checkout master docs/build/html
mv ./docs/build/html/* ./
rm -rf ./docs
git add --all
#git commit -m "publishing updated docs..."
#git push origin gh-pages

# switch back
#git checkout master
