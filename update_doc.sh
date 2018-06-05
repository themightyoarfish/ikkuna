#!/usr/bin/env bash

cwd=$(pwd)
cd doc && make html
cd $cwd
if [ ! -d "docs" ]; then
    mkdir docs
fi
cp doc/build/html/*.{html,js} docs
find doc/build/html ! -name html -type d -exec cp -r {} docs \;
