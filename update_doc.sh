#!/usr/bin/env bash

set -x

# caution: this will fail if spaces in path
cwd=$(pwd)
cd sphinx && make clean && make html
cd $cwd
if [ ! -d "docs" ]; then
    mkdir docs
fi
cp -r sphinx/build/html/. docs
