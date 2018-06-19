#!/usr/bin/env bash

set -x

cwd=$(pwd)
cd sphinx && make html
cd $cwd
if [ ! -d "docs" ]; then
    mkdir docs
fi
cp -r sphinx/build/html/. docs
