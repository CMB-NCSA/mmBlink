#!/usr/bin/env bash

PRODUCT_DIR="$1"

if [ $PRODUCT_DIR = "." ]; then
    PRODUCT_DIR=$PWD
fi

echo "Adding: $PRODUCT_DIR to paths"
export SPT3G_DETECT_DIR=$PRODUCT_DIR
export PYTHONPATH=${PRODUCT_DIR}/python:${PYTHONPATH}
export PATH=${PRODUCT_DIR}/bin:${PATH}

# For other software (e.g. Lyrebird) that needs to link to this
export SPT3G_SOFTWARE_PATH=/Users/felipe/SPT-3G/macos/spt3g_software
export SPT3G_SOFTWARE_BUILD_PATH=/Users/felipe/SPT-3G/macos/spt3g_software/build

# Add binaries to PATH, relative to this script to allow files to be movable
export SPT3G_BUILD_ROOT=${SPT3G_SOFTWARE_BUILD_PATH}
export PATH=${SPT3G_BUILD_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${SPT3G_BUILD_ROOT}/spt3g:$LD_LIBRARY_PATH

# And python bits...
export PYTHONPATH=${SPT3G_BUILD_ROOT}:$PYTHONPATH
echo 'SPT3G Environment Variables Set'
