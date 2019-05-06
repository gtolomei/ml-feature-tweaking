#!/bin/bash

PARALLEL=4

if [ $# -eq 0 ]
  then
    echo "Error! No input argument has been supplied!"
    echo ""
    echo "Usage:"
    echo "> ./extract_knn.sh EPSILON"
    echo ""
    echo "where:"
    echo "EPSILON is the value used for selecting transformations and spatial index (if any)"
    exit 1
fi

if [ -z "$1" ]
  then
    echo "Error! Empty-string argument has been supplied!"
    echo ""
    echo "Usage:"
    echo "> ./extract_knn.sh EPSILON"
    echo ""
    echo "where:"
    echo "EPSILON is the value used for selecting transformations and spatial index (if any)"
    exit 1
fi

EPSILON=$1
PYTHON_SCRIPT=extract_knn.py
PYTHON_SCRIPTNAME=${PYTHON_SCRIPT%.py}

TEST_SET_FILENAME=data/test.csv
TRANSFORMATIONS_FILENAMES=out/transformations-eps_${EPSILON}.gz
INDICES=("" indices/kd-tree-eps_${EPSILON}.idx.gz indices/ball-tree-eps_${EPSILON}.idx.gz)
OUTPUT_ROOT_FILENAME=out/results

KNN=(1 5 10)
N_SAMPLES=(100 250 500 1000 2000 2500 3000 4000)

echo "*********** Extracting k-NN [epsilon = ${EPSILON}] ***********"
echo ""

echo "parallel --eta --bar -j ${PARALLEL} --joblog /tmp/${PYTHON_SCRIPTNAME}-eps-${EPSILON}.log ./${PYTHON_SCRIPT} {1} {2} {3} -n {4} -k {5} -i {6} ::: "${TEST_SET_FILENAME}" ::: "${TRANSFORMATIONS_FILENAMES[@]}" ::: "${OUTPUT_ROOT_FILENAME}" ::: "${N_SAMPLES[@]}" ::: "${KNN[@]}" ::: "${INDICES[@]}""
echo ""
parallel --eta --bar -j ${PARALLEL} --joblog /tmp/${PYTHON_SCRIPTNAME}-eps-${EPSILON}.log ./${PYTHON_SCRIPT} {1} {2} {3} -n {4} -k {5} -i {6} ::: "${TEST_SET_FILENAME}" ::: "${TRANSFORMATIONS_FILENAMES[@]}" ::: "${OUTPUT_ROOT_FILENAME}" ::: "${N_SAMPLES[@]}" ::: "${KNN[@]}" ::: "${INDICES[@]}"
echo ""
echo "**************************************************************"