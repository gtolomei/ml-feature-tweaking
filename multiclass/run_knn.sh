#!/bin/bash

PARALLEL=8

TEST_SET_FILENAME=data/test.csv
TRANSFORMATIONS_ROOT_FILENAME=out/transformations-eps_
KD_TREE_ROOT_FILENAME=indices/kd-tree-eps_
BALL_TREE_ROOT_FILENAME=indices/ball-tree-eps_
OUTPUT_ROOT_FILENAME=out/results

EPSILON=(1 5 10)
KNN=(1 2 5 10)
N_SAMPLES=(100 250 500 1000 2000 5000)

for e in ${EPSILON[@]}
do
    for k in ${KNN[@]}
    do  
        for n in ${N_SAMPLES[@]}
        do  
            PYTHON_SCRIPT="./extract_knn.py ${TEST_SET_FILENAME} ${TRANSFORMATIONS_ROOT_FILENAME}${e}.gz ${OUTPUT_ROOT_FILENAME} -n ${n} -k ${k}"
            PYTHON_SCRIPT=${PYTHON_SCRIPT}" & ./extract_knn.py ${TEST_SET_FILENAME} ${TRANSFORMATIONS_ROOT_FILENAME}${e}.gz ${OUTPUT_ROOT_FILENAME} -n ${n} -k ${k} -i ${KD_TREE_ROOT_FILENAME}${e}.idx.gz"
            PYTHON_SCRIPT=${PYTHON_SCRIPT}" & ./extract_knn.py ${TEST_SET_FILENAME} ${TRANSFORMATIONS_ROOT_FILENAME}${e}.gz ${OUTPUT_ROOT_FILENAME} -n ${n} -k ${k} -i ${BALL_TREE_ROOT_FILENAME}${e}.idx.gz" 
            echo "Executing $PYTHON_SCRIPT";
            sem -j ${PARALLEL} ${PYTHON_SCRIPT};
        done
    done
done

