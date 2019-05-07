#!/bin/bash

PYTHON_SCRIPT=extract_knn.py

EPSILON=(1 5 10)
KNN=(1 5 10)
N_SAMPLES=(100 250 500 1000 2000 2500 3000 4000)

TEST_SET_FILENAME=data/test.csv
OUTPUT_ROOT_FILENAME=out/results-seq


echo "********************************* Running all the experiments sequentially *********************************"

for e in "${EPSILON[@]}"
do
    TRANSFORMATIONS_FILENAME=out/transformations-eps_${e}.gz
    INDICES=("" indices/kd-tree-eps_${e}.idx.gz indices/ball-tree-eps_${e}.idx.gz)

    for k in "${KNN[@]}"
    do
        for n in "${N_SAMPLES[@]}"
        do
            for i in "${INDICES[@]}"
            do
                echo ""
                echo "./${PYTHON_SCRIPT} "${TEST_SET_FILENAME}" "${TRANSFORMATIONS_FILENAME}" "${OUTPUT_ROOT_FILENAME}" -n "${n}" -k "${k}" -i '"${i}"'"
                echo ""
                ./${PYTHON_SCRIPT} "${TEST_SET_FILENAME}" "${TRANSFORMATIONS_FILENAME}" "${OUTPUT_ROOT_FILENAME}" -n "${n}" -k "${k}" -i "${i}"
                echo ""
            done
        done
    done
done

echo "Eventually, merging all the results into a single file ${OUTPUT_ROOT_FILENAME}.csv"
echo ""
echo "cat ${OUTPUT_ROOT_FILENAME}-*.csv | awk -F',' '{if(NR == 1 || NR % 2 == 0){print}}' > ${OUTPUT_ROOT_FILENAME}.csv"
cat ${OUTPUT_ROOT_FILENAME}-*.csv | awk -F',' '{if(NR == 1 || NR % 2 == 0){print}}' > ${OUTPUT_ROOT_FILENAME}.csv
echo ""
echo "Removing all the intermediate result files ..."
echo ""
echo "rm ${OUTPUT_ROOT_FILENAME}-*.csv"
rm ${OUTPUT_ROOT_FILENAME}-*.csv
echo ""
echo "************************************************************************************************************"