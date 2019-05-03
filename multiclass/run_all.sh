#!/bin/bash

BASH_SCRIPT=extract_knn.sh
OUTPUT_ROOT_FILENAME=out/results
EPSILON=(1 5 10)

echo "********************************* Running all the experiments *********************************"

for e in ${EPSILON[@]}
do
    echo ""
    echo "./${BASH_SCRIPT} ${e}"
    echo ""
    ./${BASH_SCRIPT} ${e}
    echo ""
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
echo "***********************************************************************************************"