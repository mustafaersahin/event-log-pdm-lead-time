#!/bin/bash
# Downloads the Microsoft Azure predictive maintenance dataset into data/ and checks the MD5 sums.
set -e
cd "$(dirname "$0")/.."; mkdir -p data
BASE=https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data
for f in PdM_errors PdM_failures PdM_machines PdM_maint PdM_telemetry; do
  [ -f data/$f.csv ] || curl -fsSL -o data/$f.csv $BASE/$f.csv
done
md5sum -c code/data.md5
