#!/bin/sh
set -e

export PYTHONPATH=/usr/local/lib/python3.12/site-packages/:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

echo "Generating Figure 2"
python3 ./Fig_2/generate_plots.py

echo "Generating Figure 4"
python3 ./Fig_4/generate_plots.py

echo "Generating Figure 7"
python3 ./Fig_7/generate_plots.py

echo "Generating Table 3"
python3 ./Tab_3/generate_table.py

echo "Generating Table 2"
python3 ./Tab_2/generate_table.py

echo "Generating Figure 6"
python3 ./Fig_6/generate_plots.py

echo "Complete"
echo "Outputs can be found in ./plot"
