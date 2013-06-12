# File: generateInput.sh
# Author: Sammy El Ghazzal
# Date: Started 05/07/2013
# Purpose: Script that will run cudaccl on an input file and produces 
#          the corresponding labeled image. 
# Usage: ./processCuda inputFile [outputImgName = result]

#!/bin/sh

if [ $# -lt 2 ]; then
  output="result"
else
  output="$2"
fi
./cudaccl $1 out 
python createimg.py -i out -o "$output"
#rm out 
