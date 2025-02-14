#!/usr/bin/env bash

# Copyright 2018 Amazon.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to 
# deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.

if [ $# -ne 1 ]
then
    echo "Usage: $0 inputFile"
    exit 1
fi

inputFile=$1
totalCount=$(wc -l "$inputFile" | cut -d" " -f1)
trainCount=$[ ($totalCount * 90 / 100) + 1 ]
testCount=$[ $totalCount - $trainCount ]
shufTmp="$inputFile.shuf"

rm -f $shufTmp
shuf $inputFile > $shufTmp
split -l $trainCount -a 1 $shufTmp "$inputFile.SPLIT"
mv "$inputFile.SPLITa" "$inputFile.train"
mv "$inputFile.SPLITb" "$inputFile.test"
rm -f $shufTmp