#!/bin/bash

[[ $# -eq 0 ]] && echo "$0 results ..." && exit 0
results=$@

python ~/tools/roc_compare_new.py person_upper,person_face $results
