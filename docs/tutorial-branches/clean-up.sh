#!/bin/bash
                           
find . -path "*/infra/helm/README" -delete
find . -path "*/docs/whitepaper" -delete
find . -path "*/docs/example-chat-queries.txt" -delete

for TESTDIR in `find . -path "*/tests"`; do 
  echo "Removing ${TESTDIR}"
  rm -rf ${TESTDIR}
done

for TESTDIR in `find . -path "*/docs/whitepaper"`; do 
  echo "Removing ${TESTDIR}"
  rm -rf ${TESTDIR}
done


find . -type file -name "*-old.py" -delete
find . -type file -name "*.pdf" -delete
find . -type file -name "*.tex" -delete
