#!/bin/zsh
cd -- "$(dirname -- "$0")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print 'Set up the project virtual environment first; see README.md.'
  read '?Press Return to close.'
  exit 1
fi
if ! .venv/bin/python -m traffic_intelligence share; then
  read '?Press Return to close.'
  exit 1
fi
