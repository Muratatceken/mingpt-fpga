#!/usr/bin/env bash
# Background driver: every 10 minutes verify newly finished runs (pipeline.py); once no training is
# running any more, run the final steps (finalize.sh) and stop. One process -> no concurrent writers.
# The [.] in the patterns keeps pgrep from matching this script's own command line.
cd "$(dirname "$0")"
LOG=../../model/zoo/runs
while true; do
  python3 pipeline.py 2>&1 | grep -v -i warn >> $LOG/pipeline.log
  if ! pgrep -f "train[.]py --tag" > /dev/null && ! pgrep -f "experiments[.]py" > /dev/null; then
    ./finalize.sh > $LOG/finalize.log 2>&1
    echo "finalize exit $?" >> $LOG/finalize.log
    break
  fi
  sleep 600
done
