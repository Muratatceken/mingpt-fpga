#!/usr/bin/env bash
# Loghub 2k-line samples of 16 systems (He et al., arXiv 2008.06448; github.com/logpai/loghub) -> data/loghub/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/data/loghub"
mkdir -p "$D"
for s in Android Apache BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Windows Zookeeper; do
  [ -s "$D/${s}_2k.log" ] || curl -sSfL -o "$D/${s}_2k.log" "https://raw.githubusercontent.com/logpai/loghub/master/$s/${s}_2k.log"
done
ls -la "$D"
