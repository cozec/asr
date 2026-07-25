#!/usr/bin/env bash
# Download + verify + extract the full LibriSpeech corpus (960h train + dev/test).
# Resumable: re-run it and curl -C - picks up where it left off; verified archives are skipped.
set -uo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DL_DIR="$DATA_DIR/downloads"
MIRROR="${MIRROR:-https://openslr.trmal.net/resources/12}"

# The ASR sets only -- skips original-mp3.tar.gz (87G) and the book/metadata extras.
PARTS=(
  dev-clean.tar.gz
  dev-other.tar.gz
  test-clean.tar.gz
  test-other.tar.gz
  train-clean-100.tar.gz
  train-clean-360.tar.gz
  train-other-500.tar.gz
)

mkdir -p "$DL_DIR"
cd "$DL_DIR" || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "mirror: $MIRROR"
log "target: $DATA_DIR/LibriSpeech"

# md5sum.txt covers every archive on the mirror; used to verify each download.
if [[ ! -s md5sum.txt ]]; then
  log "fetching md5sum.txt"
  curl -fsSL -o md5sum.txt "$MIRROR/md5sum.txt" || log "WARN: could not fetch md5sum.txt (will skip verification)"
fi

expected_md5() {
  [[ -s md5sum.txt ]] || return 1
  awk -v f="$1" '$2 == f || $2 == "./"f {print $1; found=1} END {exit !found}' md5sum.txt
}

verify() {
  local file="$1" want
  want="$(expected_md5 "$file")" || { log "  no md5 on record for $file -- accepting"; return 0; }
  local got
  got="$(md5 -q "$file" 2>/dev/null || md5sum "$file" | cut -d' ' -f1)"
  [[ "$got" == "$want" ]]
}

FAILED=()
for part in "${PARTS[@]}"; do
  marker="$DL_DIR/.done-$part"
  if [[ -f "$marker" ]]; then
    log "SKIP $part (already downloaded + extracted)"
    continue
  fi

  # Up to 3 attempts: transient resets on the 23G/30G archives are common.
  ok=0
  for attempt in 1 2 3; do
    log "GET  $part (attempt $attempt)"
    curl -fL --retry 5 --retry-delay 10 --retry-all-errors \
         -C - -o "$part" "$MIRROR/$part"
    rc=$?
    # rc 33 = server refused a byte-range resume on an already-complete file.
    if [[ $rc -ne 0 && $rc -ne 33 ]]; then
      log "  curl exit $rc -- retrying"
      sleep 15
      continue
    fi
    log "  verifying md5 of $part"
    if verify "$part"; then ok=1; break; fi
    log "  md5 MISMATCH -- discarding and re-downloading"
    rm -f "$part"
  done

  if [[ $ok -ne 1 ]]; then
    log "FAIL $part after 3 attempts"
    FAILED+=("$part")
    continue
  fi

  log "  extracting $part"
  # Every archive unpacks into a top-level LibriSpeech/ dir, so they merge cleanly.
  if tar -xzf "$part" -C "$DATA_DIR"; then
    touch "$marker"
    log "DONE $part"
  else
    log "FAIL extracting $part"
    FAILED+=("$part")
  fi
done

log "----------------------------------------"
if ((${#FAILED[@]})); then
  log "COMPLETED WITH FAILURES: ${FAILED[*]}"
  log "re-run this script to retry just those."
  exit 1
fi

log "ALL PARTS OK"
du -sh "$DATA_DIR/LibriSpeech" 2>/dev/null
find "$DATA_DIR/LibriSpeech" -maxdepth 1 -mindepth 1 -type d | sort
log "flac files: $(find "$DATA_DIR/LibriSpeech" -name '*.flac' | wc -l | tr -d ' ')"
