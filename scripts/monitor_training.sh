#!/usr/bin/env bash
# Monitor Pi0.5 and ACT training jobs.
# Resubmits automatically on failure, adding the failed node to the exclusion
# list each time. Always uses GPU partitions — never falls back to CPU.
#
# Usage:
#   nohup bash scripts/monitor_training.sh > logs/slurm/monitor.console.log 2>&1 &
#
#   # Adopt already-submitted jobs:
#   bash scripts/monitor_training.sh --pi05-job 12539 --act-job 12716
#
# Env overrides:
#   EXCLUDED_NODES      comma-separated node list to always skip
#   PI05_PARTITION      SLURM partition for Pi0.5  (default: agent-xlong-15)
#   ACT_PARTITION       SLURM partition for ACT    (default: agent-long-15)
#   PI05_STEPS          training steps for Pi0.5   (default: 10000)
#   ACT_STEPS           training steps for ACT     (default: 100000)
#   POLL_INTERVAL       seconds between checks     (default: 60)
#   SKIP_PI05=1 / SKIP_ACT=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/slurm"
MONITOR_LOG="$LOG_DIR/monitor_training.log"
mkdir -p "$LOG_DIR"

# ── config ─────────────────────────────────────────────────────────────────────
# Default to 507-15 partitions — 507-11..14 all have broken CUDA drivers.
# The -15 partitions only contain 507-15, so --exclude is not strictly needed
# but kept for safety in case the cluster adds more nodes later.
PI05_PARTITION="${PI05_PARTITION:-agent-xlong-15}"
ACT_PARTITION="${ACT_PARTITION:-agent-long-15}"
EXCLUDED_NODES="${EXCLUDED_NODES:-507-11,507-12,507-13,507-14}"
PI05_STEPS="${PI05_STEPS:-10000}"
ACT_STEPS="${ACT_STEPS:-100000}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
SKIP_PI05="${SKIP_PI05:-0}"
SKIP_ACT="${SKIP_ACT:-0}"

PI05_JOB_ID=""
ACT_JOB_ID=""
PI05_DONE=0
ACT_DONE=0

# ── logging ────────────────────────────────────────────────────────────────────
# IMPORTANT: log() writes to STDERR (not stdout).
# All submit functions are called inside $(...) which captures stdout only.
# Keeping log output on stderr prevents log lines from polluting job ID variables.
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" >&2
    echo "$msg" >> "$MONITOR_LOG"
}

# ── SLURM helpers ──────────────────────────────────────────────────────────────
queue_count() {
    squeue --me -h 2>/dev/null | wc -l || echo 99
}

job_state() {
    local jid="$1"
    local state
    state=$(squeue -j "$jid" -h --format="%T" 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "$state" ]]; then echo "$state"; return; fi
    sacct -j "$jid" -n --format=State -X 2>/dev/null | head -1 | tr -d ' '
}

job_node() {
    local jid="$1"
    sacct -j "$jid" -n --format=NodeList -X 2>/dev/null | head -1 | tr -d ' '
}

wait_for_slot() {
    local count
    while true; do
        count=$(queue_count)
        if [[ "$count" -lt 6 ]]; then return 0; fi
        log "  Queue full ($count/6). Waiting for a slot..."
        sleep "$POLL_INTERVAL"
    done
}

# Each submit function prints ONLY the numeric job ID to stdout (captured by caller).
# All other output goes to stderr via log().

submit_pi05() {
    wait_for_slot
    local out_dir="$ROOT_DIR/outputs/train/pi05_so101_bowl_placement"
    if [[ -d "$out_dir" ]]; then
        log "[Pi0.5] Removing partial output dir before submit"
        rm -rf "$out_dir"
    fi
    local raw jid
    raw=$(sbatch \
        --partition="$PI05_PARTITION" \
        --exclude="$EXCLUDED_NODES" \
        --export=ALL,STEPS="$PI05_STEPS",BATCH_SIZE=8,DEVICE=cuda,SKIP_PREPROCESS=1 \
        "$ROOT_DIR/scripts/train_pi05_lora.sh" 2>&1)
    jid=$(echo "$raw" | grep -oE '[0-9]+$' | tail -1)
    if [[ -z "$jid" ]]; then
        log "[Pi0.5] sbatch failed: $raw"
    fi
    echo "$jid"   # only output: the job number (or empty on failure)
}

submit_act() {
    wait_for_slot
    local out_dir="$ROOT_DIR/outputs/train/act_so101_bowl_placement"
    if [[ -d "$out_dir" ]]; then
        log "[ACT]   Removing partial output dir before submit"
        rm -rf "$out_dir"
    fi
    local raw jid
    raw=$(sbatch \
        --partition="$ACT_PARTITION" \
        --exclude="$EXCLUDED_NODES" \
        --export=ALL,STEPS="$ACT_STEPS",BATCH_SIZE=32,DEVICE=cuda,SKIP_PREPROCESS=1 \
        "$ROOT_DIR/scripts/train_act.sh" 2>&1)
    jid=$(echo "$raw" | grep -oE '[0-9]+$' | tail -1)
    if [[ -z "$jid" ]]; then
        log "[ACT]   sbatch failed: $raw"
    fi
    echo "$jid"
}

add_excluded_node() {
    local node="$1"
    if [[ -n "$node" ]] && [[ ",$EXCLUDED_NODES," != *",$node,"* ]]; then
        EXCLUDED_NODES="${EXCLUDED_NODES},${node}"
        log "  Updated excluded nodes: $EXCLUDED_NODES"
    fi
}

# ── parse CLI args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pi05-job) PI05_JOB_ID="$2"; shift 2 ;;
        --act-job)  ACT_JOB_ID="$2";  shift 2 ;;
        --skip-pi05) SKIP_PI05=1; shift ;;
        --skip-act)  SKIP_ACT=1;  shift ;;
        *) log "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── startup ────────────────────────────────────────────────────────────────────
log "=========================================="
log "Training monitor started  (PID $$)"
log "  Pi0.5 partition : $PI05_PARTITION  steps=$PI05_STEPS"
log "  ACT partition   : $ACT_PARTITION   steps=$ACT_STEPS"
log "  Excluded nodes  : $EXCLUDED_NODES"
log "  Poll interval   : ${POLL_INTERVAL}s"
log "=========================================="

if [[ "$SKIP_PI05" == "1" ]]; then
    PI05_DONE=1; log "[Pi0.5] Skipped"
elif [[ -n "$PI05_JOB_ID" ]]; then
    log "[Pi0.5] Adopting existing job $PI05_JOB_ID"
else
    PI05_JOB_ID=$(submit_pi05)
    if [[ -n "$PI05_JOB_ID" ]]; then
        log "[Pi0.5] Submitted job $PI05_JOB_ID  (partition: $PI05_PARTITION)"
    else
        log "[Pi0.5] Initial submission failed — will retry in first poll"
    fi
fi

if [[ "$SKIP_ACT" == "1" ]]; then
    ACT_DONE=1; log "[ACT]   Skipped"
elif [[ -n "$ACT_JOB_ID" ]]; then
    log "[ACT]   Adopting existing job $ACT_JOB_ID"
else
    ACT_JOB_ID=$(submit_act)
    if [[ -n "$ACT_JOB_ID" ]]; then
        log "[ACT]   Submitted job $ACT_JOB_ID  (partition: $ACT_PARTITION)"
    else
        log "[ACT]   Initial submission failed — will retry in first poll"
    fi
fi

# ── polling loop ───────────────────────────────────────────────────────────────
while [[ "$PI05_DONE" -eq 0 || "$ACT_DONE" -eq 0 ]]; do
    sleep "$POLL_INTERVAL"

    # ── Pi0.5 ──────────────────────────────────────────────────────────────────
    if [[ "$PI05_DONE" -eq 0 ]]; then
        if [[ -z "$PI05_JOB_ID" ]]; then
            log "[Pi0.5] No active job — resubmitting"
            PI05_JOB_ID=$(submit_pi05)
            [[ -n "$PI05_JOB_ID" ]] && log "[Pi0.5] Submitted job $PI05_JOB_ID  (partition: $PI05_PARTITION)"
        else
            STATE=$(job_state "$PI05_JOB_ID")
            log "[Pi0.5] Job $PI05_JOB_ID → $STATE"
            case "$STATE" in
                COMPLETED)
                    log "[Pi0.5] DONE ✓  outputs/train/pi05_so101_bowl_placement/checkpoints/last"
                    PI05_DONE=1 ;;
                FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                    add_excluded_node "$(job_node "$PI05_JOB_ID")"
                    PI05_JOB_ID=$(submit_pi05)
                    [[ -n "$PI05_JOB_ID" ]] && log "[Pi0.5] Resubmitted as job $PI05_JOB_ID" ;;
                RUNNING|PENDING|COMPLETING)
                    : ;;  # still going
                *)
                    log "[Pi0.5] Unexpected state '$STATE'" ;;
            esac
        fi
    fi

    # ── ACT ────────────────────────────────────────────────────────────────────
    if [[ "$ACT_DONE" -eq 0 ]]; then
        if [[ -z "$ACT_JOB_ID" ]]; then
            log "[ACT]   No active job — resubmitting"
            ACT_JOB_ID=$(submit_act)
            [[ -n "$ACT_JOB_ID" ]] && log "[ACT]   Submitted job $ACT_JOB_ID  (partition: $ACT_PARTITION)"
        else
            STATE=$(job_state "$ACT_JOB_ID")
            log "[ACT]   Job $ACT_JOB_ID → $STATE"
            case "$STATE" in
                COMPLETED)
                    log "[ACT]   DONE ✓  outputs/train/act_so101_bowl_placement/checkpoints/last"
                    ACT_DONE=1 ;;
                FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                    add_excluded_node "$(job_node "$ACT_JOB_ID")"
                    ACT_JOB_ID=$(submit_act)
                    [[ -n "$ACT_JOB_ID" ]] && log "[ACT]   Resubmitted as job $ACT_JOB_ID" ;;
                RUNNING|PENDING|COMPLETING)
                    : ;;
                *)
                    log "[ACT]   Unexpected state '$STATE'" ;;
            esac
        fi
    fi
done

log "=========================================="
log "Both training jobs completed successfully."
log "  Pi0.5 : $ROOT_DIR/outputs/train/pi05_so101_bowl_placement/checkpoints/last"
log "  ACT   : $ROOT_DIR/outputs/train/act_so101_bowl_placement/checkpoints/last"
log "=========================================="
