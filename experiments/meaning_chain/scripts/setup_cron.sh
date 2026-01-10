#!/bin/bash
#
# Setup Cron Job for Nightly Decay
# =================================
#
# This script sets up a cron job to run the nightly decay job.
#
# Usage:
#   ./setup_cron.sh [install|remove|status]
#
# The cron job will:
#   - Run at 3:00 AM every day
#   - Apply 1 day of weight decay
#   - Log output to decay.log
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEANING_CHAIN_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_PATH="${PYTHON_PATH:-python3}"
LOG_FILE="$MEANING_CHAIN_DIR/logs/decay.log"
CRON_SCHEDULE="0 3 * * *"  # 3:00 AM daily

# Ensure logs directory exists
mkdir -p "$MEANING_CHAIN_DIR/logs"

# The actual command to run
CRON_COMMAND="cd $MEANING_CHAIN_DIR && $PYTHON_PATH -m scripts.nightly_decay --quiet >> $LOG_FILE 2>&1"

# Cron job identifier (for finding/removing)
CRON_ID="# meaning_chain_nightly_decay"

show_status() {
    echo "=== Nightly Decay Cron Status ==="
    echo ""
    echo "Script location: $SCRIPT_DIR/nightly_decay.py"
    echo "Log file:        $LOG_FILE"
    echo ""

    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
        echo "Status: INSTALLED"
        echo ""
        echo "Current cron entry:"
        crontab -l 2>/dev/null | grep -A1 "$CRON_ID"
    else
        echo "Status: NOT INSTALLED"
    fi
    echo ""

    # Check if log exists and show last entries
    if [ -f "$LOG_FILE" ]; then
        echo "Last 5 log entries:"
        tail -5 "$LOG_FILE"
    fi
}

install_cron() {
    echo "Installing nightly decay cron job..."
    echo ""

    # Remove existing entry if present
    remove_cron_silent

    # Add new cron entry
    (crontab -l 2>/dev/null; echo "$CRON_ID"; echo "$CRON_SCHEDULE $CRON_COMMAND") | crontab -

    echo "Cron job installed!"
    echo ""
    echo "Schedule: $CRON_SCHEDULE (3:00 AM daily)"
    echo "Command:  $CRON_COMMAND"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "To verify:"
    echo "  crontab -l | grep meaning_chain"
    echo ""
    echo "To test manually:"
    echo "  cd $MEANING_CHAIN_DIR && $PYTHON_PATH -m scripts.nightly_decay --dry-run"
}

remove_cron() {
    echo "Removing nightly decay cron job..."
    remove_cron_silent
    echo "Done."
}

remove_cron_silent() {
    # Remove existing cron entry (silently)
    crontab -l 2>/dev/null | grep -v "$CRON_ID" | grep -v "nightly_decay" | crontab -
}

show_help() {
    echo "Setup Cron Job for Nightly Decay"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  install   Install the cron job"
    echo "  remove    Remove the cron job"
    echo "  status    Show current status"
    echo "  help      Show this help message"
    echo ""
    echo "The cron job runs at 3:00 AM daily and applies weight decay"
    echo "to semantic edges in the meaning_chain Neo4j database."
    echo ""
    echo "Manual test:"
    echo "  cd $MEANING_CHAIN_DIR"
    echo "  $PYTHON_PATH -m scripts.nightly_decay --dry-run"
}

# Main
case "${1:-status}" in
    install)
        install_cron
        ;;
    remove)
        remove_cron
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac
