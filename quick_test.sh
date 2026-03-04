#!/bin/bash

# Quick test script for token averaging research
# Runs a fast test with k=1,2,4,8 on 10 sequences (~5-10 minutes)

echo "=================================="
echo "Token Averaging - Quick Test"
echo "=================================="
echo ""
echo "This will run a quick test with:"
echo "  - k values: 1, 2, 4, 8"
echo "  - Sequences: 10"
echo "  - Expected time: 5-10 minutes"
echo ""
echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
sleep 3

echo ""
echo "Starting test..."
echo ""

python run_all_analyses.py --k_max=8 --num_sequences=10

echo ""
echo "=================================="
echo "Test Complete!"
echo "=================================="
echo ""
echo "Check results in:"
echo "  - outputs/metrics/summary_metrics.csv"
echo "  - outputs/metrics/summary_metrics.json"
echo "  - outputs/plots/"
echo "  - outputs/summary_report.md"
echo ""
echo "To run full analysis:"
echo "  python run_all_analyses.py --k_max=128 --num_sequences=1000"
echo ""
