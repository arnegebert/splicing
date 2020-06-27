python extract_seq_and_estimate_psi.py --tissue brain & P1=$!
python extract_seq_and_estimate_psi.py --tissue heart & P2=$!
python extract_seq_and_estimate_psi.py --tissue cerebellum & P3=$!
wait $P1 $P2 $P3