python filter_by_tissue.py --tissue 'Brain - Cortex' --save-to '../../data/gtex_processed/brain_cortex_junction_reads_one_sample.csv' & P1=$!
python filter_by_tissue.py --tissue 'Brain - Cerebellum' --save-to '../../data/gtex_processed/cerebellum_junction_reads_one_sample.csv' & P2=$!
python filter_by_tissue.py --tissue 'Heart - Left Ventricle' --save-to '../../data/gtex_processed/heart_junction_reads_one_sample.csv' & P3=$!
wait $P1 $P2 $P3
