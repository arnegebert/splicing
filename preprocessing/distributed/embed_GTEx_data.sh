python embed_GTEx_data.py --tissue brain & P1=$!
python embed_GTEx_data.py --tissue heart & P2=$!
python embed_GTEx_data.py --tissue cerebellum & P3=$!
wait $P1 $P2 $P3