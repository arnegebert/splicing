commands I ran: 

python multipleFieldSelection.py -i ../salmon/quants/ERR1775544/quant.sf -k 1 -f 4 -o iso_tpm.txt
python SUPPA-master/suppa.py generateEvents -i ../data/gencode.v34.annotation.gtf -o hg38.events -e SE -f ioe --pool-genes
python SUPPA-master/suppa.py psiPerEvent -i hg38.events_SE_strict.ioe -e iso_tpm.txt -o helloSuppa


python SUPPA-master/suppa.py generateEvents -i formatted_gencode.v34.annotation.gtf -o second.events -e SE -f ioe --pool-genes
python SUPPA-master/suppa.py psiPerEvent -i second.events_SE_strict.ioe -e formatted_iso_tpm.txt -o second