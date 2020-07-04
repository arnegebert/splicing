commands I ran:

curl ftp://ftp.ensembl.org/pub/release-100/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz -o human.fa.gz
salmon -t index human.fa.gz -i human_index
salmon -t human.fa.gz -i human_index
salmon index -t human.fa.gz -i human_index
salmon quant -i human_index -l ISF --gcBias -1 data/ERR1775544/ERR1775544_1.fastq.gz -2 data/ERR1775544/ERR1775544_2.fastq.gz -p 8 -o quants/ERR1775544