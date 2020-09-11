To get data from the samples:  # 26
https://www.ebi.ac.uk/ena/data/view/ERR1775596
ERR1775544 - 0 
ERR1775551 - 1
ERR1775552 - 2
ERR1775553 - 3
ERR1775554 - 4
---
ERR1775594 - 5
ERR1775595 - 6
ERR1775596 - 7
ERR1775598 - 8
ERR1775600 - 9
ERR1775601 - 10
--- 
ERR1775631 - 11
ERR1775634 - 12
ERR1775637 - 13
ERR1775638 - 14
ERR1775640 - 15
ERR1775641 - 16
ERR1775643 - 17
ERR1775644 - 18
---
ERR1775684 - 19
ERR1775685 - 20
ERR1775686 - 21
ERR1775687 - 22
ERR1775688 - 23
ERR1775689 - 24
ERR1775693 - 25

79 == 20 mill reads => most => id 0 for me

To get there from IntelliJ
conda deactivate
cd ~
source env/bin/activate
cd Documents/GitHub/splicing/majiq

To be able to call MAJIQ from the commandline:  
source env/bin/activate

MAJIQ Buider
---
Neuron data:
majiq build fix_Homo_sapiens.GRCh38.100.gff3 -c neuron.config -j 8 -o builder/ --mem-profile --dump-constitutive 

iPSC data:
majiq build fix_Homo_sapiens.GRCh38.100.gff3 -c ipsc.config -j 8 -o builder_ipsc/ --mem-profile --dump-constitutive 


# inter-majiq processing of single samples
majiq psi builder/0.majiq builder/1.majiq builder/2.majiq builder/3.majiq builder/4.majiq builder/5.majiq builder/6.majiq builder/7.majiq builder/8.majiq builder/9.majiq builder/10.majiq builder/11.majiq builder/12.majiq builder/13.majiq builder/14.majiq builder/15.majiq builder/16.majiq builder/17.majiq builder/18.majiq builder/19.majiq builder/20.majiq builder/21.majiq builder/22.majiq builder/23.majiq builder/24.majiq builder/25.majiq -j 4 -o psi/ -n all2
majiq psi builder/0.majiq -j 4 -o psi/ -n all

voila tsv builder/splicegraph.sql psi/all.psi.voila -f voila/all.tsv
voila view builder/splicegraph.sql psi/all.psi.voila


majiq psi builder_ipsc/bezi1.majiq -j 4 -o psi_ipsc/ -n bezi1
majiq psi builder_ipsc/bezi2.majiq -j 4 -o psi_ipsc/ -n bezi2
majiq psi builder_ipsc/lexy2.majiq -j 4 -o psi_ipsc/ -n lexy2

voila tsv builder_ipsc/splicegraph.sql psi_ipsc/bezi1.psi.voila -f voila/bezi1.tsv
voila tsv builder_ipsc/splicegraph.sql psi_ipsc/bezi2.psi.voila -f voila/bezi2.tsv
voila tsv builder_ipsc/splicegraph.sql psi_ipsc/lexy2.psi.voila -f voila/lexy2.tsv

voila view builder_ipsc/splicegraph.sql psi_ipsc/bezi1.psi.voila
voila view builder_ipsc/splicegraph.sql psi_ipsc/bezi2.psi.voila
voila view builder_ipsc/splicegraph.sql psi_ipsc/lexy2.psi.voila



## Workflow to go from MAJIQ output to dataset
# constitutive junctions/exon:
sort_cons_juncs.py
add_strand_to_cons.py  -- associate each junction with a strand by finding junction in GENCODE GTF file
extract_seq_majiq_exon_cons.py -- go from junctions to exons and extract sequence

# non-cons junc/exons:
To get alternatively spliced exon I wanted download a list of their ids from voila by filtering for binary and cassette
filter_voila_tsv_from_voila_viewer --- use the list from voila to filter the voila tsv
extract_seq_majiq_exon_cons.py --- use filtered voila tsv to extract seqs


since texstudio sucks:
ERR177-: 
5544, 5551, 5552, 5554, 5594. 5595, 5596, 5598, 5600, 5601, 5631, 5634, 5637, 5638, 5640, 5641, 5643, 5644, 5684, 5685, 5686, 5687, 5688, 5689, 5693 25 biological replicates:
 
ERR1775544 - 0 
ERR1775551 - 1
ERR1775552 - 2
ERR1775553 - 3
ERR1775554 - 4
---
ERR1775594 - 5
ERR1775595 - 6
ERR1775596 - 7
ERR1775598 - 8
ERR1775600 - 9
ERR1775601 - 10
--- 
ERR1775631 - 11
ERR1775634 - 12
ERR1775637 - 13
ERR1775638 - 14
ERR1775640 - 15
ERR1775641 - 16
ERR1775643 - 17
ERR1775644 - 18
---
ERR1775684 - 19
ERR1775685 - 20
ERR1775686 - 21
ERR1775687 - 22
ERR1775688 - 23
ERR1775689 - 24
ERR1775693 - 25

20:
ERR-:
914342, 946968, 946976, 946983, 946984, 946990, 946992, 946994, 947011, 1203463, 1243454, 1274914, 1274917, 1724696, 1724699, 1743789, 2039345, 2039336, 2278244 2278245