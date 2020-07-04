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
majiq build fix_Homo_sapiens.GRCh38.100.gff3 -c majiq.config -j 8 -o builder/ --mem-profile --dump-constitutive 

# running this one for only one of the samples from above
majiq psi builder/0.majiq -j 4 -o psi/ -n all


voila tsv builder/splicegraph.sql psi/neural.psi.voila -f voila/ERR1775596_1.tsv

voila view builder/splicegraph.sql psi/all.psi.voila



