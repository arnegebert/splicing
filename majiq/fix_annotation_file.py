

lines = []
with open('Homo_sapiens.GRCh38.100.chr.gff3') as f:
    for line in f:
        if line[0] == '#':
            lines.append(line)
            continue
        line = line.split('\t')
        line[0] = 'chr' + line[0]
        line = '\t'.join(line)
        lines.append(line)


with open('fix_Homo_sapiens.GRCh38.100.chr.gff3', 'w') as f:
    for line in lines:
        f.write(f'{line}')


print('Done')