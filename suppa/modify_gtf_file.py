lines = []
with open('../data/gencode.v34.annotation.gtf') as f:
    for i, line in enumerate(f):
        line = line.split('\t')
        if i < 5:
            lines.append('\t'.join(line))
            continue
        print(f'Line {i}')
        comments = line[8].split(';')
        reconstructed_comment = []
        for c in comments:
            if 'gene_id' or 'transcript_id' in c:
                idx = c.find('.')
                new = c[:idx] + '"'
                reconstructed_comment.append(new)
            else:
                reconstructed_comment.append(c)
        reconstructed_comment = ';'.join(reconstructed_comment)
        line.pop()
        line.append(reconstructed_comment)
        lines.append('\t'.join(line))

with open('formatted_gencode.v34.annotation.gtf', 'w') as f:
    for i, line in enumerate(lines):
        f.write(f'{line}\n')