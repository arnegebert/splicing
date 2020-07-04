lines = []
with open('iso_tpm.txt') as f:
    for i, line in enumerate(f):
        if i==0:
            lines.append(line)
            continue
        ident, tpm = line.split('\t')
        idx = ident.find('.')
        lines.append('\t'.join([ident[:idx], tpm]))

with open('iso_tpm_formatted.txt', 'w') as f:
    for line in lines:
        f.write(line)