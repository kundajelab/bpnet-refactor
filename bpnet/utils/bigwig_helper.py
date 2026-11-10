import pyBigWig
import numpy as np
from tqdm import tqdm

def region_segments(regions, order):
    """
        The part of every region that is written to a bigWig
        
        Regions may overlap, and a bigWig holds one value per 
        position. Regions are taken in the order of the file. Where a 
        region overlaps the next one, it is written up to the 
        midpoint between the centres of the two, and the next one 
        from there: every position gets the value of the region 
        whose centre is closest.
        
        Args:
            regions (list): (chromosome, start, end, centre) of every 
                region
                
            order (list): indices of the regions to write, in the 
                order of the file (chromosomes as in the header, 
                then by start)
                
        Returns:
            list: (index of the region, chromosome, start, end) for 
                each region in `order`: the positions that are 
                written from it. A region between two others with 
                the same centre has none (start equals end).
    """
    
    segments = []
    cur_chr = ""
    cur_end = 0

    for itr in range(len(order)):

        i = order[itr]
        i_chr, i_start, i_end, i_mid = regions[i]
    
        if i_chr != cur_chr: 
            cur_chr = i_chr
            cur_end = 0
    
        # bring current end to at least start of current region
        if cur_end < i_start:
            cur_end = i_start
    
        assert(regions[i][2]>=cur_end)
    
        # figure out where to stop for this region, get next region
        # which may partially overlap with this one
        next_end = i_end
    
        if itr+1 != len(order):
            n = order[itr+1]
            next_chr, next_start, _, next_mid = regions[n]
       
            if next_chr == i_chr and next_start < i_end:
                # if next region overlaps with this, end between their midpoints
                next_end = (i_mid+next_mid)//2
        
        segments.append((i, i_chr, cur_end, next_end))
        
        cur_end = next_end
    
    return segments


def add_run(bw, chrom, start, end, values):
    """
        Add the values of the consecutive positions start .. end - 1 
        to a bigWig file that is being written
        
        The positions go into the file as one fixedStep run, which 
        holds a value per position. (An entry per position, with its 
        own start and end, takes three times the space before 
        compression and several times as long to write; what is read 
        back from the file, position by position or as a summary, is 
        the same.) Values are stored as float32, as always.
        
        Args:
            bw (pyBigWig.bigWigFile): file opened for writing, with 
                its header added
            
            chrom (str): chromosome
            
            start (int): first position
            
            end (int): position after the last
            
            values (numpy.ndarray): one value per position
    """
    
    start, end = int(start), int(end)
    if end == start:
        return
    
    # what pyBigWig refused when it was given every position (a 
    # fixedStep run has no ends to compare with)
    if start < 0 or end < start or len(values) != end - start:
        raise RuntimeError(
            "Cannot write {} values to {}:{}-{} of the bigWig file".format(
                len(values), chrom, start, end))
    
    # the conversion pyBigWig applies to Python floats: values beyond 
    # the range of float32 become infinite
    with np.errstate(over='ignore'):
        values = np.ascontiguousarray(values, dtype=np.float32)
    if not pyBigWig.numpy:
        values = [float(x) for x in values]
    
    bw.addEntries(str(chrom), start, values=values, span=1, step=1)


def write_bigwig(data, regions, header, bw_out, outstats_file):
    # regions may overlap but as we go in sorted order, at a given position,
    # we will pick the value from the interval whose summit is closest to 
    # current position
    
    chr_to_idx = {}
    for i,x in enumerate(header):
        chr_to_idx[x[0]] = i

    bw = pyBigWig.open(bw_out, 'w')
    bw.addHeader(header)
    
    # regions may not be sorted, so get their sorted order
    order_of_regs = sorted(range(len(regions)), key=lambda x:(chr_to_idx[regions[x][0]], regions[x][1]))

    all_entries = []

    for i, i_chr, cur_end, next_end in tqdm(
            region_segments(regions, order_of_regs)):
        
        i_start = regions[i][1]
        vals = data[i][cur_end - i_start:next_end - i_start]

        add_run(bw, i_chr, cur_end, next_end, vals)
    
        all_entries.append(vals)

    bw.close()

    all_entries = np.hstack(all_entries)

    with open(outstats_file, 'w') as f:
        f.write("Min\t{:.6f}\n".format(np.min(all_entries)))
        f.write(".1%\t{:.6f}\n".format(np.quantile(all_entries, 0.001)))
        f.write("1%\t{:.6f}\n".format(np.quantile(all_entries, 0.01)))
        f.write("50%\t{:.6f}\n".format(np.quantile(all_entries, 0.5)))
        f.write("99%\t{:.6f}\n".format(np.quantile(all_entries, 0.99)))
        f.write("99.9%\t{:.6f}\n".format(np.quantile(all_entries, 0.999)))
        f.write("99.95%\t{:.6f}\n".format(np.quantile(all_entries, 0.9995)))
        f.write("99.99%\t{:.6f}\n".format(np.quantile(all_entries, 0.9999)))
        f.write("Max\t{:.6f}\n".format(np.max(all_entries)))