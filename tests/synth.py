"""
    Builder for a small synthetic BPNet dataset: genome, bigWigs, peaks,
    background loci and the JSON files the command line tools take.

    The data have real structure, so a tiny model can learn from them in
    a few seconds: every peak summit carries a planted motif, the number
    of reads in a peak grows with the number of motif copies around the
    summit, plus-strand reads pile up upstream of the summit and
    minus-strand reads downstream, and a control track adds reads that
    are unrelated to the motif.

    Everything is drawn from one seeded numpy RandomState, and the
    builder returns the arrays it wrote, so tests can compare what the
    package reads against the truth without trusting a second reader.

    Only numpy, pyBigWig and the standard library are used, so the
    builder runs in every environment the package supports.
"""

import json
import os

import numpy as np
import pyBigWig

#: motif planted at every peak summit
MOTIF = "TGACTCA"

#: bases per line of the FASTA file that is written
FASTA_LINE = 60


def write_bigwig(path, chrom_sizes, tracks):
    """
        Write a bigWig from dense per-chromosome arrays

        Only non-zero positions are written, as in a coverage track of
        read 5' ends; positions that are not written read back as NaN.

        Args:
            path (str): output file
            chrom_sizes (dict): chromosome name -> length, in the order
                the chromosomes go into the header
            tracks (dict): chromosome name -> 1D numpy array of values
    """

    bw = pyBigWig.open(path, "w")
    bw.addHeader([(c, int(n)) for c, n in chrom_sizes.items()], maxZooms=0)
    for chrom in chrom_sizes:
        values = tracks[chrom]
        idx = np.flatnonzero(values)
        if len(idx) == 0:
            continue
        bw.addEntries([chrom] * len(idx), [int(i) for i in idx],
                      ends=[int(i) + 1 for i in idx],
                      values=[float(v) for v in values[idx]])
    bw.close()


def write_fasta(path, sequences):
    """
        Write a FASTA file and its .fai index

        Args:
            path (str): output file; the index goes to path + '.fai'
            sequences (dict): chromosome name -> sequence string, in
                file order
    """

    offset = 0
    with open(path, "w") as fasta, open(path + ".fai", "w") as fai:
        for chrom, seq in sequences.items():
            fasta.write(">{}\n".format(chrom))
            offset += len(chrom) + 2
            fai.write("{}\t{}\t{}\t{}\t{}\n".format(
                chrom, len(seq), offset, FASTA_LINE, FASTA_LINE + 1))
            for i in range(0, len(seq), FASTA_LINE):
                fasta.write(seq[i:i + FASTA_LINE] + "\n")
            offset += len(seq) + (len(seq) + FASTA_LINE - 1) // FASTA_LINE


def write_narrowpeak(path, rows):
    """
        Write loci in 10 column narrowPeak format

        Args:
            path (str): output file
            rows (list): tuples (chrom, start, end, signal, summit) with
                the summit as an offset from start
    """

    with open(path, "w") as fh:
        for chrom, start, end, signal, summit in rows:
            fh.write("\t".join(map(str, [
                chrom, start, end, ".", 1, ".", round(float(signal), 3),
                -1, -1, summit])) + "\n")


def make_dataset(outdir, seed=0, chroms=("chr1", "chr2", "chr3"),
                 chrom_len=60000, n_peaks=240, n_background=240,
                 input_len=306, output_len=200, stranded=True,
                 with_control=True, background_ratio=0.5, num_tasks=1):
    """
        Create the dataset under `outdir`

        Args:
            outdir (str): directory to create the files in
            seed (int): seed of the RandomState that draws everything
            chroms (tuple): chromosome names
            chrom_len (int): length of every chromosome
            n_peaks (int): number of peaks (foreground loci)
            n_background (int): number of background loci
            input_len (int): model input length; loci keep this much
                distance from the chromosome ends
            output_len (int): model output length
            stranded (bool): two signal tracks per task (plus, minus)
                if True, one otherwise
            with_control (bool): write control tracks and list them
                under 'bias' in the input json
            background_ratio (float): the 'ratio' recorded for the
                background loci in the input json
            num_tasks (int): number of tasks in the input json; all
                tasks share the loci and get their own signal tracks

        Returns:
            dict: paths ('fasta', 'chrom_sizes', 'input_json', 'peaks',
                'background', 'params_json', 'splits_json', 'signal',
                'control') and the truth ('chroms', 'chrom_len',
                'input_len', 'output_len', 'peak_rows',
                'background_rows', 'tracks' with the dense array of
                every bigWig keyed by path, 'sequences')
    """

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    chrom_sizes = {c: chrom_len for c in chroms}

    # loci stay `margin` away from the chromosome ends, which leaves
    # room for the input window plus any jitter up to input_len // 2
    margin = input_len

    # random genome
    seqs = {c: rng.choice(list("ACGT"), size=chrom_len) for c in chroms}

    # non overlapping slots; a random subset becomes peaks, another
    # background
    n_loci = n_peaks + n_background
    slots_per_chrom = int(np.ceil(n_loci / float(len(chroms))))
    slot = (chrom_len - 2 * margin) // slots_per_chrom
    if slot < 130:
        raise ValueError("chromosomes too short for the requested loci")
    loci = []
    for chrom in chroms:
        for i in range(slots_per_chrom):
            center = margin + i * slot + slot // 2 + int(rng.randint(-5, 6))
            loci.append((chrom, center))
    order = rng.permutation(len(loci))[:n_loci]
    peak_idx = set(order[:n_peaks].tolist())
    background_idx = set(order[n_peaks:].tolist())

    peak_rows, background_rows, strengths = [], [], {}
    for i, (chrom, center) in enumerate(loci):
        if i in peak_idx:
            n_copies = int(rng.randint(1, 5))
            offsets = set([0] + [int(o) for o in rng.choice(
                np.arange(-40, 41, 10), size=n_copies - 1)])
            for offset in offsets:
                pos = center + offset
                seqs[chrom][pos:pos + len(MOTIF)] = list(MOTIF)
            strength = 40.0 * len(offsets)
            width = int(rng.randint(80, 121))
            start = center - width // 2
            peak_rows.append(
                (chrom, start, start + width, strength, center - start))
            strengths[(chrom, center)] = strength
        elif i in background_idx:
            start = center - 50
            background_rows.append(
                (chrom, start, start + 100, 0.0, center - start))

    sequences = {c: "".join(seqs[c]) for c in chroms}
    fasta = os.path.join(outdir, "genome.fa")
    write_fasta(fasta, sequences)
    sizes_path = os.path.join(outdir, "chrom.sizes")
    with open(sizes_path, "w") as fh:
        for chrom in chroms:
            fh.write("{}\t{}\n".format(chrom, chrom_len))

    # signal and control tracks
    rel = np.arange(-60, 61)
    n_strands = 2 if stranded else 1
    names = ["plus", "minus"] if stranded else ["unstranded"]
    tracks, signal_paths, control_paths = {}, [], []
    for task in range(num_tasks):
        for strand in range(n_strands):
            shift = (-15 if strand == 0 else 15) if stranded else 0
            shape = np.exp(-0.5 * ((rel - shift) / 12.0) ** 2)
            shape = shape / shape.sum()
            signal = {c: rng.poisson(0.01, size=chrom_len).astype(np.float64)
                      for c in chroms}
            control = {c: rng.poisson(0.02, size=chrom_len).astype(np.float64)
                       for c in chroms}
            for (chrom, center), strength in sorted(strengths.items()):
                window = slice(center - 60, center + 61)
                signal[chrom][window] += rng.poisson(
                    strength * (1.0 + 0.5 * task) * shape)
                control[chrom][window] += rng.poisson(
                    0.05 * np.ones(len(rel)))

            path = os.path.join(
                outdir, "task{}_{}.bw".format(task, names[strand]))
            write_bigwig(path, chrom_sizes, signal)
            signal_paths.append(path)
            tracks[path] = signal
            if with_control:
                path = os.path.join(
                    outdir, "task{}_control_{}.bw".format(task, names[strand]))
                write_bigwig(path, chrom_sizes, control)
                control_paths.append(path)
                tracks[path] = control

    peaks_path = os.path.join(outdir, "peaks.bed")
    background_path = os.path.join(outdir, "background.bed")
    write_narrowpeak(peaks_path, peak_rows)
    write_narrowpeak(background_path, background_rows)

    input_data = {}
    for task in range(num_tasks):
        signal = signal_paths[task * n_strands:(task + 1) * n_strands]
        control = control_paths[task * n_strands:(task + 1) * n_strands]
        input_data[str(task)] = {
            "signal": {"source": signal},
            "loci": {"source": [peaks_path]},
            "background_loci": {"source": [background_path],
                                "ratio": [background_ratio]},
            "bias": {"source": control, "smoothing": [None] * len(control)},
        }
    input_json = os.path.join(outdir, "input_data.json")
    with open(input_json, "w") as fh:
        json.dump(input_data, fh, indent=2)

    # a small architecture with the receptive field arithmetic of the
    # default one: 20 (first conv) + 2 * (2 + 4) (two dilated convs)
    # + 74 (profile head) = input_len - output_len = 106
    params = {
        "input_len": input_len,
        "output_profile_len": output_len,
        "motif_module_params": {
            "filters": [8], "kernel_sizes": [21], "padding": "valid"},
        "syntax_module_params": {
            "num_dilation_layers": 2, "filters": 8, "kernel_size": 3,
            "padding": "valid", "pre_activation_residual_unit": True},
        "profile_head_params": {
            "filters": 1, "kernel_size": 75, "padding": "valid"},
        "counts_head_params": {
            "units": [1], "dropouts": [0.0], "activations": ["linear"]},
        "profile_bias_module_params": {"kernel_sizes": [1] * num_tasks},
        "counts_bias_module_params": {},
        "loss_weights": [1, 10],
        "counts_loss": "MSE",
    }
    params_json = os.path.join(outdir, "bpnet_params.json")
    with open(params_json, "w") as fh:
        json.dump(params, fh, indent=2)

    splits = {"0": {"test": [chroms[-1]], "val": [chroms[-2]],
                    "train": list(chroms[:-2])}}
    splits_json = os.path.join(outdir, "splits.json")
    with open(splits_json, "w") as fh:
        json.dump(splits, fh, indent=2)

    return {
        "outdir": outdir,
        "fasta": fasta, "chrom_sizes": sizes_path, "input_json": input_json,
        "peaks": peaks_path, "background": background_path,
        "params_json": params_json, "splits_json": splits_json,
        "signal": signal_paths, "control": control_paths,
        "chroms": list(chroms), "chrom_len": chrom_len,
        "input_len": input_len, "output_len": output_len,
        "peak_rows": peak_rows, "background_rows": background_rows,
        "tracks": tracks, "sequences": sequences,
        "params": params, "splits": splits, "input_data": input_data,
    }
