import pandas as pd

import h5py
import hdf5plugin
import numpy as np

import argparse
import gc
import sys
import warnings


parser = argparse.ArgumentParser(description="calculate mean shap over the given h5s")
parser.add_argument("--counts_shaps", type=str, default=None, help="counts shap h5s - counts_fold0.h5,counts_fold1.h5,counts_fold2.h5")
parser.add_argument("--profile_shaps", type=str, default=None, help="profile shap h5s - profile_fold0.h5,profile_fold1.h5,profile_fold2.h5")
parser.add_argument("--output_dir", type=str,default='/cromwell_root/', help="output directory for the mean_shap")

#: the datasets of a score file
DATASETS = ('hyp_scores', 'coords_chrom', 'coords_start', 'coords_end',
            'input_seqs')


def same(first, other, rows=4096):
    """
        Are two arrays the same? Compared a block of rows at a time,
        so that the folds of a large experiment are compared without
        another array of their size.
    """

    if first.shape != other.shape:
        return False
    if first.ndim == 0:
        return bool(np.array_equal(first, other))
    return all(np.array_equal(first[i:i + rows], other[i:i + rows])
               for i in range(0, len(first), rows))


def mean_shap(shaps_list,output_path):
    hyp_scores_lst=[]
    chrom_lst=[]
    start_lst=[]
    end_lst=[]
    input_seqs_lst=[]
    # the files that were read
    read=[]
    given=shaps_list.split(',')
    for shap_h5 in given:
        try:
            # everything of a fold or nothing of it: a file that
            # cannot be read whole is left out whole
            with h5py.File(shap_h5, 'r') as f:
                fold = [f[name][()] for name in DATASETS]
        except MemoryError:
            # (no file that cannot be read: left out, the mean would
            # be one of fewer folds for want of memory)
            raise
        except Exception as error:
            # a fold that is not there or cannot be read is left out
            # with a warning, and the mean is that of the others
            # (python -W error::UserWarning makes it an error)
            warnings.warn(f'{shap_h5} does not exist or cannot be read '
                          f'({type(error).__name__}: {error}): it is left '
                          f'out of the mean')
            continue
        read.append(shap_h5)
        hyp_scores_lst.append(fold[0])
        chrom_lst.append(fold[1])
        start_lst.append(fold[2])
        end_lst.append(fold[3])
        input_seqs_lst.append(fold[4])

    if not read:
        sys.exit(f'None of the score files could be read ({shaps_list}): '
                 f'{output_path} is not written')
    if len(read) < len(given):
        warnings.warn(f'{output_path} is the mean of {len(read)} of '
                      f'{len(given)} folds')

    # the folds must be of the same peaks and the same sequences: a
    # mean over anything else would be written under the peaks and the
    # sequences of the first fold
    for what, values in (('chromosomes', chrom_lst), ('starts', start_lst),
                         ('ends', end_lst), ('sequences', input_seqs_lst),
                         ('numbers of scores',
                          [np.array(scores.shape) for scores in
                           hyp_scores_lst])):
        for shap_h5, value in zip(read[1:], values[1:]):
            if not same(values[0], value):
                sys.exit(f'{shap_h5} and {read[0]} do not have the same '
                         f'{what}: {output_path} is not written')

    # in float32: the folds hold their scores in float16, in which
    # the sum of several folds is rounded at every step (and two
    # scores of 40,000 add up to infinity)
    hyp_scores_mean = np.nanmean(np.array(hyp_scores_lst),axis=0,
                                 dtype=np.float32)

    num_examples = hyp_scores_mean.shape[0]

    f = h5py.File(output_path, "w")

    # (the names are not compressed like the other datasets:
    # writing text of variable length through the Blosc filter
    # ends the process with a floating point exception, with
    # hdf5plugin 4.4 and 7.1 at least)
    coords_chrom_dset = f.create_dataset(
        "coords_chrom", (num_examples,),
        dtype=h5py.string_dtype(encoding="ascii")
    )
    coords_chrom_dset[:] = chrom_lst[0]

    coords_start_dset = f.create_dataset(
        "coords_start", (num_examples,), dtype=int, **hdf5plugin.Blosc()
    )
    coords_start_dset[:] = start_lst[0]

    coords_end_dset = f.create_dataset(
        "coords_end", (num_examples,), dtype=int, **hdf5plugin.Blosc()
    )
    coords_end_dset[:] = end_lst[0]

    hyp_scores_dset = f.create_dataset(
        "hyp_scores", (num_examples, hyp_scores_mean.shape[1], hyp_scores_mean.shape[2]), dtype="f4", **hdf5plugin.Blosc()
    )
    hyp_scores_dset[:, :, :] = hyp_scores_mean

    input_seqs_dset = f.create_dataset(
        "input_seqs", (num_examples, hyp_scores_mean.shape[1], hyp_scores_mean.shape[2]), dtype="f4", **hdf5plugin.Blosc()
    )
    input_seqs_dset[:, :, :] = input_seqs_lst[0]

    f.close()
    
    del(input_seqs_lst,hyp_scores_mean,chrom_lst,start_lst,end_lst,hyp_scores_lst)
    
    gc.collect()
    
    


args = parser.parse_args()

if not args.counts_shaps and not args.profile_shaps:
    parser.error("no score files: give --counts_shaps, --profile_shaps or "
                 "both")

if args.counts_shaps:
    mean_shap(args.counts_shaps,f"{args.output_dir}/counts_mean_shap_scores.h5")

if args.profile_shaps:
    mean_shap(args.profile_shaps,f"{args.output_dir}/profile_mean_shap_scores.h5")