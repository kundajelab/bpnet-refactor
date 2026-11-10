"""
    Earlier implementations of functions that have since been replaced
    by faster ones, kept verbatim so the tests can show that the new
    code returns the same values.
"""

import logging
import random

import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx

from bpnet.generators import sequtils
from bpnet.utils.exceptionhandler import NoTracebackException
from bpnet.utils.misc import gaussian1D_smoothing


def one_hot_encode_v0(sequences, seq_length):
    """
        one_hot_encode as it was before the lookup table version
        (bpnet/generators/sequtils.py at main@0b11a7c), with its helper
        fix_sequence_length. It turns the letters A C G T N Y R M S W K
        into the digits 0 to 4 with str.translate, converts every
        character with int() and indexes a five row matrix with the
        result. Any character int() does not accept raises ValueError;
        a digit above 4 raises IndexError.
    """

    def fix_sequence_length(sequence, length):
        if len(sequence) < length:
            sequence += 'N' * (length - len(sequence))
        elif len(sequence) > length:
            sequence = sequence[:length]
        return sequence

    if len(sequences) == 0:
        logging.error("'sequences' is empty")
        return None

    sequences = list(map(
        fix_sequence_length, sequences, [seq_length] * len(sequences)))
    _sequences = ''.join(sequences)
    transtab = str.maketrans('ACGTNYRMSWK', '01234444444')
    sequences_trans = _sequences.translate(transtab)
    int_sequences = list(map(int, sequences_trans))
    encoder = np.vstack([np.eye(4), np.zeros(4)])
    X = encoder[int_sequences]
    return X.reshape(len(sequences), len(sequences[0]), 4)


def interval_values_v0(bigwig, chrom, start, end):
    """
        How the batch generator read an interval of a bigWig file
        before interval_values (bpnet/generators/generators.py at
        main@0b11a7c): a Python list of floats from pyBigWig, converted
        by numpy.nan_to_num into a float64 array
    """

    return np.nan_to_num(bigwig.values(chrom, start, end))


def generate_batch_v0(self, coords):
    """
        Generate one batch of inputs and outputs for training BPNet

        For all coordinates in "coords" fetch sequences &
        one hot encode the sequences. Fetch corresponding
        signal and bias values (from bigwig files). 
        Package the one hot encoded sequences and the output
        values as a tuple.

        Args:
            coords (pandas.DataFrame): dataframe with 'chrom', 
                'pos', 'weight' & 'status' columns specifying the 
                chromosome, the coordinate and whether the loci is
                a positive(1) or negative sample(-1)

        Returns:
            tuple: 
                When 'mode' is 'train' or 'val' a batch tuple 
                with one hot encoded sequences and corresponding 
                outputs and when 'mode' is 'test' tuple of 
                cordinates & the inputs
    """

    # create a new reverse complement column in the dataframe
    # and set it to 0 for all loci (0-not reverse complemented, 
    # 1-reverse complemented)
    coords['rev_comp'] = 0

    # if reverse complement is enabled double the 'coords'
    if self._rev_comp_aug:
        # make a copy of coords, but set the rev_comp flag to 1
        coords_copy = coords.copy()
        coords_copy['rev_comp'] = 1
        coords = pd.concat([coords, coords_copy])

    # Initialize arrays to hold expected profile predictions,
    # counts predictions and profile bias and counts bias inputs

    # profile predictions across all tasks 
    profile_predictions = np.zeros(
        (coords.shape[0], self._output_flank * 2, 
         self._total_signal_tracks), dtype=np.float32)

    # counts predictions across all tasks 
    logcounts_predictions = np.zeros(
        (coords.shape[0], self._total_signal_tracks), dtype=np.float32)

    # profile and counts bias inputs for each task separately
    profile_bias_input = {}
    counts_bias_input = {}
    for i in range(self._num_tasks):
        # the number of bias tracks for the ith tasks, which
        # includes smoothed versions
        num_bias_tracks = self._get_num_bias_tracks_for_task(
            self._tasks[i])

        if num_bias_tracks > 0:
            # profile bias input for the ith task
            profile_bias_input[i] = np.zeros(
                (coords.shape[0], self._output_flank * 2, num_bias_tracks), 
                dtype=np.float32)

            # counts bias input for the ith task
            counts_bias_input[i] = np.zeros(
                (coords.shape[0], num_bias_tracks), dtype=np.float32)

    # Initialization done.

    # list of sequences in the batch, these will be one hot
    # encoded together as a single sequence after iterating
    # over the batch
    sequences = []  

    # list of chromosome start/end coordinates for the batch
    coordinates = []

    # list of jitter values for the batch coordinates
    jitters = []

    # list of index values for the batch coordinates
    idxs = coords['idx'].values

    # sample weights
    weights = coords['weight'].astype(float).values

    # open all the signal and bias bigwig files and store the  
    # file objects in a dictionary
    signal_files = {}
    bias_files = {}
    for task in self._tasks:
        signal_files[task] = []
        for signal_file in self._tasks[task]['signal']['source']:
            signal_files[task].append(pyBigWig.open(signal_file))

        bias_files[task] = []
        for bias_file in self._tasks[task]['bias']['source']:
            bias_files[task].append(pyBigWig.open(bias_file))

    # reference file to fetch sequences
    fasta_ref = pyfaidx.Fasta(self._reference)

    # iterate over the batch
    rowCnt = 0
    for _, row in coords.iterrows():
        # randomly set a jitter value to move the peak summit 
        # slightly away from the exact center (only for samples 
        # whose weight is not 0)
        jitter = 0
        if self._mode == "train" and self._max_jitter and \
            row['weight'] != 0:

            jitter = random.randint(-self._max_jitter, self._max_jitter)

        # record the jitter for this sample
        jitters.append(jitter)

        # Step 1. get the sequence 
        chrom = row['chrom']
        # we use self._input_flank here and not self._output_flank because
        # input_seq_len is different from output_len
        start = row['pos'] - self._input_flank + jitter
        end = row['pos'] + self._input_flank + jitter
        seq = fasta_ref[chrom][start:end].seq.upper()

        if row['rev_comp']==1:
            seq = sequtils.reverse_complement_of_sequences([seq])[0]

        # collect all the sequences into a list
        sequences.append(seq)

        start = row['pos'] - self._output_flank + jitter
        end = row['pos'] + self._output_flank + jitter

        # record the start/end coordinates for this sample
        coordinates.append((chrom, start, end))

        # track profile tracks across all tasks
        profile_track_idx = 0

        # iterate over each task and read the signal and bias
        # values from the bigWig files
        for i in range(self._num_tasks):

            cur_num_signal_files = len(signal_files[i])

            # Step 2. get the profile signal value
            for signal_file in signal_files[i]:
                profile_predictions[rowCnt, :, profile_track_idx] = \
                    interval_values_v0(signal_file, chrom, start, end)

                profile_track_idx += 1

            if row['rev_comp']==1:
                profile_predictions[rowCnt, :, profile_track_idx-cur_num_signal_files:profile_track_idx] = \
                    sequtils.reverse_complement_of_profiles(profile_predictions[rowCnt:rowCnt+1, :, 
                                                                                profile_track_idx-cur_num_signal_files:profile_track_idx],
                                                            stranded=(cur_num_signal_files==2))

            cur_num_bias_files = len(bias_files[i])
            # Step 3. get the bias values
            #skip setting the bias values. Initialization value of zero will be used
            if not self._set_bias_as_zero and cur_num_bias_files>0:
                assert(cur_num_bias_files <= 2)
                bias_track_idx = 0 

                for j in range(cur_num_bias_files):
                    bias_file = bias_files[i][j]
                    profile_bias_input[i][rowCnt, :, bias_track_idx] = \
                        interval_values_v0(bias_file, chrom, start, end)

                    bias_track_idx += 1                    

                cur_num_smoothed_bias_files = 0
                # add the smoothed track if 'smoothing' has been
                # specified                        
                for j in range(cur_num_bias_files):
                    if self._tasks[i]['bias']['smoothing'][j] is not None:
                        # get the smoothing params
                        sigma = self._tasks[i]['bias']['smoothing'][j][0]
                        window_size = \
                            self._tasks[i]['bias']['smoothing'][j][1]

                        # the smoothed bias track will follow after 
                        # the original bias tracks in the last
                        # dimension
                        profile_bias_input[i][rowCnt, :, bias_track_idx] = \
                            gaussian1D_smoothing(
                                profile_bias_input[i][
                                    rowCnt, :, j],
                                sigma, window_size)                            

                        bias_track_idx += 1
                        cur_num_smoothed_bias_files += 1

                if row['rev_comp']==1:
                    profile_bias_input[i][rowCnt, :, 0:cur_num_bias_files] = \
                        sequtils.reverse_complement_of_profiles(profile_bias_input[i][rowCnt:rowCnt+1, :, 
                                                                                    0:cur_num_bias_files],
                                                                                    stranded=(cur_num_bias_files==2))
                    # for edge cases when only one strand out of 2 strands is smoothed
                    if cur_num_smoothed_bias_files > 0:
                        profile_bias_input[i][rowCnt, :, cur_num_bias_files:cur_num_bias_files+cur_num_smoothed_bias_files] = \
                           sequtils.reverse_complement_of_profiles(profile_bias_input[i][rowCnt:rowCnt+1, :, 
                                                                                    cur_num_bias_files:cur_num_bias_files+cur_num_smoothed_bias_files],
                                                                                    stranded=(cur_num_smoothed_bias_files==2))

        rowCnt += 1

    fasta_ref.close()

    # Step 4. one hot encode all the sequences in the batch 
    if len(sequences) == profile_predictions.shape[0]:
        X = sequtils.one_hot_encode(sequences, self._input_flank * 2)
    else:
        raise NoTracebackException(
            "Unable to generate enough sequences for the batch")

    # we can now compute the log(sum) of the profiles and bias
    # profiles for the entire batch
    logcounts_predictions = np.log(
        np.sum(profile_predictions, axis=1) + 1)

    if not self._set_bias_as_zero:
        #skip setting the bias values. Initialization value of zero will be used
        for key in profile_bias_input:
            counts_bias_input[key] = np.log(
                np.sum(profile_bias_input[key], axis=1) + 1)

    # inputs to train, val & test
    # 'coordinates', 'jitters', 'index', 'status' & 'rev_comp'
    # are not inputs to the model, so you will see a warning 
    # about unused inputs while training. It's safe to ignore
    # the warning. We pass 'coordinates' so we can track the exact
    # coordinates of the inputs (because jitter is random)
    # 'status' refers to whether the data sample is a +ve (1)
    # or -ve (-1) example and is used by the attribution
    # prior loss function        
    inputs = {
        'sequence': X, 
        'coordinates': np.array(coordinates)}

    # add profile bias input
    for key in profile_bias_input:
        _key = 'profile_bias_input_' + str(key)
        inputs[_key] = profile_bias_input[key]

    # add counts bias input
    for key in counts_bias_input:
        _key = 'counts_bias_input_' + str(key)
        inputs[_key] = counts_bias_input[key]

    # in 'train' mode we add some extras to track all the
    # loci used in every batch of training
    if self._mode == 'train':
        inputs['jitters'] = np.array(jitters)
        inputs['index'] = np.array(idxs)
        inputs['status'] = coords['status'].values
        inputs['rev_comp'] = coords['rev_comp'].values

    # in 'train' and 'val' mode we need outputs as well     
    if self._mode == 'train' or self._mode == 'val':
        outputs = {
            'profile_predictions': profile_predictions,
            'logcounts_predictions': logcounts_predictions}

        return (inputs, outputs, weights)

    # in 'test' mode we only return inputs
    elif self._mode == 'test':
        # we add the true profiles & counts so we can use those to
        # compute metrics
        inputs['true_profiles'] = profile_predictions
        inputs['true_logcounts'] = logcounts_predictions
        inputs['rev_comp'] = coords['rev_comp'].values # send this for rev comp averaging
        return inputs
