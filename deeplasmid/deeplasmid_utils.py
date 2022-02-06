#!/usr/bin/env python
"""
some utility functions
"""
import os
import re
import collections
import subprocess
from itertools import islice
import numpy
import scipy.stats


def chunker(i: int, size: int) -> list:
    """build chunks"""
    iterator = iter(i)
    while chunk := list(islice(iterator, size)):
        yield chunk


def prodigal(infile: str) -> str:
    """Run prodigal in subprocess.run"""
    name = os.path.splitext(infile)[0]
    # cmd = 'prodigal -q -i {n}.fa -a {n}.faa -f gff -o {n}.gff -p meta'.format(n=name)
    cmd = "prodigal -q -i {n}.fa -a {n}.faa -p meta".format(n=name)
    subprocess.check_output(cmd, shell=True)
    resfile = name + ".faa"
    return resfile


def encode1hot(sequence: str) -> numpy.array:
    """1hot encoding of sequence"""
    map_1hot = {}
    map_1hot["A"] = [1.0, 0.0, 0.0, 0.0]
    map_1hot["C"] = [0.0, 1.0, 0.0, 0.0]
    map_1hot["T"] = [0.0, 0.0, 1.0, 0.0]
    map_1hot["G"] = [0.0, 0.0, 0.0, 1.0]

    map_1hot["B"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["D"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["H"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["K"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["M"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["N"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["R"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["S"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["V"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["W"] = [0.0, 0.0, 0.0, 0.0]
    map_1hot["Y"] = [0.0, 0.0, 0.0, 0.0]
    return numpy.array([map_1hot[base] for base in sequence])


def get_statistics(scores: numpy.array) -> dict:
    """
    calculate a few statistics for a numpy array
    input: model average scores
    """
    out = {"avr": -1, "median": -1, "std": -1, "err": -1, "skew": -1, "hist": {}}
    if len(scores) > 0:
        out["avr"] = numpy.mean(scores).tolist()
        out["median"] = numpy.median(scores).tolist()
        out["std"] = numpy.std(scores).tolist()
        out["err"] = scipy.stats.sem(scores).tolist()  # error on the mean
        out["skew"] = scipy.stats.skew(scores)
        hist, thresholds = numpy.histogram(scores, bins=20, range=(0, 1))
        out["hist"] = {round(i, 2): j for i, j in zip(thresholds[:-1].tolist(), hist.tolist())}
    return out


def sketch(fasta_file_name: str, reference: str, do_translate: bool = True) -> list:
    """
    Quickly compare sequence to reference via sketch to find out:
    Does the sequence have common plasmid genes, or Origin of Replication
    See for more for example: https://www.biostars.org/p/234837/
    To make a separate sketch database from an aa fasta file, run this:
    sketch.sh in=x.faa out=x.sketch amino persequence
    """
    command = ["comparesketch.sh", "-Xmx12G", "-eoom", "index=false", "threads=1", "in=" + fasta_file_name, "ref=" + reference, "persequence=true"]
    if do_translate:
        command.append("translate=true")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    # Is there a hit or not?
    # Regarding string find method:
    # If the substring exists inside the string, it returns the index of the first occurence of the substring.
    # If substring doesn't exist inside the string, it returns -1.
    return [result.stdout, result.stderr, result.returncode, result.stdout.find("No hits.")]


def find_longest_homopolymer(seq: str) -> list:
    """
    Find the most freq homopolymers
    """
    longest_homopolymer = {}
    sum_homopolymer = {}
    start_longest_homopolymer = {}
    min_length = 6
    hits = re.findall(r"(([A-Z])\2\2+)", seq)
    for hit in hits:
        hit_len = len(hit[0])
        if hit[1] in longest_homopolymer:
            if hit_len > longest_homopolymer[hit[1]]:
                longest_homopolymer[hit[1]] = hit_len
                start_longest_homopolymer[hit[1]] = seq.find(hit[0])
            if hit_len >= min_length:
                sum_homopolymer[hit[1]] += hit_len
        else:
            longest_homopolymer[hit[1]] = hit_len
            start_longest_homopolymer[hit[1]] = seq.find(hit[0])
            if hit_len >= min_length:
                sum_homopolymer[hit[1]] = hit_len
            else:
                sum_homopolymer[hit[1]] = 0

    return longest_homopolymer, sum_homopolymer


def normalize_features(input_dict: collections.OrderedDict) -> collections.OrderedDict:
    """
    normalize some features
    they are normalized to be in [-0.5,0.5]
    others are kept the same since they are already in [0,1]
    for example: GC value, all the sketch values are either 0 or 1
    """
    output_dict = collections.OrderedDict()
    for feature, value in input_dict.items():
        if feature == "len_sequence":
            output_dict[feature] = max(0.007, min(1.0, value / 3e5)) - 0.5  # 30k is the max length, 0.007 is the min accepted value for this feature
        elif feature == "genecount":
            output_dict[feature] = max(0.007, min(1.0, value / 1e3)) - 0.5  # 1k is the max genecount, 0.007 is the min accepted value for this feature
        elif feature == "aalenavg":
            output_dict[feature] = max(0.007, min(1.0, value / 1e3)) - 0.5  # 1k is the max aalenavg, 0.007 is the min accepted value for this feature
        elif "longestHomopol" in feature:
            output_dict[feature] = min(1, value / 15.0) - 0.5  # we don't expect homopols of length >15 and 1 is the max accepted value for this feature
        elif "totalLongHomopol" in feature:
            output_dict[feature] = min(1, value / 1000.0) - 0.5  # we don't expect homopols of length >1000 and 1 is the max accepted value for this feature
        else:
            output_dict[feature] = value  # others are kept the same, e.g. GC value already in [0,1]

    return output_dict


def normalize(input_dict: collections.OrderedDict) -> collections.OrderedDict:
    """
    normalize features to [-1,1]
    f: [0,inf] -> [-1,1] via y = (x-1)/(x+1)
    g: [0,1] -> [-1,1]  via y = 2.x - 1
    """
    output_dict = collections.OrderedDict()
    for feature, value in input_dict.items():
        if feature in ["len_sequence", "genecount", "aalenavg"]:
            output_dict[feature] = (value - 1) / (value + 1)
        elif "Homopol" in feature:
            output_dict[feature] = (value - 1) / (value + 1)
        else:
            output_dict[feature] = 2 * value - 1

    return output_dict


def sample_sequence(sequence: str, number_of_samples: int, length_of_subsequence: int) -> list:
    """
    Purpose: select a number of subsequences from sequence
    Input:
    :param sequence: sequence from which to sample
    :param number_of_samples: number of subsequences
    Output:
    :list of subsequences: list of subsequences
    """
    list_of_subsequences = []
    if len(sequence) > length_of_subsequence:
        maximum_number_of_subsequence_starting_points = len(sequence) - length_of_subsequence
        nsamples = min(number_of_samples, maximum_number_of_subsequence_starting_points)
        list_of_sequence_indices = numpy.random.choice(maximum_number_of_subsequence_starting_points, nsamples, replace=False)
        for i in list_of_sequence_indices:
            subsequence = sequence[i : i + length_of_subsequence]
            if "NNNN" not in subsequence:
                list_of_subsequences.append(subsequence)
    else:
        print("<W> sample_sequence: length of sequence < length of subsequence")
    return list_of_subsequences
