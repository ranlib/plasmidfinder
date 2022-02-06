#!/usr/bin/env python
"""
Get features
"""
import os
import sys
import argparse
import configparser
import datetime
import re
import math
import random
import csv
import gzip
import copy
import collections

# import logging
from mimetypes import guess_type
from functools import partial
import pathlib
import pathos
import numpy
from tensorflow import keras
from Bio import SeqIO
from Bio.SeqUtils import GC
from Bio import Seq
import yaml
import deeplasmid_utils
import deeplasmid_model


class DataGenerator(keras.utils.Sequence):
    """data generator"""

    def __init__(self, filenames: list, batch_size: int, shuffle: bool):
        self.filenames = filenames
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.filenames) // self.batch_size

    def __getitem__(self, idx: int):
        _subsequences1hot = []
        label = []
        features = []
        for _yaml_file in self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]:
            with gzip.open(_yaml_file, "rt", encoding="ascii") as file:
                _sample_data = yaml.load(file, Loader=yaml.CLoader)

            subsamples = _sample_data['classification']['model_input']['sample_list']
            normalized_features = _sample_data['classification']['model_input']['normalized_features']

            label.append(_sample_data["label"])
            features.append(_sample_data["normalized_features"])
            subsequences1hot.append(deeplasmid_utils.encode1hot(_sample_data["subsequence"]))

        return [numpy.array(subsequences1hot), numpy.array(features)], numpy.array(label)

    def on_epoch_end(self):
        """Shuffle after each epoch"""
        if self.shuffle:
            random.shuffle(self.filenames)


def process_sequence(input_fasta_file: str, configuration: configparser.ConfigParser(), do_prediction: bool, output_yaml_file: str) -> int:
    """
    calculate all the features for sequence
    input:
    input_fasta_file: single fasta file
    configuration: configparser object
    do_prediction: boolean
    output:
    output_yaml_file: output yaml files are going to be written
    """

    with open(input_fasta_file, "r", encoding="ascii") as fasta:
        records = SeqIO.parse(fasta, "fasta")  # should be a single fasta file
        record = next(records)
    print(f"process_sequence: {input_fasta_file}, {do_prediction}, {output_yaml_file}, {len(record.seq)}")

    # this_id = pathos.core.getpid()
    # fh = logging.FileHandler(os.path.join(configuration["LOGGING"]["output_directory"], 'pathos_' + record.id + '.log'))
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    # fh.setFormatter(formatter)
    # logger = pathos.logger(level=logging.DEBUG, handler=fh)

    #
    # calculate features
    #
    if configuration["PREDICTION"].getboolean("use_prot_sketch"):
        # logger.info("Process " + str(this_id) + ": Processing sketch " + record.id)
        # print("Process " + str(this_id) + ": Processing sketch " + record.id)
        chrom_sketch = deeplasmid_utils.sketch(input_fasta_file, configuration["SKETCH"]["chrom_sketch_reference"], True)
        plasm_sketch = deeplasmid_utils.sketch(input_fasta_file, configuration["SKETCH"]["plasm_sketch_reference"], True)
        plasm_ori_sketch = deeplasmid_utils.sketch(input_fasta_file, configuration["SKETCH"]["plasm_ORI_sketch_reference"], False)
    else:
        chrom_sketch = plasm_sketch = plasm_ori_sketch = 4 * [-1]

    if configuration["PREDICTION"].getboolean("use_prodigal"):
        # print("Process " + str(this_id) + ": Processing prodigal " + record.id)
        # logger.info("Process " + str(this_id) + ": Processing prodigal")
        protein_fasta_file = deeplasmid_utils.prodigal(input_fasta_file)
        genes = list(SeqIO.parse(protein_fasta_file, "fasta"))
        amino_acid_lengths = [len(gene.seq) for gene in genes]
        if len(amino_acid_lengths) > 0:
            average_amino_acid_sequence_length = sum(amino_acid_lengths) / len(amino_acid_lengths)
        else:
            average_amino_acid_sequence_length = -1

        number_of_genes = len(genes)
        fraction_of_sequence_covered_by_genes = 3 * sum(amino_acid_lengths) / len(record.seq)
    else:
        number_of_genes = fraction_of_sequence_covered_by_genes = average_amino_acid_sequence_length = 3 * [-1]

    # logger.info("Process " + str(this_id) + ": Processing homopolymers")
    longest_homopolymer, total_long_homopolymer = deeplasmid_utils.find_longest_homopolymer(str(record.seq))

    #
    # write features and other meta information to yaml file
    #
    data = {}
    # store configuration as well in output yaml file for this sequence
    data["configuration"] = configuration._sections

    # store sequence information
    sequence_information = {}
    sequence_information["id"] = record.id
    sequence_information["description"] = record.description
    sequence_information["sequence"] = str(record.seq)
    # in case of a labeled dataset
    # fasta labeling: add keyword _TRUTH_PLASMID, _TRUTH_GENOME, _TRUTH_VIRUS to end of fasta description record
    if record.description.find("_TRUTH_PLASMID") > -1:
        sequence_information["truth"] = "PLASMID"
    elif record.description.find("_TRUTH_GENOME") > -1:
        sequence_information["truth"] = "GENOME"
    elif record.description.find("_TRUTH_VIRUS") > -1:
        sequence_information["truth"] = "VIRUS"
    else:
        sequence_information["truth"] = "NONE"

    # attach training label
    if sequence_information["truth"] == "NONE":
        sequence_information["training_label"] = "NONE"
    else:
        sequence_information["training_label"] = "validation" if random.random() < configuration["TRAINING"].getfloat("fraction_for_validation") else "training"

    data["sequence"] = sequence_information

    # some run information
    header = {}
    header["run_datetime"] = datetime.datetime.now()
    header["backend"] = keras.backend.backend()
    data["header"] = header

    # store features
    # Note: order of features is critical! Order needs to be what Neural Network expects as inputs!
    features = collections.OrderedDict()
    features["gc_content"] = GC(record.seq) / 100.0
    features["len_sequence"] = len(record.seq)
    features["genecount"] = number_of_genes
    features["genesperMB"] = fraction_of_sequence_covered_by_genes
    features["aalenavg"] = average_amino_acid_sequence_length
    for base in ["A", "C", "T", "G"]:
        features[base + "_longestHomopol"] = longest_homopolymer.get(base, 0)
        features[base + "_totalLongHomopol"] = total_long_homopolymer.get(base, 0)

    features["plassketch"] = 0 if plasm_sketch[3] >= 0 else 1  # just store whether there was a hit (1) or not (0)
    features["plasORIsketch"] = 0 if plasm_ori_sketch[3] >= 0 else 1
    features["chromsketch"] = 0 if chrom_sketch[3] >= 0 else 1

    # normalize some features and store
    # Note: features and normalized_features are OrderedDict here! Later we need a vector!
    _normalized_features = deeplasmid_utils.normalize_features(features)

    data["features"] = features
    data["normalized_features"] = _normalized_features

    # Subsample sequence to get a list of subsequences
    # The number of subsequences to sample depends on sampling_rate (length-dependent)
    # times the target_samples_per_contig_pred or target_samples_per_contig_train
    # one hot encode subsequences (i.e. categorical data -> continuous)
    # put features in a python list, make sure the correct order!
    sampling_rate = 0.5 + 0.5 * math.sqrt(len(record.seq) / 1e4)
    number_of_samples = int(configuration["PREDICTION"].getint("target_samples_per_sequence_prediction") * sampling_rate)
    sample_list = deeplasmid_utils.sample_sequence(str(record.seq), number_of_samples, configuration["PREDICTION"].getint("seqlencut"))
    # 1hot encoding of subsequences
    # sample_list_1hot = [deeplasmid_utils.encode1hot(subsequence).flatten().tolist() for subsequence in sample_list]
    sample_list_1hot = [deeplasmid_utils.encode1hot(subsequence) for subsequence in sample_list]
    normalized_features_vector = list(_normalized_features.values())

    # prediction here
    classification = {}
    classification["model_input"] = {}
    classification["model_input"]["sampling_rate"] = sampling_rate
    classification["model_input"]["number_of_samples"] = len(sample_list)
    classification["model_input"]["number_of_Ns"] = [sample.count("N") for sample in sample_list]
    classification["model_input"]["sample_sequence_length"] = configuration["PREDICTION"].getint("seqlencut")
    classification["model_input"]["sample_list"] = sample_list
    # classification['model_input']['sample_list_1hot'] = [ "".join(map(str, map(int, sample))) for sample in sample_list_1hot]
    classification["model_input"]["sample_list_1hot"] = ["".join([str(int(b)) for b in subsequence1hot.flatten().tolist()]) for subsequence1hot in sample_list_1hot]
    classification["model_input"]["number_of_features"] = len(normalized_features_vector)
    classification["model_input"]["normalized_features"] = normalized_features_vector
    # tag sequences according to length
    if len(record.seq) <= configuration["PREDICTION"].getint("minseqlen"):
        classification["tag"] = "SHORTER"
    elif len(record.seq) >= configuration["PREDICTION"].getint("maxseqlen"):
        classification["tag"] = "LONGER"
    else:
        classification["tag"] = "ACCEPTED"

    if do_prediction:
        # load model
        predictor = deeplasmid_model.Model(configuration)
        for (key, path) in configuration.items("MODELS"):
            predictor.model.append(keras.models.load_model(path, compile=True))

        # call prediction
        #predictions = predictor.predict(sample_list_1hot, normalized_features_vector)
        this_subsequences1hot = numpy.array(sample_list_1hot, dtype=numpy.float32)
        features_x_subsequences = numpy.array([normalized_features_vector] * len(sample_list_1hot), dtype=numpy.float32)
        predictions = predictor.predict(this_subsequences1hot, features_x_subsequences)

        # determine score
        predictions_numpy = numpy.array([prediction.flatten() for prediction in predictions])
        mean_per_model_scores = numpy.mean(predictions_numpy, axis=1)
        mean_over_models_scores = numpy.mean(predictions_numpy, axis=0)
        score = deeplasmid_utils.get_statistics(mean_over_models_scores)  # score dict
        # store classification related information
        classification["predictions"] = [prediction.tolist() for prediction in predictions]
        classification["mean_per_model_scores"] = mean_per_model_scores.tolist()
        classification["mean_over_models_scores"] = mean_over_models_scores.tolist()
        classification["score"] = copy.deepcopy(score)
        # label data according to score
        # if score['avr'] > configuration['PREDICTION'].getfloat('score_threshold_plasmid'):
        #     classification['label'] = 'PLASMID'
        # elif score['avr'] < configuration['PREDICTION'].getfloat('score_threshold_genome'):
        #     classification['label'] = 'GENOME'
        # else:
        #     classification['label'] = 'UNCLASSIFIED'

        # if score['avr'] > configuration['PREDICTION'].getfloat('score_threshold') + score['std']*2:
        #     classification['label'] = 'PLASMID'
        # elif score['avr'] < configuration['PREDICTION'].getfloat('score_threshold') - score['std']*2:
        #     classification['label'] = 'GENOME'
        # else:
        #     classification['label'] = 'AMBIGUOUS'

        if score["avr"] > configuration["PREDICTION"].getfloat("score_threshold") + score["err"] * 2:
            classification["label"] = "PLASMID"
        elif score["avr"] < configuration["PREDICTION"].getfloat("score_threshold") - score["err"] * 2:
            classification["label"] = "GENOME"
        else:
            classification["label"] = "AMBIGUOUS"
    else:
        classification["predictions"] = []
        classification["mean_per_model_scores"] = []
        classification["mean_over_models_scores"] = []
        classification["score"] = copy.deepcopy(deeplasmid_utils.get_statistics([]))
        classification["label"] = "NONE"

    data["classification"] = classification

    # store sequence meta information
    with gzip.open(output_yaml_file, "wt") as yaml_file_handle:
        yaml.dump(data, yaml_file_handle, default_flow_style=False, sort_keys=False)

    # logger.info("Process " + str(this_id) + ": Processing THE END")
    # logger.removeHandler(fh)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plasmid Detection", prog=os.path.basename(__file__), formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80))
    parser.add_argument("-l", "--log", dest="log", help="Provide logging level. Example --log=DEBUG', default='WARNING'", default="WARNING", required=False)
    parser.add_argument("-n", "--ncpus", dest="ncpus", help="Number of cpus to use, set to smaller value if > system number of cpus", default=argparse.SUPPRESS, required=False)
    parser.add_argument("-m", "--nsamples", dest="nsamples", help="Number of samples to analyze", default=argparse.SUPPRESS, required=False)
    mutually_exclusive = parser.add_mutually_exclusive_group(required=True)
    mutually_exclusive.add_argument("-t", "--train", dest="do_train", action="store_true", help="Run in training mode")
    mutually_exclusive.add_argument("-e", "--train_only", dest="do_train_only", action="store_true", help="Run training only")
    mutually_exclusive.add_argument("-p", "--predict", dest="do_predict", action="store_true", help="Run in prediction mode")
    mutually_exclusive.add_argument("-f", "--features", dest="do_features", action="store_true", help="Calculate features only")
    mutually_exclusive.add_argument("-s", "--summarize", dest="do_summarize", action="store_true", help="Produce summary tsv file")
    required = parser.add_argument_group("required arguments")
    required.add_argument("-i", "--input", type=str, dest="input", help="Input fasta file", required=True)
    required.add_argument("-c", "--config", type=str, dest="config_file", help="Configuration file", required=True)
    required.add_argument("-o", "--output", type=str, dest="output", help="Output directory", required=True)
    options = parser.parse_args()

    if not os.path.isdir(options.output):
        print(f"<I> deeplasmid: Cannot find {options.output}, creating as new")
        os.makedirs(options.output)

    if not os.path.isfile(options.input):
        print(f"<E> deeplasmid: {options.input} does not exist!")
        sys.exit(1)

    if not os.path.isfile(options.config_file):
        print(f"<E> deeplasmid: {options.config_file} does not exist!")
        sys.exit(1)

    config = configparser.ConfigParser()
    configfile = config.read(options.config_file)

    # store command line options in configuration
    # configuration is stored later in yaml fle
    # that way we can keep track of command line options used for this analysis
    config["LOGGING"]["output_directory"] = options.output

    # print configuration
    for section in config.sections():
        for item in config[section]:
            print(section, item, config[section][item])

    # for (item, feature) in config.items('MODELS'):
    #     print(" key, feature = %s, %s" % (item, feature))

    # for (item, feature) in config.items('FEATURES'):
    #     print(" key, feature = %s, %s" % (item, feature))

    if options.do_features or options.do_train or options.do_predict:
        # split multi fasta input file into single fasta files
        # check the single fasta file record:
        # 1) id: replace all characters not valid for unix file names with underscore (_)
        # 2) seq: replace all bases not ACTG (e.g. ambiguity characters) with N
        # Note: This code only deals with the 5 characters ACTGN
        encoding = guess_type(options.input)[1]  # uses file extension
        _open = partial(gzip.open, mode="rt") if encoding == "gzip" else open
        with _open(options.input) as input_fasta_handle:
            for original_record in SeqIO.parse(input_fasta_handle, "fasta"):
                new_record = original_record
                new_record.id = re.sub(r"[^a-zA-Z0-9._-]", "_", original_record.id)
                new_record.seq = original_record.seq.upper()
                new_record.seq = Seq.Seq(re.sub("[^ACGT]", "N", str(original_record.seq)))
                file_name = os.path.join(options.output, new_record.id + ".fa")
                with open(file_name, "w", encoding="ascii") as output_fasta_handle:
                    SeqIO.write(new_record, output_fasta_handle, "fasta")

        # get all the fasta files and process each
        input_fasta_list_all = [str(path) for path in pathlib.Path(options.output).glob("*.fa")]
        if "nsamples" in options:
            input_fasta_list = input_fasta_list_all[0 : options.nsamples]  # will take all if nsamples > len(list)
        else:
            input_fasta_list = input_fasta_list_all

        if not config["LOGGING"].getboolean("do_multiprocessing"):
            for fasta_file_name in input_fasta_list:
                YAML_FILE_NAME = str(pathlib.Path(fasta_file_name).with_suffix(".yml.gz"))
                process_sequence(fasta_file_name, config, options.do_predict, YAML_FILE_NAME)
        else:
            # parallel version of loop above
            logger = pathos.logger(level=options.log)
            logger.disabled = config["LOGGING"].getboolean("turn_off_logging")
            if "ncpus" in options:
                PROCESSES = min(pathos.multiprocessing.cpu_count(), config["LOGGING"].getint("number_of_cpus"), len(input_fasta_list), options.ncpus)
            else:
                PROCESSES = min(pathos.multiprocessing.cpu_count(), config["LOGGING"].getint("number_of_cpus"), len(input_fasta_list))
            # print("<I> deeplasmid: number of processes = %s" % (PROCESSES))
            # print("<I> deeplasmid: number of workers = %s" % (len(input_fasta_list)))
            for chunk_of_fasta_files in deeplasmid_utils.chunker(input_fasta_list, PROCESSES):
                chunk_of_yaml_files = [str(pathlib.Path(fasta_file_name).with_suffix(".yml.gz")) for fasta_file_name in chunk_of_fasta_files]
                with pathos.pools.ProcessPool(nodes=PROCESSES, timeout=20) as p:
                    results = p.map(process_sequence, chunk_of_fasta_files, [config] * len(chunk_of_fasta_files), [options.do_predict] * len(chunk_of_fasta_files), chunk_of_yaml_files)

        # clean up
        if config["LOGGING"].getboolean("delete_fasta_files"):
            for file_name in input_fasta_list:
                os.remove(file_name)

        if config["LOGGING"].getboolean("delete_faa_files"):
            amino_acid_files = [str(pathlib.Path(file_name).with_suffix(".faa")) for file_name in input_fasta_list]
            for file_name in amino_acid_files:
                os.remove(file_name)

    # collect data and write prediction csv file
    if options.do_summarize or options.do_predict:
        with open(os.path.join(options.output, config["LOGGING"]["summary_file_name"]), "w", encoding="ascii") as csv_handle:
            column_names = ["contig_name", "label", "tag", "truth", "score", "error", "std", "median", "number_of_samples"]
            feature_names = ["gc_content", "len_sequence", "genecount", "genesperMB", "aalenavg", "A_longestHomopol", "A_totalLongHomopol", "C_longestHomopol", "C_totalLongHomopol", "T_longestHomopol", "T_totalLongHomopol", "G_longestHomopol", "G_totalLongHomopol", "plassketch", "plasORIsketch", "chromsketch"]
            column_names.extend(feature_names)
            column_names.extend(("normalized_" + feature) for feature in feature_names)
            writer = csv.writer(csv_handle, delimiter="\t")
            writer.writerow(column_names)

            input_yml = [str(path) for path in pathlib.Path(options.output).glob("*.yml.gz")]
            for yaml_file in input_yml:
                with gzip.open(yaml_file, "rt", encoding="ascii") as f:
                    sample_data = yaml.load(f, Loader=yaml.CLoader)

                row = [sample_data["sequence"]["id"]]
                row.append(sample_data["classification"]["label"])
                row.append(sample_data["classification"]["tag"])
                row.append(sample_data["sequence"]["truth"])
                row.append(sample_data["classification"]["score"]["avr"])
                row.append(sample_data["classification"]["score"]["err"])
                row.append(sample_data["classification"]["score"]["std"])
                row.append(sample_data["classification"]["score"]["median"])
                row.append(sample_data["classification"]["model_input"]["number_of_samples"])
                row.extend(sample_data["features"].values())
                row.extend(sample_data["normalized_features"].values())
                if len(row) == len(column_names):
                    writer.writerow(row)
                else:
                    print("<W> deeplasmid: length of row != number of column names!")

    # training here
    if options.do_train or options.do_train_only:
        # create dedicated training directory
        training_dir = os.path.join(options.output, config["TRAINING"]["training_output_directory"])
        os.makedirs(training_dir, exist_ok=True)

        # fill internal model data structure
        # get features from yaml file
        training_data = {"training": [[], [], [], []], "validation": [[], [], [], []]}

        input_yaml_list = [str(path) for path in pathlib.Path(options.output).glob("*.yml.gz")]
        for yaml_file in input_yaml_list:
            with gzip.open(yaml_file, "rt", encoding="ascii") as f:
                sample_data = yaml.load(f, Loader=yaml.CLoader)

            TRUTH_LABEL = 1.0 if sample_data["sequence"]["truth"] == "PLASMID" else 0.0
            normalized_features = sample_data["classification"]["model_input"]["normalized_features"]
            subsequences = sample_data["classification"]["model_input"]["sample_list"]
            subsequences1hot = sample_data["classification"]["model_input"]["sample_list_1hot"]
            training_label = sample_data["sequence"]["training_label"]

            training_data[training_label][0].extend(subsequences)
            training_data[training_label][1].extend([normalized_features] * len(subsequences))
            training_data[training_label][2].extend([TRUTH_LABEL] * len(subsequences))
            training_data[training_label][3].extend(subsequences1hot)

        if len(input_yaml_list) > 0:
            # instantiate model object
            trainer = deeplasmid_model.Model(config)

            # load data into model internal numpy arrays
            # 1hot encode sequences
            trainer.labelsXsubsequences_trn = numpy.array(training_data["training"][2], dtype=numpy.float32)
            trainer.labelsXsubsequences_val = numpy.array(training_data["validation"][2], dtype=numpy.float32)
            trainer.featuresXsubsequences_trn = numpy.array(training_data["training"][1], dtype=numpy.float32)
            trainer.featuresXsubsequences_val = numpy.array(training_data["validation"][1], dtype=numpy.float32)
            trainer.subsequences1hot_trn = numpy.array([deeplasmid_utils.encode1hot(subsequence) for subsequence in training_data["training"][0]], dtype=numpy.float32)
            trainer.subsequences1hot_val = numpy.array([deeplasmid_utils.encode1hot(subsequence) for subsequence in training_data["validation"][0]], dtype=numpy.float32)
            trainer.build()  # create the model
            trainer.model.summary()  # print architecture of model
            HISTORY = trainer.train()  # training

            # save training results
            # setup some file names
            model_file = os.path.join(options.output, config["TRAINING"]["training_output_directory"], config["TRAINING"]["model_file"])
            weights_file = os.path.join(options.output, config["TRAINING"]["training_output_directory"], config["TRAINING"]["weights_file"])
            plot_file = os.path.join(options.output, config["TRAINING"]["training_output_directory"], config["TRAINING"]["plot_file"])
            json_file = os.path.join(options.deeplasmid.pyoutput, config["TRAINING"]["training_output_directory"], config["TRAINING"]["json_file"])
            training_history_file = os.path.join(options.output, config["TRAINING"]["training_output_directory"], config["TRAINING"]["training_history_file"])

            # store
            trainer.model.save(model_file)
            trainer.model.save_weights(weights_file)
            #keras.utils.plot_model(trainer.model, to_file=plot_file, show_layer_names=False, show_shapes=True)
            with open(json_file, "w", encoding="ascii") as f:
                #f.write(trainer.model.to_yaml()) # removed in tensorflow 2.6.0
                f.write(trainer.model.to_json())
            with open(training_history_file, "w", encoding="ascii") as f:
                yaml.dump(HISTORY.history, f, default_flow_style=False, sort_keys=False)
