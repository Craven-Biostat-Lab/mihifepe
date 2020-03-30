"""Generates simulated data and model to test mihifepe algorithm"""

import argparse
from collections import namedtuple
import csv
import functools
import itertools
import os
import pickle
import sys
from unittest.mock import patch

import anytree
from anytree.importer import JsonImporter
import h5py
import numpy as np
from scipy.cluster.hierarchy import linkage
import sympy
from sympy.utilities.lambdify import lambdify
from sklearn.metrics import precision_recall_fscore_support

from mihifepe import constants, master, utils
from mihifepe.fdr import hierarchical_fdr_control

# TODO maybe: write arguments to separate readme.txt for documentating runs

# Simulation results object
Results = namedtuple(constants.SIMULATION_RESULTS, [constants.FDR, constants.POWER,
                                                    constants.OUTER_NODES_FDR, constants.OUTER_NODES_POWER,
                                                    constants.BASE_FEATURES_FDR, constants.BASE_FEATURES_POWER,
                                                    constants.INTERACTIONS_FDR, constants.INTERACTIONS_POWER])


def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=constants.SEED)
    parser.add_argument("-num_instances", type=int, default=10000)
    parser.add_argument("-num_features", type=int, default=100)
    parser.add_argument("-output_dir", help="Name of output directory")
    parser.add_argument("-fraction_relevant_features", type=float, default=.05)
    parser.add_argument("-noise_multiplier", type=float, default=.05,
                        help="Multiplicative factor for noise added to polynomial computation for irrelevant features")
    parser.add_argument("-noise_type", choices=[constants.ADDITIVE_GAUSSIAN, constants.EPSILON_IRRELEVANT, constants.NO_NOISE],
                        default=constants.EPSILON_IRRELEVANT)
    parser.add_argument("-hierarchy_type", help="Choice of hierarchy to generate", default=constants.CLUSTER_FROM_DATA,
                        choices=[constants.CLUSTER_FROM_DATA, constants.RANDOM])
    parser.add_argument("-clustering_instance_count", type=int, help="If provided, uses this number of instances to "
                        "cluster the data to generate a hierarchy, allowing the hierarchy to remain same across multiple "
                        "sets of instances", default=0)
    parser.add_argument("-num_interactions", type=int, default=0, help="number of interaction pairs in model")
    parser.add_argument("-exclude_interaction_only_features", help="exclude interaction-only features in model"
                        " in addition to linear + interaction features (default included)", action="store_false",
                        dest="include_interaction_only_features")
    parser.set_defaults(include_interaction_only_features=True)
    parser.add_argument("-contiguous_node_names", action="store_true", help="enable to change node names in hierarchy "
                        "to be contiguous for better visualization (but creating mismatch between node names and features indices)")
    # Arguments used to qualify output directory, then passed to mihifepe.master
    parser.add_argument("-perturbation", default=constants.SHUFFLING, choices=[constants.ZEROING, constants.SHUFFLING])
    parser.add_argument("-num_shuffling_trials", type=int, default=100, help="Number of shuffling trials to average over, "
                        "when shuffling perturbations are selected")
    parser.add_argument("-analyze_interactions", help="enable analyzing interactions", action="store_true")

    args, pass_args = parser.parse_known_args()
    pass_args = " ".join(pass_args)
    if not args.output_dir:
        args.output_dir = ("sim_outputs_inst_%d_feat_%d_noise_%.3f_relfraction_%.3f_pert_%s_shufftrials_%d" %
                           (args.num_instances, args.num_features, args.noise_multiplier,
                            args.fraction_relevant_features, args.perturbation, args.num_shuffling_trials))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.RandomState(args.seed)
    args.logger = utils.get_logger(__name__, "%s/simulation.log" % args.output_dir)

    pipeline(args, pass_args)


def pipeline(args, pass_args):
    """Simulation pipeline"""
    # TODO: Features other than binary
    args.logger.info("Begin mihifepe simulation with args: %s" % args)
    # Synthesize polynomial that generates ground truth
    sym_vars, relevant_feature_map, polynomial_fn = gen_polynomial(args, get_relevant_features(args))
    # Synthesize data
    probs, test_data, clustering_data = synthesize_data(args)
    # Generate hierarchy using clustering
    hierarchy_root, feature_id_map = gen_hierarchy(args, clustering_data)
    # Update hierarchy descriptions for future visualization
    update_hierarchy_relevance(hierarchy_root, relevant_feature_map, probs)
    # Generate targets (ground truth)
    targets = gen_targets(polynomial_fn, test_data)
    # Write outputs - data, gen_model.py, hierarchy
    data_filename = write_data(args, test_data, targets)
    hierarchy_filename = write_hierarchy(args, hierarchy_root)
    gen_model_filename = write_model(args, sym_vars)
    # Invoke feature importance algorithm
    run_mihifepe(args, pass_args, data_filename, hierarchy_filename, gen_model_filename)
    # Compare mihifepe outputs with ground truth outputs
    compare_with_ground_truth(args, hierarchy_root)
    # Evaluate mihifepe outputs - power/FDR for all nodes/outer nodes/base features
    results = evaluate(args, relevant_feature_map, feature_id_map)
    args.logger.info("Results:\n%s" % str(results))
    write_results(args, results)
    args.logger.info("End mihifepe simulation")


def synthesize_data(args):
    """Synthesize data"""
    # TODO: Correlations between features
    args.logger.info("Begin generating data")
    probs = args.rng.uniform(size=args.num_features)
    data = args.rng.binomial(1, probs, size=(max(args.num_instances, args.clustering_instance_count), args.num_features))
    test_data = data
    clustering_data = data
    if args.clustering_instance_count:
        clustering_data = data[:args.clustering_instance_count, :]
        if args.clustering_instance_count > args.num_instances:
            test_data = data[:args.num_instances, :]
    args.logger.info("End generating data")
    return probs, test_data, clustering_data


def gen_hierarchy(args, clustering_data):
    """
    Generate hierarchy over features

    Args:
        args: Command-line arguments
        clustering_data: Data potentially used to cluster features
                         (depending on hierarchy generation method)

    Returns:
        hierarchy_root: root fo resulting hierarchy over features
    """
    # Generate hierarchy
    hierarchy_root = None
    if args.hierarchy_type == constants.CLUSTER_FROM_DATA:
        clusters = cluster_data(args, clustering_data)
        hierarchy_root = gen_hierarchy_from_clusters(args, clusters)
    elif args.hierarchy_type == constants.RANDOM:
        hierarchy_root = gen_random_hierarchy(args)
    else:
        raise NotImplementedError("Need valid hierarchy type")
    # Improve visualization - contiguous feature names
    feature_id_map = {}  # mapping from visual feature ids to original ids
    if args.contiguous_node_names:
        for idx, node in enumerate(anytree.PostOrderIter(hierarchy_root)):
            node.idx = idx
            if node.is_leaf:
                node.min_child_idx = idx
                node.max_child_idx = idx
                node.num_base_features = 1
                node.name = str(idx)
                feature_id_map[idx] = int(node.static_indices)
            else:
                node.min_child_idx = min([child.min_child_idx for child in node.children])
                node.max_child_idx = max([child.idx for child in node.children])
                node.num_base_features = sum([child.num_base_features for child in node.children])
                node.name = "[%d-%d] (size: %d)" % (node.min_child_idx, node.max_child_idx, node.num_base_features)
    return hierarchy_root, feature_id_map


def gen_random_hierarchy(args):
    """Generates balanced random hierarchy"""
    args.logger.info("Begin generating hierarchy")
    nodes = [anytree.Node(str(idx), static_indices=str(idx)) for idx in range(args.num_features)]
    args.rng.shuffle(nodes)
    node_count = len(nodes)
    while len(nodes) > 1:
        parents = []
        for left_idx in range(0, len(nodes), 2):
            parent = anytree.Node(str(node_count))
            node_count += 1
            nodes[left_idx].parent = parent
            right_idx = left_idx + 1
            if right_idx < len(nodes):
                nodes[right_idx].parent = parent
            parents.append(parent)
        nodes = parents
    hierarchy_root = nodes[0]
    args.logger.info("End generating hierarchy")
    return hierarchy_root


def cluster_data(args, data):
    """Cluster data using hierarchical clustering with Hamming distance"""
    # Cluster data
    args.logger.info("Begin clustering data")
    clusters = linkage(data.transpose(), metric="hamming", method="complete")
    args.logger.info("End clustering data")
    return clusters


def gen_hierarchy_from_clusters(args, clusters):
    """
    Organize clusters into hierarchy

    Args:
        clusters: linkage matrix (num_features-1 X 4)
                  rows indicate successive clustering iterations
                  columns, respectively: 1st cluster index, 2nd cluster index, distance, sample count
    Returns:
        hierarchy_root: root of resulting hierarchy over features
    """
    # Generate hierarchy from clusters
    nodes = [anytree.Node(str(idx), static_indices=str(idx)) for idx in range(args.num_features)]
    for idx, cluster in enumerate(clusters):
        cluster_idx = idx + args.num_features
        left_idx, right_idx, _, _ = cluster
        left_idx = int(left_idx)
        right_idx = int(right_idx)
        cluster_node = anytree.Node(str(cluster_idx))
        nodes[left_idx].parent = cluster_node
        nodes[right_idx].parent = cluster_node
        nodes.append(cluster_node)
    hierarchy_root = nodes[-1]
    return hierarchy_root


def gen_polynomial(args, relevant_features):
    """Generate polynomial which decides the ground truth and noisy model"""
    # Note: using sympy to build function appears to be 1.5-2x slower than erstwhile raw numpy implementation (for linear terms)
    # TODO: possibly negative coefficients
    sym_features = sympy.symbols(["x%d" % x for x in range(args.num_features)])
    relevant_feature_map = {}  # map of relevant feature sets to coefficients
    # Generate polynomial expression
    # Pairwise interaction terms
    sym_polynomial_fn = 0
    sym_polynomial_fn = update_interaction_terms(args, relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn)
    # Linear terms
    sym_polynomial_fn = update_linear_terms(args, relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn)
    args.logger.info("Ground truth polynomial:\ny = %s" % sym_polynomial_fn)
    # Generate model expression
    polynomial_fn = lambdify([sym_features], sym_polynomial_fn, "numpy")
    # Add noise terms
    sym_noise = []
    sym_model_fn = sym_polynomial_fn
    if args.noise_type == constants.NO_NOISE:
        pass
    elif args.noise_type == constants.EPSILON_IRRELEVANT:
        sym_noise = sympy.symbols(["noise%d" % x for x in range(args.num_features)])
        irrelevant_features = np.array([0 if x in relevant_features else 1 for x in range(args.num_features)])
        sym_model_fn = sym_polynomial_fn + (sym_noise * irrelevant_features).dot(sym_features)
    elif args.noise_type == constants.ADDITIVE_GAUSSIAN:
        sym_noise = sympy.symbols("noise")
        sym_model_fn = sym_polynomial_fn + sym_noise
    else:
        raise NotImplementedError("Unknown noise type")
    sym_vars = (sym_features, sym_noise, sym_model_fn)
    return sym_vars, relevant_feature_map, polynomial_fn


def get_relevant_features(args):
    """Get set of relevant feature identifiers"""
    num_relevant_features = max(1, round(args.num_features * args.fraction_relevant_features))
    coefficients = np.zeros(args.num_features)
    coefficients[:num_relevant_features] = 1
    args.rng.shuffle(coefficients)
    relevant_features = {idx for idx in range(args.num_features) if coefficients[idx]}
    return relevant_features


def update_interaction_terms(args, relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn):
    """Pairwise interaction terms for polynomial"""
    # TODO: higher-order interactions
    num_relevant_features = len(relevant_features)
    num_interactions = min(args.num_interactions, num_relevant_features * (num_relevant_features - 1) / 2)
    if not num_interactions:
        return sym_polynomial_fn
    potential_pairs = list(itertools.combinations(sorted(relevant_features), 2))
    potential_pairs_arr = np.empty(len(potential_pairs), dtype=np.object)
    potential_pairs_arr[:] = potential_pairs
    interaction_pairs = args.rng.choice(potential_pairs_arr, size=num_interactions, replace=False)
    for interaction_pair in interaction_pairs:
        coefficient = args.rng.uniform()
        relevant_feature_map[frozenset(interaction_pair)] = coefficient
        sym_polynomial_fn += coefficient * functools.reduce(lambda sym_x, y: sym_x * sym_features[y], interaction_pair, 1)
    return sym_polynomial_fn


def update_linear_terms(args, relevant_features, relevant_feature_map, sym_features, sym_polynomial_fn):
    """Order one terms for polynomial"""
    interaction_features = set()
    for interaction in relevant_feature_map.keys():
        interaction_features.update(interaction)
    # Let half the interaction features have nonzero interaction coefficients but zero linear coefficients
    interaction_only_features = []
    if interaction_features and args.include_interaction_only_features:
        interaction_only_features = args.rng.choice(sorted(interaction_features),
                                                    len(interaction_features) // 2,
                                                    replace=False)
    linear_features = sorted(relevant_features.difference(interaction_only_features))
    coefficients = np.zeros(args.num_features)
    coefficients[linear_features] = args.rng.uniform(size=len(linear_features))
    for linear_feature in linear_features:
        relevant_feature_map[frozenset([linear_feature])] = coefficients[linear_feature]
    sym_polynomial_fn += coefficients.dot(sym_features)
    return sym_polynomial_fn


def update_hierarchy_relevance(hierarchy_root, relevant_feature_map, probs):
    """
    Add feature relevance information to nodes of hierarchy:
    their probabilty of being enabled,
    their polynomial coefficient
    """
    relevant_features = set()
    for key in relevant_feature_map:
        relevant_features.update(key)
    for node in anytree.PostOrderIter(hierarchy_root):
        node.description = constants.IRRELEVANT
        if node.is_leaf:
            idx = int(node.static_indices)
            node.poly_coeff = 0.0
            node.bin_prob = probs[idx]
            coeff = relevant_feature_map.get(frozenset([idx]))
            if coeff:
                node.poly_coeff = coeff
                node.description = ("%s feature:\nPolynomial coefficient: %f\nBinomial probability: %f"
                                    % (constants.RELEVANT, coeff, probs[idx]))
            elif idx in relevant_features:
                node.description = ("%s feature\n(Interaction-only)" % constants.RELEVANT)
        else:
            for child in node.children:
                if child.description != constants.IRRELEVANT:
                    node.description = constants.RELEVANT


def gen_targets(polynomial_fn, data):
    """Generate targets (ground truth) from polynomial"""
    return [polynomial_fn(instance) for instance in data]


def write_data(args, data, targets):
    """
    Write data in HDF5 format.

    Groups:
        /temporal               (Group containing temporal data)

    Datasets:
        /record_ids             (List of record identifiers (strings) of length M = number of records/instances)
        /targets                (vector of target values (regression/classification outputs) of length M)
        /static                 (matrix of static data of size M x L)
        /temporal/<record_id>   (One dataset per record_id) (List (of variable length V) of vectors (of fixed length W))
    """
    data_filename = "%s/%s" % (args.output_dir, "data.hdf5")
    root = h5py.File(data_filename, "w")
    record_ids = [str(idx).encode("utf8") for idx in range(args.num_instances)]
    root.create_dataset(constants.RECORD_IDS, data=record_ids)
    root.create_dataset(constants.TARGETS, data=targets)
    root.create_dataset(constants.STATIC, data=data)
    root.close()
    return data_filename


def write_hierarchy(args, hierarchy_root):
    """
    Write hierarchy in CSV format.

    Columns:    *name*:             feature name, must be unique across features
                *parent_name*:      name of parent if it exists, else '' (root node)
                *description*:      node description
                *static_indices*:   [only required for leaf nodes] list of tab-separated indices corresponding to the indices
                                    of these features in the static data
                *temporal_indices*: [only required for leaf nodes] list of tab-separated indices corresponding to the indices
                                    of these features in the temporal data
    """
    hierarchy_filename = "%s/%s" % (args.output_dir, "hierarchy.csv")
    with open(hierarchy_filename, "w", newline="") as hierarchy_file:
        writer = csv.writer(hierarchy_file, delimiter=",")
        writer.writerow([constants.NODE_NAME, constants.PARENT_NAME,
                         constants.DESCRIPTION, constants.STATIC_INDICES, constants.TEMPORAL_INDICES])
        for node in anytree.PreOrderIter(hierarchy_root):
            static_indices = node.static_indices if node.is_leaf else ""
            parent_name = node.parent.name if node.parent else ""
            writer.writerow([node.name, parent_name, node.description, static_indices, ""])
    return hierarchy_filename


def write_model(args, sym_vars):
    """
    Write model to file in output directory.
    Write model_filename to config file in script directory.
    gen_model.py uses config file to load model.
    """
    # Write model to file
    model_filename = "%s/%s" % (args.output_dir, constants.MODEL_FILENAME)
    with open(model_filename, "wb") as model_file:
        pickle.dump(sym_vars, model_file)
    # Write model_filename to config
    gen_model_config_filename = "%s/%s" % (args.output_dir, constants.GEN_MODEL_CONFIG_FILENAME)
    with open(gen_model_config_filename, "wb") as gen_model_config_file:
        pickle.dump(model_filename, gen_model_config_file)
        pickle.dump(args.noise_multiplier, gen_model_config_file)
        pickle.dump(args.noise_type, gen_model_config_file)
    # Write gen_model.py to output_dir
    gen_model_filename = "%s/%s" % (args.output_dir, constants.GEN_MODEL_FILENAME)
    gen_model_template_filename = "%s/%s" % (os.path.dirname(os.path.abspath(__file__)), constants.GEN_MODEL_TEMPLATE_FILENAME)
    gen_model_file = open(gen_model_filename, "w")
    with open(gen_model_template_filename, "r") as gen_model_template_file:
        for line in gen_model_template_file:
            line = line.replace(constants.GEN_MODEL_CONFIG_FILENAME_PLACEHOLDER, gen_model_config_filename)
            gen_model_file.write(line)
    gen_model_file.close()
    return gen_model_filename


def run_mihifepe(args, pass_args, data_filename, hierarchy_filename, gen_model_filename):
    """Run mihifepe algorithm"""
    args.logger.info("Begin running mihifepe")
    analyze_interactions = "-analyze_interactions" if args.analyze_interactions else ""
    args.logger.info("Passing the following arguments to mihifepe.master without parsing: %s" % pass_args)
    memory_requirement = 1 + (os.stat(data_filename).st_size // (2 ** 30))  # Compute approximate memory requirement in GB
    cmd = ("python -m mihifepe.master -data_filename %s -hierarchy_filename %s -model_generator_filename %s -output_dir %s "
           "-perturbation %s -num_shuffling_trials %d -memory_requirement %d %s %s"
           % (data_filename, hierarchy_filename, gen_model_filename, args.output_dir,
              args.perturbation, args.num_shuffling_trials, memory_requirement, analyze_interactions, pass_args))
    args.logger.info("Running cmd: %s" % cmd)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        master.main()
    args.logger.info("End running mihifepe")


def compare_with_ground_truth(args, hierarchy_root):
    """Compare results from mihifepe with ground truth results"""
    # Generate ground truth results
    # Write hierarchical FDR input file for ground truth values
    args.logger.info("Compare mihifepe results to ground truth")
    input_filename = "%s/ground_truth_pvalues.csv" % args.output_dir
    with open(input_filename, "w", newline="") as input_file:
        writer = csv.writer(input_file)
        writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.PVALUE_LOSSES, constants.DESCRIPTION])
        for node in anytree.PostOrderIter(hierarchy_root):
            parent_name = node.parent.name if node.parent else ""
            # Decide p-values based on rough heuristic for relevance
            node.pvalue = 1.0
            if node.description != constants.IRRELEVANT:
                if node.is_leaf:
                    node.pvalue = 0.001
                    if node.poly_coeff:
                        node.pvalue = min(node.pvalue, 1e-10 / (node.poly_coeff * node.bin_prob) ** 3)
                else:
                    node.pvalue = 0.999 * min([child.pvalue for child in node.children])
            writer.writerow([node.name, parent_name, node.pvalue, node.description])
    # Generate hierarchical FDR results for ground truth values
    ground_truth_dir = "%s/ground_truth_fdr" % args.output_dir
    cmd = ("python -m mihifepe.fdr.hierarchical_fdr_control -output_dir %s -procedure yekutieli "
           "-rectangle_leaves %s" % (ground_truth_dir, input_filename))
    args.logger.info("Running cmd: %s" % cmd)
    pass_args = cmd.split()[2:]
    with patch.object(sys, 'argv', pass_args):
        hierarchical_fdr_control.main()
    # Compare results
    ground_truth_outputs_filename = "%s/%s.png" % (ground_truth_dir, constants.TREE)
    args.logger.info("Ground truth results: %s" % ground_truth_outputs_filename)
    mihifepe_outputs_filename = "%s/%s/%s.png" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR, constants.TREE)
    args.logger.info("mihifepe results: %s" % mihifepe_outputs_filename)


def evaluate(args, relevant_feature_map, feature_id_map):
    """
    Evaluate mihifepe results - obtain power/FDR measures for all nodes/outer nodes/base features/interactions
    """
    # pylint: disable = too-many-locals
    def get_relevant_rejected(nodes, outer=False, leaves=False):
        """Get set of relevant and rejected nodes"""
        assert not (outer and leaves)
        if outer:
            nodes = [node for node in nodes if node.rejected and all([not child.rejected for child in node.children])]
        elif leaves:
            nodes = [node for node in nodes if node.is_leaf]
        relevant = [0 if node.description == constants.IRRELEVANT else 1 for node in nodes]
        rejected = [1 if node.rejected else 0 for node in nodes]
        return relevant, rejected

    tree_filename = "%s/%s/%s.json" % (args.output_dir, constants.HIERARCHICAL_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    with open(tree_filename, "r") as tree_file:
        tree = JsonImporter().read(tree_file)
        nodes = list(anytree.PreOrderIter(tree))
        # All nodes FDR/power
        relevant, rejected = get_relevant_rejected(nodes)
        precision, recall, _, _ = precision_recall_fscore_support(relevant, rejected, average="binary")
        # Outer nodes FDR/power
        outer_relevant, outer_rejected = get_relevant_rejected(nodes, outer=True)
        outer_precision, outer_recall, _, _ = precision_recall_fscore_support(outer_relevant, outer_rejected, average="binary")
        # Base features FDR/power
        bf_relevant, bf_rejected = get_relevant_rejected(nodes, leaves=True)
        bf_precision, bf_recall, _, _ = precision_recall_fscore_support(bf_relevant, bf_rejected, average="binary")
        # Interactions FDR/power
        interaction_precision, interaction_recall = get_precision_recall_interactions(args, relevant_feature_map, feature_id_map)

        return Results(1 - precision, recall, 1 - outer_precision, outer_recall,
                       1 - bf_precision, bf_recall, 1 - interaction_precision, interaction_recall)


def get_precision_recall_interactions(args, relevant_feature_map, feature_id_map):
    """Computes precision (1 - FDR) and recall (power) for detecting interactions"""
    # pylint: disable = invalid-name, too-many-locals
    # The set of all possible interactions might be very big, so don't construct label vector for all
    # possible interactions - compute precision/recall from basics
    # TODO: alter to handle higher-order interactions
    if not args.analyze_interactions:
        return (0.0, 0.0)
    true_interactions = {key for key in relevant_feature_map.keys() if len(key) > 1}
    tree_filename = "%s/%s/%s.json" % (args.output_dir, constants.INTERACTIONS_FDR_DIR, constants.HIERARCHICAL_FDR_OUTPUTS)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tested = set()
    with open(tree_filename, "r") as tree_file:
        tree = JsonImporter().read(tree_file)
        # Two-level tree with tested interactions on level 2
        for node in tree.children:
            pair = frozenset({int(idx) for idx in node.name.split(" + ")})
            if feature_id_map:
                pair = frozenset({feature_id_map[visual_id] for visual_id in pair})
            tested.add(pair)
            if node.rejected:
                if relevant_feature_map.get(pair):
                    tp += 1
                else:
                    fp += 1
            else:
                if relevant_feature_map.get(pair):
                    fn += 1
                else:
                    tn += 1
    if not tp > 0:
        return (0.0, 0.0)
    missed = true_interactions.difference(tested)
    fn += len(missed)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def write_results(args, results):
    """Write results to pickle file"""
    results_filename = "%s/%s" % (args.output_dir, constants.SIMULATION_RESULTS_FILENAME)
    with open(results_filename, "wb") as results_file:
        pickle.dump(results._asdict(), results_file)


if __name__ == "__main__":
    main()
