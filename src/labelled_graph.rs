use itertools::Itertools;

use crate::graph::{Graph, NeighborMap, PartialGraph};
use crate::permutable::{Automorphisms, GroupPermutable, GroupPermutableFromFused, Orbit, Orbits, Permutable, Permutation, PermutationGroup};

/// A graph with natural numbers as node labels.
pub type NodeLabelledGraph = Graph<usize>;
/// A graph with natural numbers as edge and node labels.
pub type EdgeLabelledGraph = Graph<usize, usize>;

impl GroupPermutableFromFused for NodeLabelledGraph {
    fn auts(&self) -> Automorphisms {
        self.unlabelled_automorphisms(Some(self.node_labels().collect_vec().orbits()))
    }
}

impl EdgeLabelledGraph {
    /// Equivalent graph where labelled edges are converted into labelled nodes.
    pub fn with_edges_as_nodes(&self) -> NodeLabelledGraph {
        let edge_label_shift = self.node_labels().max().unwrap_or(0) + 1;
        let mut extended = NodeLabelledGraph::new(self.node_labels(), vec![]);
        for ((i, j), e) in self.edges() {
            let is_undirected = self.edge((i, j)) == self.edge((j, i));
            if is_undirected && i > j {
                continue;
            }
            let new_node = extended.len();
            extended.nodes.push((edge_label_shift + e, NeighborMap::new()));

            assert!(extended.insert_plain_edge((i, new_node)));
            assert!(extended.insert_plain_edge((new_node, j)));
            if is_undirected && i != j {
                assert!(extended.insert_plain_edge((j, new_node)));
                assert!(extended.insert_plain_edge((new_node, i)));
            }
        }
        assert_eq!(self.is_undirected(), extended.is_undirected());
        extended
    }

    fn truncate_permutation(&self, p: &Permutation) -> Permutation {
        Permutation(p.0.clone().into_iter().filter(|(k, i)| {
            let keep = k < &self.len();
            assert_eq!(keep, i < &self.len());
            keep
        }).collect())
    }

    fn truncated_orbits(&self, orbits: Orbits) -> Vec<Orbit> {
        orbits.into_iter().filter(|o| {
            let keep = o.iter().all(|&i| i < self.len());
            assert!(keep || !o.iter().any(|&i| i < self.len()));
            keep
        }).collect()
    }
}

impl GroupPermutableFromFused for EdgeLabelledGraph {
    fn auts(&self) -> Automorphisms {
        let Automorphisms { group, canon, orbits, bits, .. } = self.with_edges_as_nodes().automorphisms();
        let group = PermutationGroup::new(self.len(), group.
            generators.into_iter().map(|g| self.truncate_permutation(&g)).collect_vec());
        let canon = self.truncate_permutation(&canon);
        let decanon = canon.inverse();
        let orbits = self.truncated_orbits(orbits);
        Automorphisms { group, canon, decanon, orbits, bits }
    }
}

pub type PartialEdgeLabelledGraph = PartialGraph<usize, usize>;

impl GroupPermutableFromFused for PartialEdgeLabelledGraph {
    fn auts(&self) -> Automorphisms {
        let mut a = self.with_unknown_nodes_uniquely_labelled().automorphisms();
        a.group.adjust_len(self.len());
        a
    }
}

impl PartialEdgeLabelledGraph {
    fn with_unknown_nodes_uniquely_labelled(&self) -> EdgeLabelledGraph {
        let node_label_shift = self.graph.node_labels().filter_map(|n| n).
            into_iter().max().unwrap_or(0) + 1;
        EdgeLabelledGraph::from_neighbors(self.graph.nodes.iter().enumerate().map(
            |(i, (n, ne))| (
                if self.unknown_nodes().contains(&i) {
                    node_label_shift + i - self.unknown_nodes().start
                } else { n.clone().unwrap() }, ne.clone())))
    }
}

#[cfg(test)]
pub mod tests {
    use crate::ans::{assert_bits_eq, Codec, Message, Symbol, Uniform, UniformCodec};
    use crate::graph_ans::{EmptyCodec, GraphIID};
    use crate::graph_ans::tests::{graph_codecs, plain_erdos_renyi, test_graph_shuffle_codecs, with_sampled_symbols};
    use crate::multiset::OrdSymbol;
    use crate::permutable::GroupPermutable;
    use crate::shuffle_ans::{PermutationUniform, TestConfig, Unordered};

    use super::*;

    pub fn node_labelled<EdgeC: Codec>(
        graph_codecs: impl IntoIterator<Item=GraphIID<EmptyCodec, EdgeC>>, num_labels: usize,
    ) -> impl Iterator<Item=GraphIID<Uniform, EdgeC>> where EdgeC::Symbol: OrdSymbol {
        graph_codecs.into_iter().map(move |c| GraphIID::new(
            c.nodes.len, c.edges.indices, Uniform::new(num_labels), c.edges.label, c.undirected))
    }

    pub fn node_labelled_codec_for(x: &Graph<usize, impl Symbol>, edge_prob: f64, undirected: bool, loops: bool) -> GraphIID<Uniform> {
        let num_labels = x.node_labels().max().unwrap() + 1;
        let c = plain_erdos_renyi(x.len(), edge_prob, undirected, loops);
        GraphIID::new(c.nodes.len, c.edges.indices, Uniform::new(num_labels), c.edges.label, c.undirected)
    }

    pub fn with_edge_labelled_uniform(graphs: impl IntoIterator<Item=EdgeLabelledGraph>, edge_prob: f64, undirected: bool, loops: bool) -> impl Iterator<Item=(GraphIID<Uniform, Uniform>, EdgeLabelledGraph)> {
        graphs.into_iter().map(move |x| {
            let c = node_labelled_codec_for(&x, edge_prob, undirected, loops);
            let num_labels = x.edge_labels().max().unwrap_or(0) + 1;
            (GraphIID::new(c.nodes.len, c.edges.indices, c.nodes.item, Uniform::new(num_labels), c.undirected), x)
        })
    }

    pub fn with_node_labelled_uniform(graphs: impl IntoIterator<Item=NodeLabelledGraph>, edge_prob: f64, undirected: bool, loops: bool) -> impl Iterator<Item=(GraphIID<Uniform>, NodeLabelledGraph)> {
        graphs.into_iter().map(move |x| {
            let num_labels = x.node_labels().max().unwrap() + 1;
            let c = plain_erdos_renyi(x.len(), edge_prob, undirected, loops);
            (GraphIID::new(c.nodes.len, c.edges.indices, Uniform::new(num_labels), c.edges.label, c.undirected), x)
        })
    }

    #[test]
    fn edge_labelled_permutable_axioms_fixed_slow_cases() {
        let codec = edge_labelled([plain_erdos_renyi(10, 0.4, false, false)]).into_iter().next().unwrap();
        for seed in 0..5 {
            codec.sample(seed).with_edges_as_nodes().automorphisms();
        }
    }

    #[test]
    fn edge_labelled_permutable_axioms_fixed_small_slow_case() {
        let codec = edge_labelled([plain_erdos_renyi(9, 0.5, false, false)]).into_iter().next().unwrap();
        let extended = codec.sample(49).with_edges_as_nodes();
        assert!(timeit_loops!(1, { extended.automorphisms(); }) < 0.001);
    }

    #[test]
    fn water_automorphisms() {
        let h2o = EdgeLabelledGraph::undirected(
            [0, 1, 0],
            [((1, 0), 0), ((1, 2), 0)]);
        assert_bits_eq(1., h2o.automorphisms().bits);
    }

    #[test]
    fn hydrogen_peroxide_automorphisms() {
        let h2o2 = EdgeLabelledGraph::undirected(
            [0, 1, 1, 0],
            [((1, 0), 0), ((2, 3), 0), ((1, 2), 0)]);
        assert_bits_eq(1., h2o2.automorphisms().bits);
    }

    #[test]
    fn boric_acid_automorphisms() {
        let bh3o3 = EdgeLabelledGraph::undirected(
            [0, 1, 2, 1, 2, 1, 2],
            [
                ((0, 1), 0), ((0, 3), 0), ((0, 5), 0),
                ((1, 2), 0), ((3, 4), 0), ((5, 6), 0)]);
        assert_bits_eq(PermutationUniform { len: 3 }.uni_bits(), bh3o3.automorphisms().bits);
    }

    #[test]
    fn ethylene_automorphisms() {
        let c2h4 = EdgeLabelledGraph::undirected(
            [0, 0, 1, 1, 0, 0],
            [((2, 0), 0), ((2, 1), 0), ((2, 3), 1), ((3, 4), 0), ((3, 5), 0)]);
        assert_bits_eq(3., c2h4.automorphisms().bits);
    }

    #[test]
    fn chiral_automorphisms() {
        let chiral = EdgeLabelledGraph::new(
            [0, 0, 0, 1, 2],
            [((0, 1), 0), ((0, 2), 1), ((0, 3), 0), ((0, 4), 0)]);
        assert_bits_eq(0., chiral.automorphisms().bits);
    }

    pub fn node_labelled_graph_codecs(undirected: bool, loops: bool) -> impl Iterator<Item=GraphIID<Uniform>> {
        node_labelled(graph_codecs(undirected, loops), 7)
    }

    pub fn edge_labelled(graph_codecs: impl IntoIterator<Item=GraphIID>) -> impl Iterator<Item=GraphIID<Uniform, Uniform>> {
        node_labelled(graph_codecs, 7).into_iter().map(|c| {
            GraphIID::new(c.nodes.len, c.edges.indices, c.nodes.item, Uniform::new(3), c.undirected)
        })
    }

    #[test]
    fn sampled_node_labelled_shuffle() {
        test_graph_shuffle_codecs(with_sampled_symbols(node_labelled_graph_codecs(false, false)), &TestConfig::test(0));
    }

    #[test]
    fn sampled_node_labelled_undirected_shuffle() {
        test_graph_shuffle_codecs(with_sampled_symbols(node_labelled_graph_codecs(true, false)), &TestConfig::test(0));
    }

    #[test]
    fn sampled_edge_labelled_shuffle() {
        for loops in [false, true] {
            test_graph_shuffle_codecs(with_sampled_symbols(edge_labelled(graph_codecs(false, loops))), &TestConfig::test(0));
        }
    }

    #[test]
    fn test_dense_edge_indices_codec() {
        for (c, x) in with_sampled_symbols(edge_labelled(graph_codecs(false, false))) {
            c.edges.indices.test(&Unordered(x.edge_indices().collect_vec()), &Message::random(0));
        }
    }

    #[test]
    fn test_edges_iid() {
        for (c, x) in with_sampled_symbols(edge_labelled(graph_codecs(false, false))) {
            c.edges.test(&Unordered(x.edges().collect_vec()), &Message::random(0));
        }
    }

    #[test]
    fn sampled_edge_labelled_undirected_shuffle() {
        let codec_and_graphs = with_sampled_symbols(edge_labelled(graph_codecs(true, false))).into_iter().collect_vec();
        for (_, g) in &codec_and_graphs {
            assert!(!g.has_selfloops());
            assert!(g.is_undirected());
        }
        test_graph_shuffle_codecs(codec_and_graphs, &TestConfig::test(0));
    }

    #[test]
    fn no_traces_issue() {
        let g = NodeLabelledGraph::undirected(
            [0, 1, 1, 0, 1, 1],
            [(0, 1), (1, 2), (3, 4), (4, 5)].map(|e| (e, ())));
        let canon_graph = g.canonized().0;
        for seed in 0..50 {
            assert_eq!(canon_graph, g.shuffled(seed).canonized().0);
        }
    }
}
