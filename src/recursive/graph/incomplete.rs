//! Incomplete shuffle coding on graphs based on the Weisfeiler-Lehman hash.
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

use itertools::Itertools;

use crate::codec::Codec;
use crate::codec::OrdSymbol;
use crate::graph::{Directed, EdgeType, Graph};
use crate::graph_codec::{EdgeIndicesCodec, GraphIID, PolyaUrnEdgeIndexCodec, PolyaUrnEdgeIndicesCodec};
use crate::permutable::{Len, Orbit, Permutable, Unordered};
use crate::recursive::{OrbitCodec, Prefix, PrefixFn};
use crate::recursive::graph::GraphPrefix;
use crate::recursive::graph::slice::PolyaUrnSliceEdgeIndicesCodec;
use crate::recursive::joint::JointPrefix;
use crate::recursive::prefix_orbit::{hash, orbits_by_id, PrefixOrbitCodec};

#[allow(unused)]
fn orbits(ids: &Vec<usize>) -> BTreeSet<Orbit> {
    orbits_by_id(ids.iter()).into_values().collect()
}

impl<N: OrdSymbol, E: OrdSymbol, Ty: EdgeType> Graph<N, E, Ty> {
    #[allow(unused)]
    pub fn wl_node_hashes(&self, num_iter: usize, extra_half_iter: bool, stop_on_convergence: bool) -> Vec<usize> {
        assert!(!Ty::is_directed());
        let mut hashes = self.node_labels().map(hash).collect_vec();
        let mut p = if stop_on_convergence { Some(orbits(&hashes)) } else { None };

        if extra_half_iter {
            hashes = self.half_wl_iter(&hashes);
        }

        for _ in 0..num_iter {
            hashes = self.wl_iter(&hashes);
            if stop_on_convergence {
                let new_p = Some(orbits(&hashes));
                if new_p == p { break; }
                p = new_p;
            }
        }
        hashes
    }

    pub fn half_wl_iter(&self, hashes: &Vec<usize>) -> Vec<usize> {
        hashes.iter().enumerate().map(|(index, hash)| self.half_wl_node_iter(index, *hash)).collect_vec()
    }

    pub fn half_wl_node_iter(&self, i: usize, h: usize) -> usize {
        hash((h, self.nodes[i].1.iter().map(|(_, e)| e).sorted_unstable().collect_vec()))
    }

    pub fn wl_iter(&self, hashes: &Vec<usize>) -> Vec<usize> {
        hashes.iter().enumerate().map(|(i, h)|
        hash((h, self.nodes[i].1.iter().map(|(ne, e)| (hashes[*ne], e)).sorted_unstable().collect_vec()))
        ).collect_vec()
    }

    #[allow(unused)]
    /// Weisfeiler-Lehman hash of the graph.
    pub fn wl_hash(&self, max_iter: usize, extra_half_iter: bool, stop_on_convergence: bool) -> usize {
        let num_iter = max_iter.min(self.len());
        let mut hashes = self.wl_node_hashes(num_iter, extra_half_iter, stop_on_convergence);
        hashes.sort_unstable();
        hash(hashes)
    }

    #[allow(unused)]
    /// Weishfeiler-Lehman hash of the graph, run to convergence of partition (up to N iterations).
    pub fn full_wl_hash(&self) -> usize {
        self.wl_hash(self.len(), false, true)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WLOrbitCodecs<P: Prefix<Full=Graph<N, E, Ty>>, N: OrdSymbol = (), E: OrdSymbol = (), Ty: EdgeType = Directed> {
    pub num_iter: usize,
    pub extra_half_iter: bool,
    pub phantom: PhantomData<P>,
}

impl<P: Prefix<Full=Graph<N, E, Ty>>, N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType> WLOrbitCodecs<P, N, E, Ty> {
    pub fn new(num_iter: usize, extra_half_iter: bool) -> Self {
        Self { num_iter, extra_half_iter, phantom: PhantomData }
    }

    fn update_around_slice(&self, x: &GraphPrefix<N, E, Ty>, orbits: &mut PrefixOrbitCodec<Vec<usize>>, (_, Unordered(edges)): &<GraphPrefix<N, E, Ty> as Prefix>::Slice, index: usize) {
        assert_eq!(x.len(), orbits.len());
        let mut neighbors = edges.iter().map(|((i, j), _)| {
            assert_eq!(*i, index);
            *j
        }).collect_vec();
        neighbors.push(index);
        self.update_around(&x.graph, orbits, neighbors);
    }

    fn update_around<M: OrdSymbol>(&self, x: &Graph<M, E, Ty>, orbits: &mut PrefixOrbitCodec<Vec<usize>>, indices: Vec<usize>) {
        assert!(!Ty::is_directed());
        // Add elements at distance 0.
        let mut new_ids_by_distance = vec![BTreeMap::from_iter(indices.into_iter().map(|i| (i, orbits.id(i))))];

        // Add neighbors using breadth-first search at distances >0.
        for _ in 0..self.num_iter {
            let mut neighbors_ = BTreeMap::new();
            for p in new_ids_by_distance.last().unwrap().keys() {
                neighbors_.extend(x.nodes[*p].1.keys().cloned().
                    filter(|i| new_ids_by_distance.iter().all(|ids| !ids.contains_key(i))).
                    map(|i| (i, orbits.id(i))))
            }
            new_ids_by_distance.push(neighbors_);
        }

        // Calculate updated hashes at distance 0 from the node labels and optionally the multiset of neighboring edge labels.
        for (&i, new_id) in new_ids_by_distance[0].iter_mut() {
            let n = x.nodes[i].0.clone();
            let index = if i < orbits.len() { None } else { Some(i) };
            let mut h = hash((n, index));

            if self.extra_half_iter {
                h = x.half_wl_node_iter(i, h)
            }

            new_id[0] = h;
        }

        // Calculate updated hashes at distances >0.
        for iter in 0..self.num_iter {
            let indices = new_ids_by_distance.iter().take(iter + 2).enumerate().flat_map(
                |(update_i, x)| x.keys().map(move |i| (update_i, *i))).collect_vec();
            for (update_i, i) in indices {
                let prev_hash = |i: usize| {
                    for new_id in new_ids_by_distance.iter() {
                        if let Some(h) = new_id.get(&i) {
                            return h.clone()[iter];
                        }
                    }
                    return orbits.id(i)[iter];
                };

                let self_hash = prev_hash(i).clone();
                let neighbor_hashes = x.nodes[i].1.iter().map(|(ne, e)| (prev_hash(*ne), e)).sorted_unstable().collect_vec();
                let new_hash = hash((self_hash, neighbor_hashes));
                new_ids_by_distance[update_i].get_mut(&i).unwrap()[iter + 1] = new_hash;
            }
        }

        // Apply the updates.
        for (i, new_id) in new_ids_by_distance.into_iter().flat_map(|x| x) {
            orbits.update_id(i, new_id);
        }
    }
}

impl<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType> PrefixFn for WLOrbitCodecs<JointPrefix<Graph<N, E, Ty>>, N, E, Ty> {
    type Prefix = JointPrefix<Graph<N, E, Ty>>;
    type Output = PrefixOrbitCodec<Vec<usize>>;

    fn apply(&self, x: &JointPrefix<Graph<N, E, Ty>>) -> Self::Output {
        let mut prefix = GraphPrefix::from_full(x.full.clone());
        prefix.num_unknown_nodes = x.full.len() - x.len;
        WLOrbitCodecs::<GraphPrefix<N, E, Ty>, N, E, Ty>::new(self.num_iter, self.extra_half_iter).apply(&prefix)
    }

    fn update_after_pop_slice(&self, orbits: &mut Self::Output, x: &Self::Prefix, _slice: &()) {
        if Ty::is_directed() {
            *orbits = self.apply(x);
            return;
        }

        orbits.pop_id();
        self.update_around(&x.full, orbits, vec![x.len()]);
    }

    fn update_after_push_slice(&self, orbits: &mut Self::Output, x: &Self::Prefix, _slice: &()) {
        if Ty::is_directed() {
            *orbits = self.apply(x);
            return;
        }

        orbits.push_id();
        self.update_around(&x.full, orbits, vec![x.last_index()]);
    }

    fn swap(&self, orbits: &mut Self::Output, i: usize, j: usize) {
        orbits.swap(i, j)
    }
}

impl<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType> PrefixFn for WLOrbitCodecs<GraphPrefix<N, E, Ty>, N, E, Ty> {
    type Prefix = GraphPrefix<N, E, Ty>;
    type Output = PrefixOrbitCodec<Vec<usize>>;

    fn apply(&self, x: &Self::Prefix) -> Self::Output {
        let len = x.len();
        let mut hashes = x.graph.node_labels().enumerate().map(|(i, n)|
        hash((n, if i < len { None } else { Some(i) }))).collect();
        if self.extra_half_iter {
            hashes = x.graph.half_wl_iter(&hashes);
        }
        let mut ids = hashes.iter().map(|h| vec![h.clone()]).collect_vec();
        for _ in 0..self.num_iter {
            hashes = x.graph.wl_iter(&hashes);

            for (id, hash) in ids.iter_mut().zip_eq(&hashes) {
                id.push(hash.clone());
            }
        }

        PrefixOrbitCodec::new(ids, x.num_unknown_nodes)
    }

    fn update_after_pop_slice(&self, orbits: &mut Self::Output, x: &Self::Prefix, slice: &<Self::Prefix as Prefix>::Slice) {
        if Ty::is_directed() {
            *orbits = self.apply(x);
            return;
        }

        orbits.pop_id();
        assert_eq!(orbits.len(), x.len());
        self.update_around_slice(x, orbits, &slice, x.len());
    }

    fn update_after_push_slice(&self, orbits: &mut Self::Output, x: &Self::Prefix, slice: &<Self::Prefix as Prefix>::Slice) {
        if Ty::is_directed() {
            *orbits = self.apply(x);
            return;
        }

        orbits.push_id();
        self.update_around_slice(x, orbits, &slice, x.last_index());
    }

    fn swap(&self, orbits: &mut Self::Output, i: usize, j: usize) {
        orbits.swap(i, j)
    }
}

impl<Ty: EdgeType> Len for PolyaUrnEdgeIndexCodec<Ty> {
    fn len(&self) -> usize { 2 }
}

impl<Ty: EdgeType> Len for PolyaUrnEdgeIndicesCodec<Ty> {
    fn len(&self) -> usize { self.num_edges }
}

impl<'a, Ty: EdgeType> Len for PolyaUrnSliceEdgeIndicesCodec<'a, Ty> {
    fn len(&self) -> usize { self.len }
}

impl<NodeC: Codec, EdgeC: Codec<Symbol: OrdSymbol>, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> Len for GraphIID<NodeC, EdgeC, IndicesC> {
    fn len(&self) -> usize {
        self.nodes.len
    }
}

#[cfg(test)]
pub mod tests {
    use itertools::Itertools;

    use crate::benchmark::{DatasetStats, TestConfig};
    use crate::codec::Codec;
    use crate::datasets::dataset;
    use crate::graph::UnGraph;
    use crate::graph_codec::tests::{plain_erdos_renyi, test_graph_shuffle_codecs};
    use crate::permutable::Permutable;
    use crate::plain::labelled_graph::tests::with_edge_labelled_uniform;
    use crate::plain::PlainPermutable;
    use crate::recursive::prefix_orbit::PrefixOrbitCodec;

    #[test]
    fn test_full() {
        for g in dataset("as").unlabelled_graphs() {
            let shuffled = g.shuffled(0);
            let sec = timeit_loops!(1, {assert_eq!(g.full_wl_hash(), shuffled.full_wl_hash())});
            assert!(sec < 3.);
        }
    }

    #[test]
    fn test_5() {
        for g in dataset("as").unlabelled_graphs() {
            for extra_half_iter in [false, true] {
                let shuffled = g.shuffled(0);
                let sec = timeit_loops!(1, {assert_eq!(
                        g.wl_hash(5, extra_half_iter, false),
                        shuffled.wl_hash(5, extra_half_iter, false))});
                dbg!(sec);
                assert!(sec < 2.);
            }
        }
    }

    #[test]
    fn false_positive() {
        let cycle7 = UnGraph::plain(7, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 4)]);
        let cycle4_cycle3 = UnGraph::plain(7, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)]);
        assert!(!cycle7.is_isomorphic(&cycle4_cycle3));
        assert_eq!(cycle7.full_wl_hash(), cycle4_cycle3.full_wl_hash());
    }

    #[test]
    fn test_orbits() {
        let orbits = PrefixOrbitCodec::new(vec![0, 4, 6, 1, 1, 3, 3, 4, 5, 3, 2134, 123, 1, 23, 123, 124, 123, 53, 456, 45, 674, 56, 34, 5, 324, 64,
                                                567, 5, 8, 56, 63, 4, 3, 4, 5, 45, 4, 54, 5, 45, 4, 54, 54, 5, 45, 34, 23, 42, 3, 213, 42, 34, 23, 423, 42, 1, 3, 4], 5);
        orbits.test_on_samples(5000);
    }

    #[test]
    fn minor() {
        assert!(vec![0, 2] < vec![1]);
        assert!(vec![0, 2] < vec![1, 1]);
    }

    #[test]
    #[ignore = "long-running"]
    fn incomplete_deezer_ego_nets() {
        let graphs = dataset("deezer_ego_nets").unlabelled_graphs();
        let mut config = TestConfig::test(0);
        config.plain = false;
        config.plain_coset_recursive = false;
        config.joint = false;
        config.recursive = true;
        let edge_prob = DatasetStats::unlabelled(&graphs).edge.prob();
        let codecs_and_graphs = graphs.into_iter().map(
            |x| (plain_erdos_renyi(x.len(), edge_prob, false), x)).collect_vec();
        test_graph_shuffle_codecs(codecs_and_graphs.clone(), &TestConfig { no_incomplete: true, complete: true, ..config.clone() });
        for wl_iter in 0..10 {
            config.wl_iter = wl_iter;
            test_graph_shuffle_codecs(codecs_and_graphs.clone(), &config);
        }
    }

    #[test]
    #[ignore = "long-running"]
    fn incomplete_mutag() {
        let name = "MUTAG"; // "DBLP_v1";
        let graphs = dataset(name).edge_labelled_graphs().unwrap();
        let mut config = TestConfig::test(0);
        config.plain = false;
        config.plain_coset_recursive = false;
        config.joint = false;
        config.recursive = true;
        let edge_prob = DatasetStats::edge_labelled(&graphs).edge.prob();
        let codecs_and_graphs = with_edge_labelled_uniform(graphs, edge_prob, false).collect_vec();
        test_graph_shuffle_codecs(codecs_and_graphs.clone(), &TestConfig { no_incomplete: true, complete: true, ..config.clone() });
        for wl_iter in 0..10 {
            config.wl_iter = wl_iter;
            test_graph_shuffle_codecs(codecs_and_graphs.clone(), &config);
        }
    }

    #[test]
    #[ignore = "long-running"]
    fn incomplete_github_stargazers() {
        let graphs = dataset("github_stargazers").unlabelled_graphs();
        let mut config = TestConfig::test(0);
        config.plain = false;
        config.plain_coset_recursive = false;
        let edge_prob = DatasetStats::unlabelled(&graphs).edge.prob();
        let codecs_and_graphs = graphs.into_iter().map(
            |x| (plain_erdos_renyi(x.len(), edge_prob, false), x)).collect_vec();
        for wl_iter in 0..1 {
            config.wl_iter = wl_iter;
            test_graph_shuffle_codecs(codecs_and_graphs.clone(), &config);
        }
    }

    #[test]
    #[ignore = "long-running"]
    fn incomplete_github_stargazers_2() {
        let graphs = dataset("github_stargazers").unlabelled_graphs();
        let mut config = TestConfig::test(0);
        config.plain = false;
        config.plain_coset_recursive = false;
        let edge_prob = DatasetStats::unlabelled(&graphs).edge.prob();
        let codecs_and_graphs = graphs.into_iter().map(
            |x| (plain_erdos_renyi(x.len(), edge_prob, false), x)).collect_vec();
        for wl_iter in 1..2 {
            config.wl_iter = wl_iter;
            test_graph_shuffle_codecs(codecs_and_graphs.clone(), &config);
        }
    }
}