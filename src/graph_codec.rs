use std::cell::{RefCell, RefMut};
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::vec::IntoIter;

use itertools::{Itertools, repeat_n};

use crate::codec::{Bernoulli, Codec, ConstantCodec, IID, Message, MutCategorical, MutDistribution, OrdSymbol};
use crate::graph::{Directed, Edge, EdgeIndex, EdgeType, Graph, PlainGraph};
use crate::multiset::Multiset;
use crate::permutable::Unordered;
use crate::plain::{plain_shuffle_codec, PlainShuffleCodec};
use crate::recursive::multiset::{multiset_shuffle_codec, MultisetShuffleCodec};

pub type EmptyCodec = ConstantCodec<()>;

/// Coding edge indices independently of the i.i.d. node and edge labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphIID<
    NodeC: Codec = EmptyCodec,
    EdgeC: Codec<Symbol: OrdSymbol> = EmptyCodec,
    IndicesC: EdgeIndicesCodec = ErdosRenyi<Directed>> {
    pub nodes: IID<NodeC>,
    pub edges: EdgesIID<EdgeC, IndicesC>,
}

impl<NodeC: Codec, EdgeC: Codec<Symbol: OrdSymbol>, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> Codec for GraphIID<NodeC, EdgeC, IndicesC> {
    type Symbol = Graph<NodeC::Symbol, EdgeC::Symbol, Ty>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.edges.push(m, &Unordered(x.edges()));
        self.nodes.push(m, &x.node_labels().collect());
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Self::Symbol::new(self.nodes.pop(m), self.edges.pop(m).into_ordered())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        Some(self.nodes.bits(&x.node_labels().collect())? + self.edges.bits(&Unordered(x.edges()))?)
    }
}

impl<NodeC: Codec, EdgeC: Codec<Symbol: OrdSymbol>, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> GraphIID<NodeC, EdgeC, IndicesC> {
    pub fn new(num_nodes: usize, edge_indices: IndicesC, node: NodeC, edge: EdgeC) -> Self {
        Self { nodes: IID::new(node, num_nodes), edges: EdgesIID::new(edge_indices, edge) }
    }
}

/// Coding edge indices independently of the i.i.d. labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgesIID<EdgeC: Codec = EmptyCodec, IndicesC: EdgeIndicesCodec = ErdosRenyi<Directed>> {
    pub indices: IndicesC,
    pub label: EdgeC,
}

impl<EdgeC: Codec<Symbol: OrdSymbol>, IndicesC: EdgeIndicesCodec> Codec for EdgesIID<EdgeC, IndicesC> {
    type Symbol = Multiset<Edge<EdgeC::Symbol>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let (indices, labels) = Self::split(x);
        self.labels(indices.len()).push(m, &labels);
        self.indices.push(m, &indices);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut indices = self.indices.pop(m).into_ordered();
        indices.sort_unstable();
        let labels = self.labels(indices.len()).pop(m);
        Unordered(indices.into_iter().zip_eq(labels.into_iter()).collect_vec())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let (indices, edges) = Self::split(x);
        let labels_codec = self.labels(indices.len());
        Some(self.indices.bits(&indices)? + labels_codec.bits(&edges)?)
    }
}

impl<EdgeC: Codec<Symbol: OrdSymbol>, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> EdgesIID<EdgeC, IndicesC> {
    fn split(x: &Multiset<Edge<<EdgeC as Codec>::Symbol>>) -> (Multiset<EdgeIndex>, Vec<<EdgeC as Codec>::Symbol>) {
        let (indices, labels) = x.to_ordered().iter().cloned().sorted_unstable_by_key(|(i, _)| *i).unzip();
        (Unordered(indices), labels)
    }

    fn labels(&self, len: usize) -> IID<EdgeC> {
        IID::new(self.label.clone(), len)
    }

    pub fn new(indices: IndicesC, label: EdgeC) -> Self {
        Self { indices, label }
    }
}

pub trait Alphabet<S>: IntoIterator<Item=S> + Clone {
    fn len(&self) -> usize;
}

impl<S: OrdSymbol> Alphabet<S> for Vec<S> {
    fn len(&self) -> usize { self.len() }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseSetIID<S: OrdSymbol, A: Alphabet<S>> {
    pub alphabet: A,
    pub contains: IID<Bernoulli>,
    pub phantom: PhantomData<S>,
}

impl<S: OrdSymbol, A: Alphabet<S>> Codec for DenseSetIID<S, A> {
    type Symbol = Multiset<S>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.contains.push(m, &self.dense(x));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let dense = self.contains.pop(m);
        Unordered(self.alphabet.clone().into_iter().zip_eq(dense).filter_map(|(i, b)| if b { Some(i.clone()) } else { None }).collect_vec())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.contains.bits(&self.dense(x))
    }
}

impl<S: OrdSymbol, A: Alphabet<S>> DenseSetIID<S, A> {
    pub fn new(contains: Bernoulli, alphabet: A) -> Self {
        Self { contains: IID::new(contains, alphabet.len()), alphabet, phantom: PhantomData }
    }

    fn dense(&self, x: &Multiset<S>) -> Vec<bool> {
        let mut x = x.to_ordered().iter().cloned().collect::<HashSet<_>>();
        let as_vec = self.alphabet.clone().into_iter().map(|i| x.remove(&i)).collect_vec();
        assert!(x.is_empty());
        as_vec
    }
}

pub trait EdgeIndicesCodec: Codec<Symbol=Multiset<EdgeIndex>> {
    type Ty: EdgeType;

    fn loops(&self) -> bool;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErdosRenyi<Ty: EdgeType, I: Alphabet<EdgeIndex> = AllEdgeIndices<Ty>> {
    pub dense: DenseSetIID<EdgeIndex, I>,
    pub loops: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType, I: Alphabet<EdgeIndex>> Codec for ErdosRenyi<Ty, I> {
    type Symbol = Multiset<EdgeIndex>;
    fn push(&self, m: &mut Message, x: &Self::Symbol) { self.dense.push(m, x) }
    fn pop(&self, m: &mut Message) -> Self::Symbol { self.dense.pop(m) }
    fn bits(&self, x: &Self::Symbol) -> Option<f64> { self.dense.bits(x) }
}

impl<Ty: EdgeType, I: Alphabet<EdgeIndex>> EdgeIndicesCodec for ErdosRenyi<Ty, I> {
    type Ty = Ty;

    fn loops(&self) -> bool { self.loops }
}

impl<Ty: EdgeType, I: Alphabet<EdgeIndex>> ErdosRenyi<Ty, I> {
    pub fn new(edge: Bernoulli, indices: I, loops: bool) -> ErdosRenyi<Ty, I> {
        ErdosRenyi { dense: DenseSetIID::new(edge, indices), loops, phantom: PhantomData }
    }
}

pub fn erdos_renyi_indices<Ty: EdgeType>(num_nodes: usize, edge: Bernoulli, loops: bool) -> ErdosRenyi<Ty> {
    ErdosRenyi::new(edge, AllEdgeIndices::<Ty> { num_nodes, loops, phantom: PhantomData }, loops)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllEdgeIndices<Ty: EdgeType> {
    pub num_nodes: usize,
    pub loops: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> IntoIterator for AllEdgeIndices<Ty> {
    type Item = EdgeIndex;
    type IntoIter = IntoIter<EdgeIndex>;

    fn into_iter(self) -> Self::IntoIter {
        let num_nodes = self.num_nodes;
        let loops = self.loops;
        let mut out = if loops { (0..num_nodes).map(|i| (i, i)).collect_vec() } else { vec![] };
        let fwd = (0..num_nodes).flat_map(|j| (0..j).map(move |i| (i, j)));
        if Ty::is_directed() {
            out.extend(fwd.flat_map(|(i, j)| [(i, j), (j, i)]));
        } else {
            out.extend(fwd);
        }
        out.into_iter()
    }
}

impl<Ty: EdgeType> Alphabet<EdgeIndex> for AllEdgeIndices<Ty> {
    fn len(&self) -> usize { num_all_edge_indices::<Ty>(self.num_nodes, self.loops) }
}

pub fn num_all_edge_indices<Ty: EdgeType>(num_nodes: usize, loops: bool) -> usize {
    (if loops { num_nodes } else { 0 }) + (num_nodes * num_nodes - num_nodes) / if Ty::is_directed() { 1 } else { 2 }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyaUrnEdgeIndexCodec<Ty: EdgeType> {
    graph: PlainGraph<Ty>,
    codec: RefCell<MutCategorical>,
    loops: bool,
    redraws: bool,
}

impl<Ty: EdgeType> PolyaUrnEdgeIndexCodec<Ty> {
    fn new(graph: PlainGraph<Ty>, loops: bool, redraws: bool) -> Self {
        assert!(!Ty::is_directed());
        if !loops {
            assert!(!graph.has_selfloops());
        }
        let masses = graph.nodes.iter().map(|(_, es)| es.len() + 1).enumerate();
        let codec = RefCell::new(MutCategorical::new(masses));
        Self { graph, codec, loops, redraws }
    }

    fn use_codec_without<R>(&self, node: usize, mut body: impl FnMut(RefMut<MutCategorical>) -> R) -> R {
        let mut codec = self.codec.borrow_mut();
        let mut excluded = vec![];
        if !self.loops {
            excluded.push((node, codec.remove_all(&node)));
        }
        if !self.redraws {
            for (&neighbor, _) in &self.graph.nodes[node].1 {
                excluded.push((neighbor, codec.remove_all(&neighbor)));
            }
        }
        let result = body(codec);
        for (node, mass) in excluded {
            self.codec.borrow_mut().insert(node, mass)
        }
        result
    }

    fn remove_edge(&mut self, (i, j): (usize, usize)) {
        assert!(self.graph.remove_plain_edge((i, j)));
        let codec = self.codec.get_mut();
        codec.remove(&i, 1);
        if i != j {
            codec.remove(&j, 1);
        }
    }

    fn insert_edge(&mut self, (i, j): (usize, usize)) {
        assert!(self.graph.insert_plain_edge((i, j)));
        let codec = self.codec.get_mut();
        codec.insert(i, 1);
        if i != j {
            codec.insert(j, 1);
        }
    }
}

impl<Ty: EdgeType> Codec for PolyaUrnEdgeIndexCodec<Ty> {
    type Symbol = VecEdgeIndex;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        if let [node, node_] = x[..] {
            self.use_codec_without(node, |codec| codec.push(m, &node_));
            self.codec.push(m, &node);
        } else {
            panic!("Expected two nodes, got {:?}", x)
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let node = self.codec.pop(m);
        let node_ = self.use_codec_without(node, |codec| codec.pop(m));
        vec![node, node_]
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        if let [node, node_] = x[..] {
            Some(self.codec.bits(&node)? + self.use_codec_without(node, |codec| codec.bits(&node_))?)
        } else {
            panic!("Expected two nodes, got {:?}", x)
        }
    }
}

/// Reuse Vec's implementation of Permutable for undirected edges of a Polya Urn.
type VecEdgeIndex = Vec<usize>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyaUrnEdgeIndicesCodec<Ty: EdgeType> {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub loops: bool,
    pub redraws: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> PolyaUrnEdgeIndicesCodec<Ty> {
    pub fn new(num_nodes: usize, num_edges: usize, loops: bool, redraws: bool) -> Self {
        Self { num_nodes, num_edges, loops, redraws, phantom: PhantomData }
    }

    fn edge_codec(&self, graph: PlainGraph<Ty>) -> PlainShuffleCodec<PolyaUrnEdgeIndexCodec<Ty>> {
        plain_shuffle_codec(PolyaUrnEdgeIndexCodec::new(graph, self.loops, self.redraws))
    }
}

impl<Ty: EdgeType> Codec for PolyaUrnEdgeIndicesCodec<Ty> {
    type Symbol = Vec<EdgeIndex>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(x.len(), self.num_edges);
        let graph = PlainGraph::<Ty>::new(repeat_n((), self.num_nodes), x.iter().map(|i| (i.clone(), ())));
        let mut edge_codec = self.edge_codec(graph);
        for (i, j) in x.iter().rev() {
            edge_codec.ordered.remove_edge((*i, *j));
            edge_codec.push(m, &Unordered(vec![*i, *j]));
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let graph = Graph::plain_empty(self.num_nodes);
        let mut edge_codec = self.edge_codec(graph);
        let mut edges = vec![];
        for _i in 0..self.num_edges {
            let edge = edge_codec.pop(m).into_ordered();
            if let [i, j] = edge[..] {
                edge_codec.ordered.insert_edge((i, j));
                edges.push((i, j));
            } else {
                panic!("Expected two nodes, got {:?}", edge)
            };
        }
        edges
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        if self.loops && self.redraws {
            let m = &mut Message::zeros();
            let bits = m.virtual_bits();
            self.push(m, x);
            Some(m.virtual_bits() - bits)
        } else { None }
    }
}

pub type PolyaUrn<Ty> = MultisetShuffleCodec<EdgeIndex, PolyaUrnEdgeIndicesCodec<Ty>>;

impl<Ty: EdgeType> EdgeIndicesCodec for PolyaUrn<Ty> {
    type Ty = Ty;

    fn loops(&self) -> bool {
        self.ordered().loops
    }
}

impl<Ty: EdgeType> PolyaUrn<Ty> {
    pub fn ordered(&self) -> &PolyaUrnEdgeIndicesCodec<Ty> {
        match self {
            Self::A(c) => &c.ordered,
            Self::B(c) => &c.prefix.slice_codecs.empty.full,
        }
    }
}

pub fn polya_urn<Ty: EdgeType>(num_nodes: usize, num_edges: usize, loops: bool, redraws: bool) -> PolyaUrn<Ty> {
    multiset_shuffle_codec(PolyaUrnEdgeIndicesCodec::new(num_nodes, num_edges, loops, redraws))
}

#[cfg(test)]
pub mod tests {
    use crate::benchmark::TestConfig;
    use crate::codec::Symbol;
    use crate::codec::tests::test_and_print_vec;
    use crate::graph::{DiGraph, Undirected, UnGraph, with_isomorphism_test_max_len};
    use crate::permutable::{Permutable, PermutableCodec};
    use crate::plain::PlainPermutable;
    use crate::plain::test::test_plain_shuffle_codecs;
    use crate::recursive::{Autoregressive, ShuffleCodec};
    use crate::recursive::graph::GraphPrefix;
    use crate::recursive::graph::incomplete::WLOrbitCodecs;
    use crate::recursive::graph::slice::{ErdosRenyiSliceCodecs, PolyaUrnSliceCodecs};
    use crate::recursive::joint::{JointPrefix, JointSliceCodecs};
    use crate::recursive::plain_orbit::PlainOrbitCodecs;

    use super::*;

    pub type UnGraphIID<NodeC = EmptyCodec, EdgeC = EmptyCodec, IndicesC = ErdosRenyi<Undirected>> = GraphIID<NodeC, EdgeC, IndicesC>;
    pub type PlainGraphIID<IndicesC = ErdosRenyi<Directed>> = GraphIID<EmptyCodec, EmptyCodec, IndicesC>;

    pub fn plain_erdos_renyi<Ty: EdgeType>(n: usize, p: f64, loops: bool) -> PlainGraphIID<ErdosRenyi<Ty>> {
        let norm = 1 << 28;
        let mass = (p * norm as f64) as usize;
        let edge_indices = erdos_renyi_indices(n, Bernoulli::new(mass, norm), loops);
        GraphIID::new(n, edge_indices, EmptyCodec::default(), EmptyCodec::default())
    }

    pub fn test_joint_shuffle_codecs<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType, C: PermutableCodec<Symbol=Graph<N, E, Ty>> + Symbol>(
        codecs: &Vec<C>,
        unordered: &Vec<Unordered<C::Symbol>>,
        config: &TestConfig)
    where
        Graph<N, E, Ty>: PlainPermutable,
        GraphPrefix<N, E, Ty>: PlainPermutable,
    {
        test_plain_shuffle_codecs(codecs, unordered, config);

        if config.joint {
            if config.complete {
                let complete_joint = |c| ShuffleCodec::new(JointSliceCodecs::new(c), PlainOrbitCodecs::new());
                test_and_print_vec(codecs.iter().cloned().map(complete_joint), unordered, &config.initial_message());
            }

            if !config.no_incomplete {
                let incomplete_joint = |c| ShuffleCodec::new(JointSliceCodecs::new(c), WLOrbitCodecs::<JointPrefix<_>, _, _, _>::new(config.wl_iter, config.wl_extra_half_iter));
                test_and_print_vec(codecs.iter().cloned().map(incomplete_joint), unordered, &config.initial_message());
            }
        }
    }

    pub fn test_graph_shuffle_codecs<NodeC: Codec<Symbol: OrdSymbol + Default> + Symbol, EdgeC: Codec<Symbol: OrdSymbol> + Symbol, Ty: EdgeType>(
        er_codecs_and_graphs: impl IntoIterator<Item=(GraphIID<NodeC, EdgeC, ErdosRenyi<Ty>>, Graph<NodeC::Symbol, EdgeC::Symbol, Ty>)>,
        config: &TestConfig)
    where
        Graph<NodeC::Symbol, EdgeC::Symbol, Ty>: PlainPermutable,
        GraphPrefix<NodeC::Symbol, EdgeC::Symbol, Ty>: PlainPermutable,
    {
        with_isomorphism_test_max_len(config.isomorphism_test_max_len, || {
            let er_and_graphs = er_codecs_and_graphs.into_iter().collect_vec();
            let (er, graphs): (Vec<_>, Vec<_>) = er_and_graphs.iter().cloned().unzip();
            let unordered = &graphs.iter().map(|x| Unordered(x.clone())).collect();

            if config.er {
                test_joint_shuffle_codecs(&er, unordered, config);
            }
            if config.pu {
                let pu = er_and_graphs.into_iter().map(|(c, x)| {
                    let edge_indices = polya_urn(x.len(), x.num_edges(), false, false);
                    GraphIID::new(x.len(), edge_indices, c.nodes.item.clone(), c.edges.label.clone())
                }).collect_vec();
                test_joint_shuffle_codecs(&pu, unordered, config);
            }

            if config.recursive {
                let wl_orbit_codecs = WLOrbitCodecs::<GraphPrefix<_, _, _>, _, _, _>::new(config.wl_iter, config.wl_extra_half_iter);
                if config.ae {
                    let aer = er.clone().into_iter().map(|c| Autoregressive(ErdosRenyiSliceCodecs {
                        len: c.nodes.len,
                        has_edge: c.edges.indices.dense.contains.item.clone(),
                        node: c.nodes.item.clone(),
                        edge: c.edges.label.clone(),
                        loops: c.edges.indices.loops(),
                        phantom: PhantomData,
                    }));
                    if config.complete {
                        let codecs = aer.clone().map(|c| ShuffleCodec::new(c.0, PlainOrbitCodecs::new()));
                        test_and_print_vec(codecs, unordered, &config.initial_message());
                    }
                    if !config.no_incomplete {
                        let codecs = aer.map(|c| ShuffleCodec::new(c.0, wl_orbit_codecs.clone()));
                        test_and_print_vec(codecs, unordered, &config.initial_message());
                    }
                }
                if config.ap && !Ty::is_directed() {
                    let apu = er.clone().into_iter().map(|c| Autoregressive(PolyaUrnSliceCodecs {
                        len: c.nodes.len,
                        node: c.nodes.item.clone(),
                        edge: c.edges.label.clone(),
                        loops: c.edges.indices.loops(),
                        phantom: PhantomData,
                    }));
                    if config.complete {
                        let codecs = apu.clone().map(|c| ShuffleCodec::new(c.0, PlainOrbitCodecs::new()));
                        test_and_print_vec(codecs, unordered, &config.initial_message());
                    }
                    if !config.no_incomplete {
                        let codecs = apu.map(|c| ShuffleCodec::new(c.0, wl_orbit_codecs.clone()));
                        test_and_print_vec(codecs, unordered, &config.initial_message());
                    }
                }
            }
        })
    }


    pub fn graph_codecs<Ty: EdgeType>(loops: bool) -> impl Iterator<Item=PlainGraphIID<ErdosRenyi<Ty>>> {
        (0..20).map(move |len| plain_erdos_renyi::<Ty>(len, 0.4, loops))
    }

    fn small_codecs_and_digraphs() -> impl Iterator<Item=(GraphIID, Graph)> {
        small_digraphs().into_iter().map(|x| (plain_erdos_renyi(x.len(), 0.1, false), x))
    }

    pub fn small_digraphs() -> impl Iterator<Item=Graph> {
        [
            DiGraph::plain(0, vec![]),
            DiGraph::plain(1, vec![]),
            DiGraph::plain(3, vec![(0, 1)]),
            DiGraph::plain(4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]),
            DiGraph::plain(11, vec![
                (0, 1), (0, 2), (0, 3),
                (1, 4), (2, 5), (3, 6),
                (4, 7), (5, 8), (6, 9),
                (7, 10), (8, 10), (9, 10)])
        ].into_iter()
    }

    fn small_codecs_and_ungraphs() -> impl Iterator<Item=(UnGraphIID, UnGraph)> {
        small_ungraphs().into_iter().map(|x| (plain_erdos_renyi(x.len(), 0.1, false), x))
    }

    fn small_ungraphs() -> impl IntoIterator<Item=UnGraph> {
        [
            UnGraph::plain(0, vec![]),
            UnGraph::plain(1, vec![]),
            UnGraph::plain(3, vec![(0, 1)]),
            UnGraph::plain(4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]),
            UnGraph::plain(11, vec![
                (0, 1), (0, 2), (0, 3),
                (1, 4), (2, 5), (3, 6),
                (4, 7), (5, 8), (6, 9),
                (7, 10), (8, 10), (9, 10)])
        ]
    }

    pub fn with_sampled_symbols<C: Codec>(codecs: impl IntoIterator<Item=C>) -> impl IntoIterator<Item=(C, C::Symbol)> {
        codecs.into_iter().enumerate().map(|(seed, c)| {
            let symbol = c.sample(seed);
            (c, symbol)
        })
    }

    #[test]
    fn small_shuffle_digraphs() {
        for seed in 0..20 {
            test_graph_shuffle_codecs(small_codecs_and_digraphs(), &TestConfig { pu: false, ..TestConfig::test(seed) });
        }
    }

    #[test]
    fn small_shuffle_ungraphs() {
        for seed in 0..20 {
            test_graph_shuffle_codecs(small_codecs_and_ungraphs(), &TestConfig::test(seed));
        }
    }

    #[test]
    fn sampled_shuffle_digraphs() {
        test_graph_shuffle_codecs(with_sampled_symbols(graph_codecs::<Directed>(false)), &TestConfig { pu: false, ..TestConfig::test(0) });
    }

    #[test]
    fn sampled_shuffle_ungraphs() {
        test_graph_shuffle_codecs(with_sampled_symbols(graph_codecs::<Undirected>(false)), &TestConfig::test(0));
    }

    #[test]
    fn incomplete_shuffle_coding_stochastic() {
        let graph = UnGraph::plain(8, [(0, 1), (0, 4), (0, 7), (1, 5), (1, 6), (2, 6), (3, 4), (3, 5), (4, 7)]);
        let codec = plain_erdos_renyi::<Undirected>(8, 0.4, false);
        test_graph_shuffle_codecs(vec![(codec, graph)], &TestConfig::test(0));
    }

    #[test]
    fn incomplete_shuffle_coding_no_iter() {
        for half_iter in [false, true] {
            for wl_iter in [0, 1, 2] {
                let graph = UnGraph::plain(8, [(0, 1), (0, 4), (0, 7), (1, 5), (1, 6), (2, 6), (3, 4), (3, 5), (4, 7)]);
                let codec = plain_erdos_renyi::<Undirected>(graph.len(), 0.4, false);
                let mut config = TestConfig::test(0);
                config.plain = false;
                config.plain_coset_recursive = false;
                config.wl_extra_half_iter = half_iter;
                config.wl_iter = wl_iter;
                test_graph_shuffle_codecs(vec![(codec, graph)], &config);
            }
        }
    }

    #[test]
    fn sample_polya_urn() {
        for loops in [false, true] {
            for redraws in [false] { // Sampling with redraws not supported.
                // TODO num_nodes = 6 fails because if node is sampled that already has edges to all other nodes, we get an error:
                let c = PolyaUrnEdgeIndicesCodec::<Undirected>::new(24, 9, loops, redraws);
                c.test_on_samples(50);
            }
        }
    }
}