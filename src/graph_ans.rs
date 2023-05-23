use std::collections::HashSet;
use std::fmt::Debug;

use itertools::Itertools;
use lazy_static::lazy_static;

use crate::ans::{Benford, Bernoulli, Categorical, Codec, ConstantCodec, IID, Message, OptionCodec, Symbol, Uniform, VecCodec};
use crate::graph::{Edge, EdgeIndex, Graph};
use crate::multiset::{MultiSet, OrdSymbol};
use crate::shuffle_ans::{RCosetUniform, shuffle_codec, ShuffleCodec, Unordered};

pub type EmptyCodec = ConstantCodec<()>;

/// Coding edge indices independently of the i.i.d. node and edge labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphIID<
    NodeC: Codec = EmptyCodec,
    EdgeC: Codec = EmptyCodec,
    IndicesC: EdgeIndicesCodec = ErdosRenyi> where EdgeC::Symbol: OrdSymbol
{
    pub nodes: IID<NodeC>,
    pub edges: EdgesIID<EdgeC, IndicesC>,
    pub undirected: bool,
}

impl<NodeC: Codec, EdgeC: Codec, IndicesC: EdgeIndicesCodec> Codec for GraphIID<NodeC, EdgeC, IndicesC> where EdgeC::Symbol: OrdSymbol {
    type Symbol = Graph<NodeC::Symbol, EdgeC::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.edges.push(m, &self.edges(x));
        self.nodes.push(m, &x.node_labels().collect());
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let new = if self.undirected { Self::Symbol::undirected } else { Self::Symbol::new };
        new(self.nodes.pop(m), self.edges.pop(m).into_ordered())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        Some(self.nodes.bits(&x.node_labels().collect())? + self.edges.bits(&self.edges(x))?)
    }
}

impl<NodeC: Codec, EdgeC: Codec, IndicesC: EdgeIndicesCodec> GraphIID<NodeC, EdgeC, IndicesC> where EdgeC::Symbol: OrdSymbol {
    pub fn new(num_nodes: usize, edge_indices: IndicesC, node: NodeC, edge: EdgeC, undirected: bool) -> Self {
        Self { nodes: IID::new(node, num_nodes), edges: EdgesIID::new(edge_indices, edge), undirected }
    }

    fn edges(&self, x: &Graph<NodeC::Symbol, EdgeC::Symbol>) -> MultiSet<(EdgeIndex, EdgeC::Symbol)> {
        Unordered(if self.undirected { x.undirected_edges().collect_vec() } else { x.edges().collect_vec() })
    }
}

/// Coding edge indices independently of the i.i.d. labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgesIID<EdgeC: Codec = EmptyCodec, IndicesC: EdgeIndicesCodec = ErdosRenyi> {
    pub indices: IndicesC,
    pub label: EdgeC,
}

impl<EdgeC: Codec, IndicesC: EdgeIndicesCodec> Codec for EdgesIID<EdgeC, IndicesC> where EdgeC::Symbol: OrdSymbol {
    type Symbol = MultiSet<Edge<EdgeC::Symbol>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let (indices, labels) = Self::split(x);
        self.labels(indices.len()).push(m, &labels);
        self.indices.push(m, &indices);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let indices = self.indices.pop(m).canonized();
        let labels = self.labels(indices.len()).pop(m);
        Unordered(indices.into_iter().zip_eq(labels.into_iter()).collect_vec())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let (indices, edges) = Self::split(x);
        let labels_codec = self.labels(indices.len());
        Some(self.indices.bits(&indices)? + labels_codec.bits(&edges)?)
    }
}

impl<EdgeC: Codec, IndicesC: EdgeIndicesCodec> EdgesIID<EdgeC, IndicesC> {
    fn split(x: &MultiSet<Edge<<EdgeC as Codec>::Symbol>>) -> (MultiSet<EdgeIndex>, Vec<<EdgeC as Codec>::Symbol>) {
        let (indices, labels) = x.to_ordered().iter().sorted_by_key(|(i, _)| *i).cloned().unzip();
        (Unordered(indices), labels)
    }

    fn labels(&self, len: usize) -> IID<EdgeC> {
        IID::new(self.label.clone(), len)
    }

    pub fn new(indices: IndicesC, label: EdgeC) -> Self {
        Self { indices, label }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseSetIID<S: OrdSymbol> {
    pub alphabet: Vec<S>,
    pub contains: IID<Bernoulli>,
}

impl<S: OrdSymbol> Codec for DenseSetIID<S> {
    type Symbol = MultiSet<S>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.contains.push(m, &self.dense(x));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let dense = self.contains.pop(m);
        Unordered(self.alphabet.iter().zip_eq(dense).filter_map(|(i, b)| if b { Some(i.clone()) } else { None }).collect_vec())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.contains.bits(&self.dense(x))
    }
}

impl<S: OrdSymbol> DenseSetIID<S> {
    pub fn new(contains: Bernoulli, alphabet: Vec<S>) -> Self {
        Self { contains: IID::new(contains, alphabet.len()), alphabet }
    }

    fn dense(&self, x: &MultiSet<S>) -> Vec<bool> {
        let mut x = x.to_ordered().iter().cloned().collect::<HashSet<_>>();
        let as_vec = self.alphabet.iter().map(|i| x.remove(i)).collect_vec();
        assert!(x.is_empty());
        as_vec
    }
}

pub trait EdgeIndicesCodec: Codec<Symbol=MultiSet<EdgeIndex>> {
    fn loops(&self) -> bool;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErdosRenyi {
    dense: DenseSetIID<EdgeIndex>,
    loops: bool,
}

impl Codec for ErdosRenyi {
    type Symbol = MultiSet<EdgeIndex>;
    fn push(&self, m: &mut Message, x: &Self::Symbol) { self.dense.push(m, x); }
    fn pop(&self, m: &mut Message) -> Self::Symbol { self.dense.pop(m) }
    fn bits(&self, x: &Self::Symbol) -> Option<f64> { self.dense.bits(x) }
}

impl EdgeIndicesCodec for ErdosRenyi {
    fn loops(&self) -> bool { self.loops }
}

pub fn erdos_renyi_indices(num_nodes: usize, edge: Bernoulli, undirected: bool, loops: bool) -> ErdosRenyi {
    ErdosRenyi { dense: DenseSetIID::new(edge, all_edge_indices(num_nodes, undirected, loops)), loops }
}

fn all_edge_indices(num_nodes: usize, undirected: bool, loops: bool) -> Vec<EdgeIndex> {
    let mut out = if loops { (0..num_nodes).map(|i| (i, i)).collect_vec() } else { vec![] };
    let fwd = (0..num_nodes).flat_map(|j| (0..j).map(move |i| (i, j)));
    if undirected {
        out.extend(fwd);
    } else {
        out.extend(fwd.flat_map(|(i, j)| [(i, j), (j, i)]));
    }
    out
}

pub fn num_all_edge_indices(num_nodes: usize, undirected: bool, loops: bool) -> usize {
    (if loops { num_nodes } else { 0 }) + (num_nodes * num_nodes - num_nodes) / if undirected { 2 } else { 1 }
}

#[derive(Clone, Debug)]
pub struct DiffEdgesCodec<EdgeC: Codec> {
    pub is_undirected: bool,
    pub edges: IID<OptionCodec<EdgeC>>,
}

impl<EdgeC: Codec> Codec for DiffEdgesCodec<EdgeC> {
    type Symbol = Vec<Option<EdgeC::Symbol>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.edges.push(m, &self.half_if_undirected(x));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let r = self.edges.pop(m);
        if !self.is_undirected {
            return r;
        }
        r.iter().chain(r.iter()).cloned().collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.edges.bits(&self.half_if_undirected(x))
    }
}

impl<EdgeC: Codec> DiffEdgesCodec<EdgeC> {
    fn half_if_undirected(&self, x: &Vec<Option<EdgeC::Symbol>>) -> Vec<Option<EdgeC::Symbol>> {
        if !self.is_undirected {
            return x.clone();
        }

        if x.len() == 0 { vec!() } else {
            let halfs = x.chunks_exact(x.len() / 2);
            assert!(halfs.remainder().is_empty());
            let (half, half_) = halfs.collect_tuple().unwrap();
            assert_eq!(half, half_, "Directed edges cannot be encoded with an undirected edge codec.");
            half.iter().cloned().collect()
        }
    }

    #[allow(unused)]
    pub fn new(edges: IID<OptionCodec<EdgeC>>, is_undirected: bool) -> Self {
        Self { edges, is_undirected }
    }
}

#[derive(Clone, Debug)]
struct PolyaUrnEdgeIndexCodec {
    graph: Graph,
    loops: bool,
    redraws: bool,
}

impl PolyaUrnEdgeIndexCodec {
    fn new(graph: Graph, undirected: bool, loops: bool, redraws: bool) -> Self {
        assert!(undirected);
        if !loops {
            assert!(!graph.has_selfloops());
        }
        Self { graph, loops, redraws }
    }

    fn dist(&self) -> Categorical {
        Categorical::new(&self.masses())
    }

    fn masses(&self) -> Vec<usize> {
        self.graph.nodes.iter().map(|(_, es)| es.len() + 1).collect_vec()
    }

    fn dist_without(&self, node: usize) -> Categorical {
        let mut masses_ = self.masses();
        if !self.loops {
            masses_[node] = 0;
        }
        if !self.redraws {
            for (&neighbor, _) in &self.graph.nodes[node].1 {
                masses_[neighbor] = 0;
            }
        }

        Categorical::new(&masses_)
    }
}

impl Codec for PolyaUrnEdgeIndexCodec {
    type Symbol = VecEdgeIndex;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        if let [node, node_] = x[..] {
            self.dist_without(node).push(m, &node_);
            self.dist().push(m, &node);
        } else {
            panic!("Expected two nodes, got {:?}", x)
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let node = self.dist().pop(m);
        let node_ = self.dist_without(node).pop(m);
        vec![node, node_]
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        if let [node, node_] = x[..] {
            Some(self.dist().bits(&node)? + self.dist_without(node).bits(&node_)?)
        } else {
            panic!("Expected two nodes, got {:?}", x)
        }
    }
}

/// Reuse Vec's implementation of Permutable for undirected edges of a Polya Urn.
type VecEdgeIndex = Vec<usize>;

#[derive(Clone, Debug)]
pub struct PolyaUrnEdgeIndicesCodec {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub undirected: bool,
    pub loops: bool,
    pub redraws: bool,
}

impl PolyaUrnEdgeIndicesCodec {
    pub fn new(num_nodes: usize, num_edges: usize, undirected: bool, loops: bool, redraws: bool) -> Self {
        Self { num_nodes, num_edges, undirected, loops, redraws }
    }

    fn edge_codec(&self, graph: Graph) -> ShuffleCodec<PolyaUrnEdgeIndexCodec, impl Fn(&VecEdgeIndex) -> RCosetUniform + Clone> {
        shuffle_codec(PolyaUrnEdgeIndexCodec::new(graph, self.undirected, self.loops, self.redraws))
    }
}

impl Codec for PolyaUrnEdgeIndicesCodec {
    type Symbol = Vec<EdgeIndex>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(x.len(), self.num_edges);
        let graph = Graph::plain_undirected(self.num_nodes, x.iter().cloned());
        let mut edge_codec = self.edge_codec(graph);
        for (i, j) in x.iter().rev() {
            assert!(edge_codec.ordered.graph.remove_edge((*i, *j)).is_some());
            assert!(edge_codec.ordered.graph.remove_edge((*j, *i)).is_some());
            edge_codec.push(m, &Unordered(vec![*i, *j]));
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let graph = Graph::plain_empty(self.num_nodes);
        let mut edge_codec = self.edge_codec(graph);
        let mut edges = vec![];
        for _ in 0..self.num_edges {
            let edge = edge_codec.pop(m);
            if let [i, j] = edge.to_ordered()[..] {
                assert!(edge_codec.ordered.graph.insert_plain_edge((i, j)));
                assert!(edge_codec.ordered.graph.insert_plain_edge((j, i)));
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

impl EdgeIndicesCodec for PolyaUrn {
    fn loops(&self) -> bool { self.ordered.loops }
}

pub fn polya_urn(num_nodes: usize, num_edges: usize, undirected: bool, loops: bool, redraws: bool) -> PolyaUrn {
    shuffle_codec(PolyaUrnEdgeIndicesCodec::new(num_nodes, num_edges, undirected, loops, redraws))
}

type PolyaUrn = ShuffleCodec<PolyaUrnEdgeIndicesCodec, fn(&Vec<EdgeIndex>) -> RCosetUniform>;

#[derive(Clone, Debug)]
pub struct UniformParamCodec {
    pub size: Uniform,
}

impl Codec for UniformParamCodec {
    type Symbol = Uniform;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.size.push(m, &(x.size as usize));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Uniform::new(self.size.pop(m))
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.size.bits(&(x.size as usize))
    }
}

impl UniformParamCodec {
    pub fn new(size: Uniform) -> Self { Self { size } }
}

impl Default for UniformParamCodec {
    fn default() -> Self { Self::new(Uniform::max().clone()) }
}

#[derive(Clone, Debug)]
pub struct SortedDiffRunLengthCodec;

impl Codec for SortedDiffRunLengthCodec {
    /// Array of ascendingly sorted integers.
    type Symbol = Vec<usize>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(&x.iter().sorted().cloned().collect_vec(), x);
        MaxBenfordIID.push(m, &Self::diffs_lens(x));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let diffs_lens = MaxBenfordIID.pop(m);
        assert_eq!(diffs_lens.len() % 2, 0);
        let diffs = diffs_lens.iter().enumerate().filter(|&(i, _)| i % 2 == 0).map(|(_, &x)| x).collect_vec();
        let lens = diffs_lens.iter().enumerate().filter(|&(i, _)| i % 2 != 0).map(|(_, &x)| x).collect_vec();
        assert_eq!(diffs.len(), lens.len());
        let values = diffs.into_iter().scan(0, |state, x| {
            *state += x;
            Some(*state)
        }).collect_vec();
        assert_eq!(values.len(), lens.len());
        values.into_iter().zip(lens).flat_map(|(v, len)| vec![v; len]).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        MaxBenfordIID.bits(&Self::diffs_lens(x))
    }
}

impl SortedDiffRunLengthCodec {
    fn diffs_lens(x: &Vec<usize>) -> Vec<usize> {
        let run_lengths = x.iter().group_by(|&&x| x).
            into_iter().map(|(v, g)| (v, g.count())).collect_vec();
        [(0, 0)].iter().chain(run_lengths.iter()).collect_vec().windows(2).
            flat_map(|window| {
                if let [(prev, _), (next, len)] = window {
                    assert!(next > prev);
                    [next - prev, *len]
                } else { unreachable!() }
            }).collect_vec()
    }
}

impl Default for SortedDiffRunLengthCodec {
    fn default() -> Self { Self }
}

#[derive(Clone, Debug)]
pub struct MaxBenfordIID;

impl Codec for MaxBenfordIID {
    type Symbol = Vec<usize>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let len = x.len();
        let bits = Benford::get_bits(x.iter().max().unwrap());
        self.iid(len, bits).push(m, &x);
        Self::bits_codec().push(m, &bits);
        Self::len_codec().push(m, &len);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.iid(Self::len_codec().pop(m), Self::bits_codec().pop(m)).pop(m)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let len = x.len();
        let bits = Benford::get_bits(x.iter().max().unwrap());
        Some(self.iid(len, bits).bits(x)? + Self::len_codec().bits(&len)? + Self::bits_codec().bits(&bits)?)
    }
}

impl MaxBenfordIID {
    fn iid(&self, len: usize, bits: usize) -> IID<Benford> {
        IID::new(Benford::new(bits + 1), len)
    }

    fn bits_codec() -> &'static Benford {
        lazy_static! {static ref C: Benford = Benford::new(6);}
        &C
    }

    fn len_codec() -> &'static Benford {
        Benford::max()
    }
}

#[derive(Clone, Debug)]
pub struct CategoricalParamCodec;

impl Codec for CategoricalParamCodec {
    type Symbol = Categorical;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        MaxBenfordIID.push(m, &x.masses());
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Categorical::new(&MaxBenfordIID.pop(m))
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        MaxBenfordIID.bits(&x.masses())
    }
}

#[derive(Clone, Debug)]
pub struct BernoulliParamCodec;

impl Codec for BernoulliParamCodec {
    type Symbol = Bernoulli;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        CategoricalParamCodec.push(m, &x.categorical)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Bernoulli { categorical: CategoricalParamCodec.pop(m) }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        CategoricalParamCodec.bits(&x.categorical)
    }
}

#[derive(Clone, Debug)]
pub struct ErdosRenyiParamCodec {
    pub nums_nodes: Vec<usize>,
    pub undirected: bool,
    pub loops: bool,
    pub edge_codec: BernoulliParamCodec,
}

impl Codec for ErdosRenyiParamCodec {
    type Symbol = Vec<ErdosRenyi>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert!(!x.is_empty());
        self.edge_codec.push(m, &x.first().unwrap().dense.contains.item)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let edge_index = self.edge_codec.pop(m);
        self.nums_nodes.iter().map(|&num_nodes| erdos_renyi_indices(num_nodes, edge_index.clone(), self.undirected, self.loops)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.edge_codec.bits(&x.first().unwrap().dense.contains.item)
    }
}

impl ErdosRenyiParamCodec {
    pub fn new(nums_nodes: Vec<usize>, undirected: bool, loops: bool) -> Self {
        Self { nums_nodes, undirected, loops, edge_codec: BernoulliParamCodec }
    }
}

#[derive(Clone, Debug)]
pub struct PolyaUrnParamCodec {
    pub nums_nodes: Vec<usize>,
    pub nums_edges: VecCodec<Uniform>,
    pub undirected: bool,
    pub loops: bool,
    pub redraws: bool,
}

impl Codec for PolyaUrnParamCodec {
    type Symbol = Vec<PolyaUrn>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert!(x.iter().all(|x| x.ordered.undirected == self.undirected));
        assert!(x.iter().all(|x| x.ordered.loops == self.loops));
        assert!(x.iter().all(|x| x.ordered.redraws == self.redraws));
        assert_eq!(x.iter().map(|x| x.ordered.num_nodes).collect_vec(), self.nums_nodes);
        self.nums_edges.push(m, &x.iter().map(|x| x.ordered.num_edges).collect())
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let nums_edges = self.nums_edges.pop(m);

        self.nums_nodes.iter().zip_eq(nums_edges.into_iter()).map(|(num_nodes, nums_edges)|
            polya_urn(*num_nodes, nums_edges, self.undirected, self.loops, self.redraws)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.nums_edges.bits(&x.iter().map(|x| x.ordered.num_edges).collect())
    }
}

impl PolyaUrnParamCodec {
    pub fn new(nums_nodes: Vec<usize>, undirected: bool, loops: bool, redraws: bool) -> Self {
        let nums_edges = VecCodec::new(nums_nodes.iter().map(
            |n| Uniform::new(num_all_edge_indices(*n, undirected, loops) + 1)));
        Self { nums_nodes, nums_edges, undirected, loops, redraws }
    }
}

pub type EmptyParamCodec = ConstantCodec<EmptyCodec>;

#[derive(Clone, Debug)]
pub struct GraphDatasetParamCodec<
    IndicesParamCFromNumsNodesAndLoops: Fn(Vec<usize>, bool) -> IndicesParamC + Clone,
    NodeC: Symbol + Codec = EmptyCodec,
    NodeParamC: Codec<Symbol=NodeC> = EmptyParamCodec,
    EdgeC: Symbol + Codec = EmptyCodec,
    EdgeParamC: Codec<Symbol=EdgeC> = EmptyParamCodec,
    IndicesC: Symbol + EdgeIndicesCodec = ErdosRenyi,
    IndicesParamC: Codec<Symbol=Vec<IndicesC>> = ErdosRenyiParamCodec>
    where EdgeC::Symbol: OrdSymbol {
    /// Code the sequence of numbers of nodes for each graph as a categorical distribution:
    pub node: NodeParamC,
    pub edge: EdgeParamC,
    pub indices: IndicesParamCFromNumsNodesAndLoops,
    pub undirected: bool,
}

impl<
    NodeC: Symbol + Codec,
    NodeParamC: Codec<Symbol=NodeC>,
    EdgeC: Symbol + Codec,
    EdgeParamC: Codec<Symbol=EdgeC>,
    IndicesC: Symbol + EdgeIndicesCodec,
    IndicesParamC: Codec<Symbol=Vec<IndicesC>>,
    IndicesParamCFromNumsNodesAndLoops: Fn(Vec<usize>, bool) -> IndicesParamC + Clone>
Codec for GraphDatasetParamCodec<IndicesParamCFromNumsNodesAndLoops, NodeC, NodeParamC, EdgeC, EdgeParamC, IndicesC, IndicesParamC> where EdgeC::Symbol: OrdSymbol {
    type Symbol = VecCodec<GraphIID<NodeC, EdgeC, IndicesC>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert!(x.codecs.iter().all(|x| x.undirected == self.undirected));
        let first = x.codecs.first().clone().unwrap();
        self.edge.push(m, &first.edges.label);
        self.node.push(m, &first.nodes.item);
        let mut nums_nodes = x.codecs.iter().map(|x| x.nodes.len).collect_vec();
        let indices = x.codecs.iter().map(|x| x.edges.indices.clone()).collect();
        let loops = first.edges.indices.loops();
        (self.indices)(nums_nodes.clone(), loops).push(m, &indices);
        Bernoulli::new(1, 2).push(m, &loops);
        nums_nodes.reverse();
        SortedDiffRunLengthCodec::default().push(m, &nums_nodes);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut nums_nodes = SortedDiffRunLengthCodec::default().pop(m);
        nums_nodes.reverse();
        let loops = Bernoulli::new(1, 2).pop(m);
        let indices = (self.indices)(nums_nodes.clone(), loops).pop(m);
        let node = self.node.pop(m);
        let edge = self.edge.pop(m);
        VecCodec::new(nums_nodes.into_iter().zip_eq(indices.into_iter()).map(|(n, indices)|
            GraphIID::new(n, indices, node.clone(), edge.clone(), self.undirected)))
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        assert!(x.codecs.iter().all(|x| x.undirected == self.undirected));
        let first = x.codecs.first().clone().unwrap();
        let mut nums_nodes = x.codecs.iter().map(|x| x.nodes.len).collect_vec();
        nums_nodes.reverse();
        let indices = x.codecs.iter().map(|x| x.edges.indices.clone()).collect();
        let loops_bit = 1.;
        Some(SortedDiffRunLengthCodec::default().bits(&nums_nodes)? + (self.indices)(nums_nodes, first.edges.indices.loops()).bits(&indices)? +
            self.node.bits(&first.nodes.item)? + self.edge.bits(&first.edges.label)? + loops_bit)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParametrizedIndependent<
    DataC: Symbol + Codec,
    ParamC: Codec<Symbol=DataC>,
    Infer: Fn(&DataC::Symbol) -> DataC + Clone>
{
    pub param_codec: ParamC,
    pub infer: Infer,
}

impl<
    DataC: Symbol + Codec,
    ParamC: Codec<Symbol=DataC>,
    Infer: Fn(&DataC::Symbol) -> DataC + Clone>
Codec for ParametrizedIndependent<DataC, ParamC, Infer> {
    type Symbol = DataC::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let data_codec = (self.infer)(&x);
        data_codec.push(m, &x);
        self.param_codec.push(m, &data_codec);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.param_codec.pop(m).pop(m)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let data_codec = (self.infer)(&x);
        Some(data_codec.bits(x)? + self.param_codec.bits(&data_codec)?)
    }
}

pub struct WrappedParametrizedIndependent<
    InnerC: Symbol + Codec,
    ParamC: Codec<Symbol=InnerC>,
    Infer: Fn(&InnerC::Symbol) -> InnerC + Clone,
    C: Symbol + Codec>
{
    pub parametrized_codec: ParametrizedIndependent<InnerC, ParamC, Infer>,
    pub from_inner_codec: Box<dyn Fn(InnerC) -> C>,
    pub data_to_inner: Box<dyn Fn(C::Symbol) -> InnerC::Symbol>,
    pub data_from_inner: Box<dyn Fn(InnerC::Symbol) -> C::Symbol>,
}

impl<InnerC: Symbol + Codec, ParamC: Codec<Symbol=InnerC>,
    Infer: Fn(&InnerC::Symbol) -> InnerC + Clone,
    C: Symbol + Codec> Clone for WrappedParametrizedIndependent<InnerC, ParamC, Infer, C> {
    fn clone(&self) -> Self {
        unimplemented!()
    }
}

impl<InnerC: Symbol + Codec, ParamC: Codec<Symbol=InnerC>,
    Infer: Fn(&InnerC::Symbol) -> InnerC + Clone, C: Symbol + Codec>
Codec for WrappedParametrizedIndependent<InnerC, ParamC, Infer, C> {
    type Symbol = C::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let param = self.infer(&x);
        (self.from_inner_codec)(param.clone()).push(m, &x);
        self.parametrized_codec.param_codec.push(m, &param);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let param = self.parametrized_codec.param_codec.pop(m);
        (self.from_inner_codec)(param).pop(m)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let param = self.infer(x);
        Some((self.from_inner_codec)(param.clone()).bits(x)? + self.parametrized_codec.param_codec.bits(&param)?)
    }
}

impl<
    InnerC: Symbol + Codec,
    ParamC: Codec<Symbol=InnerC>,
    Infer: Fn(&InnerC::Symbol) -> InnerC + Clone,
    C: Symbol + Codec> WrappedParametrizedIndependent<InnerC, ParamC, Infer, C> {
    pub fn infer(&self, x: &<C as Codec>::Symbol) -> InnerC {
        (self.parametrized_codec.infer)(&(self.data_to_inner)(x.clone()))
    }
}

#[cfg(test)]
pub mod tests {
    use crate::graph::PartialGraph;
    use crate::permutable::{GroupPermutable, Permutable};
    use crate::shuffle_ans::interleaved::optimal_interleaved;
    use crate::shuffle_ans::test::{test_and_print_vec, test_shuffle_codecs};
    use crate::shuffle_ans::TestConfig;

    use super::*;

    pub fn erdos_renyi<NodeC: Codec, EdgeC: Codec>(n: usize, p: f64, undirected: bool, loops: bool, node: NodeC, edge: EdgeC) -> GraphIID<NodeC, EdgeC> where EdgeC::Symbol: OrdSymbol {
        let edge_indices = erdos_renyi_indices(n, Bernoulli::from_prob(p, 1 << 28), undirected, loops);
        GraphIID::new(n, edge_indices, node, edge, undirected)
    }

    pub fn plain_erdos_renyi(n: usize, p: f64, undirected: bool, loops: bool) -> GraphIID {
        erdos_renyi(n, p, undirected, loops, EmptyCodec::default(), EmptyCodec::default())
    }

    pub fn test_graph_shuffle_codecs<NodeC: Codec, EdgeC: Codec>(
        codecs_and_graphs: impl IntoIterator<Item=(GraphIID<NodeC, EdgeC>, Graph<NodeC::Symbol, EdgeC::Symbol>)>,
        config: &TestConfig) where
        NodeC::Symbol: OrdSymbol, EdgeC::Symbol: OrdSymbol,
        <GraphIID<NodeC, EdgeC> as Codec>::Symbol: GroupPermutable,
        PartialGraph<NodeC::Symbol, EdgeC::Symbol>: GroupPermutable
    {
        let (codecs, graphs): (Vec<_>, Vec<_>) = codecs_and_graphs.into_iter().unzip();
        let unordered = &graphs.iter().map(|x| Unordered(x.clone())).collect();
        test_shuffle_codecs(&codecs, unordered, config);
        if config.interleaved {
            let interleaved_codecs = codecs.into_iter().map(|c|
                optimal_interleaved::<PartialGraph<_, _>, _, _>(c.nodes.len, move |len| {
                    let edge = OptionCodec { is_some: c.edges.indices.dense.contains.item.clone(), some: c.edges.label.clone() };
                    let dense_edges = IID::new(edge, len * if c.undirected { 1 } else { 2 });
                    (c.nodes.item.clone(), DiffEdgesCodec::new(dense_edges, c.undirected))
                }));
            test_and_print_vec(interleaved_codecs, unordered, &config.initial_message());
        }
        println!();
    }

    #[test]
    pub fn param_codecs() {
        UniformParamCodec::new(Uniform::new(10)).test(&Uniform::new(9), &Message::random(0));

        let label = Categorical::new(&vec!(1, 2, 3, 0, 0, 27, 54));
        CategoricalParamCodec.test(&label, &Message::random(0));
        let edge_index = Bernoulli::from_prob(0.4, 50);
        BernoulliParamCodec.test(&edge_index, &Message::random(0));
        let indices = erdos_renyi_indices(10, edge_index.clone(), false, false);
        let indices_param = |nums_nodes, loops|
            ErdosRenyiParamCodec { nums_nodes, undirected: false, loops, edge_codec: BernoulliParamCodec };
        indices_param(vec![10], false).test(&vec![indices.clone()], &Message::random(0));
        let graphs = VecCodec::new(vec![GraphIID::new(10, indices, EmptyCodec::default(), label.clone(), true)]);
        GraphDatasetParamCodec {
            node: ConstantCodec(EmptyCodec::default()),
            edge: CategoricalParamCodec,
            indices: indices_param,
            undirected: true,
        }.test(&graphs, &Message::random(0));
    }

    pub fn graph_codecs(undirected: bool, loops: bool) -> impl Iterator<Item=GraphIID> {
        (0..20).map(move |len| plain_erdos_renyi(len, 0.4, undirected, loops))
    }

    fn small_codecs_and_graphs() -> impl Iterator<Item=(GraphIID, Graph)> {
        small_graphs().into_iter().map(|x| (plain_erdos_renyi(x.len(), 0.1, false, false), x))
    }

    pub fn small_graphs() -> impl Iterator<Item=Graph> {
        [
            Graph::plain(0, vec![]),
            Graph::plain(1, vec![]),
            Graph::plain(3, vec![(0, 1)]),
            Graph::plain(4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]),
            Graph::plain(11, vec![
                (0, 1), (0, 2), (0, 3),
                (1, 4), (2, 5), (3, 6),
                (4, 7), (5, 8), (6, 9),
                (7, 10), (8, 10), (9, 10)])
        ].into_iter().chain(small_undirected_graphs())
    }

    fn small_undirected_codecs_and_graphs() -> impl Iterator<Item=(GraphIID, Graph)> {
        small_undirected_graphs().into_iter().map(|x| (plain_erdos_renyi(x.len(), 0.1, true, false), x))
    }

    fn small_undirected_graphs() -> impl IntoIterator<Item=Graph> {
        [
            Graph::plain_undirected(0, vec![]),
            Graph::plain_undirected(1, vec![]),
            Graph::plain_undirected(3, vec![(0, 1)]),
            Graph::plain_undirected(4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]),
            Graph::plain_undirected(11, vec![
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
    fn small_shuffle() {
        for seed in 0..20 {
            test_graph_shuffle_codecs(small_codecs_and_graphs(), &TestConfig::test(seed));
        }
    }

    #[test]
    fn small_undirected_shuffle() {
        for seed in 0..20 {
            test_graph_shuffle_codecs(small_undirected_codecs_and_graphs(), &TestConfig::test(seed));
        }
    }

    #[test]
    fn sampled_shuffle() {
        test_graph_shuffle_codecs(with_sampled_symbols(graph_codecs(false, false)), &TestConfig::test(0));
    }

    #[test]
    fn sampled_undirected_shuffle() {
        test_graph_shuffle_codecs(with_sampled_symbols(graph_codecs(true, false)), &TestConfig::test(0));
    }

    #[test]
    fn sorted_diff_run_length() {
        let v = vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 26];
        SortedDiffRunLengthCodec::default().test(&v, &Message::random(0));
    }
}
