//! Autoregressive models for graphs.
use std::cell::RefCell;
use std::marker::PhantomData;

use itertools::{Itertools, repeat_n};

use crate::codec::{Bernoulli, Codec, ConstantCodec, LogUniform, Message, MutCategorical, MutDistribution, OrdSymbol, Symbol};
use crate::graph::{Directed, EdgeIndex, EdgeType, Graph};
use crate::graph_codec::{EdgeIndicesCodec, EdgesIID, ErdosRenyi};
use crate::multiset::Multiset;
use crate::permutable::{Permutable, Unordered};
use crate::recursive::{Len, Prefix, PrefixFn, SliceCodecs, UncachedPrefixFn};
use crate::recursive::graph::GraphPrefix;
use crate::recursive::multiset::{multiset_shuffle_codec, MultisetShuffleCodec};

/// Autoregressive version of the Erdos-Renyi model for graphs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErdosRenyiSliceCodecs<NodeC: Codec + Symbol, EdgeC: Codec<Symbol: OrdSymbol> + Symbol, Ty: EdgeType> {
    pub len: usize,
    pub has_edge: Bernoulli,
    pub node: NodeC,
    pub edge: EdgeC,
    pub loops: bool,
    pub phantom: PhantomData<Ty>,
}

impl<NodeC: Codec + Symbol, EdgeC: Codec<Symbol: OrdSymbol> + Symbol, Ty: EdgeType> Len for ErdosRenyiSliceCodecs<NodeC, EdgeC, Ty> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<NodeC: Codec<Symbol: Default> + Symbol, EdgeC: Codec<Symbol: OrdSymbol> + Symbol, Ty: EdgeType> UncachedPrefixFn for ErdosRenyiSliceCodecs<NodeC, EdgeC, Ty> {
    type Prefix = GraphPrefix<NodeC::Symbol, EdgeC::Symbol, Ty>;
    type Output = (NodeC, EdgesIID<EdgeC, ErdosRenyi<Ty, Vec<EdgeIndex>>>);

    fn apply(&self, x: &Self::Prefix) -> Self::Output {
        let node = x.len();
        let unknown = node + 1..x.graph.len();
        let mut slice_indices = unknown.clone().map(|j| (node, j)).collect_vec();
        if Ty::is_directed() {
            slice_indices.extend(unknown.map(|j| (j, node)));
        }
        if self.loops {
            slice_indices.push((node, node));
        }

        let indices = ErdosRenyi::<Ty, Vec<EdgeIndex>>::new(self.has_edge.clone(), slice_indices, self.loops);
        (self.node.clone(), EdgesIID::new(indices, self.edge.clone()))
    }
}

impl<N: Symbol + Default, E: Symbol, Ty: EdgeType, GraphS: PrefixFn<Prefix=GraphPrefix<N, E, Ty>,
    Output: Codec<Symbol=<Self::Prefix as Prefix>::Slice>> + Len> SliceCodecs for GraphS {
    fn empty_prefix(&self) -> impl Codec<Symbol=Self::Prefix> {
        let graph = Graph::empty(repeat_n(N::default(), self.len()));
        ConstantCodec(Self::Prefix { graph, num_unknown_nodes: self.len() })
    }
}

#[derive(Clone)]
pub struct PolyaUrnSliceCodec<Ty: EdgeType = Directed> {
    /// RefCell to avoid need for mutable self reference in pop/push impl and Codec trait.
    pub first_edge_codec: RefCell<MutCategorical>,
    pub node: usize,
    pub loops: bool,
    pub graph_len: usize,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> PolyaUrnSliceCodec<Ty> {
    fn len_codec(&self) -> LogUniform {
        let slice_len = self.loops as usize +
            (self.graph_len - self.node - 1) * if Ty::is_directed() { 2 } else { 1 };
        LogUniform::new(LogUniform::get_bits(&slice_len) + 1)
    }

    fn from_graph<N: Symbol, E: Symbol>(prefix: &GraphPrefix<N, E, Ty>, loops: bool) -> Self {
        assert!(!Ty::is_directed(), "Autoregressive Polya model does not support directed graphs.");
        let node = prefix.len();
        let graph_len = prefix.graph.len();
        let masses = (node + !loops as usize..graph_len).
            map(|j| (j, prefix.graph.nodes[j].1.len() + 1));
        let first_edge_codec = RefCell::new(MutCategorical::new(masses));
        Self { first_edge_codec, node, graph_len, loops, phantom: PhantomData }
    }

    fn codec(&self, len: usize) -> MultisetShuffleCodec<EdgeIndex, PolyaUrnSliceEdgeIndicesCodec<Ty>> {
        assert!(len > 0);
        multiset_shuffle_codec(PolyaUrnSliceEdgeIndicesCodec {
            first_edge_codec: &self.first_edge_codec,
            len,
            node: self.node,
            phantom: self.phantom,
        })
    }
}

impl<Ty: EdgeType> Codec for PolyaUrnSliceCodec<Ty> {
    type Symbol = Multiset<EdgeIndex>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let len = x.len();
        if len > 0 {
            self.codec(len).push(m, x);
        }
        self.len_codec().push(m, &len);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let len = self.len_codec().pop(m);
        if len > 0 {
            self.codec(len).pop(m)
        } else {
            Unordered(vec![])
        }
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

impl<Ty: EdgeType> EdgeIndicesCodec for PolyaUrnSliceCodec<Ty> {
    type Ty = Ty;

    fn loops(&self) -> bool { self.loops }
}

#[derive(Clone)]
pub struct PolyaUrnSliceEdgeIndicesCodec<'a, Ty: EdgeType = Directed> {
    /// RefCell to avoid need for mutable self reference in pop/push impl and Codec trait.
    pub first_edge_codec: &'a RefCell<MutCategorical>,
    pub len: usize,
    pub node: usize,
    pub phantom: PhantomData<Ty>,
}

impl<'a, Ty: EdgeType> Codec for PolyaUrnSliceEdgeIndicesCodec<'a, Ty> {
    type Symbol = Vec<EdgeIndex>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let mut edge_codec = self.first_edge_codec.borrow_mut();
        assert_eq!(self.len, x.len());
        let masses = x.iter().map(|(node_, j)| {
            assert_eq!(self.node, *node_);
            edge_codec.remove_all(j)
        }).collect_vec();

        for ((_, j), mass) in x.iter().rev().zip_eq(masses.iter().rev()) {
            edge_codec.insert(*j, *mass);
            edge_codec.push(m, j);
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut edge_codec = self.first_edge_codec.borrow_mut();
        let mut edges = vec![];
        let mut masses = vec![];
        for _ in 0..self.len {
            let j = edge_codec.pop(m);
            masses.push(edge_codec.remove_all(&j));
            edges.push((self.node, j))
        }

        // Revert changes:
        for ((_, j), mass) in edges.iter().zip_eq(masses.into_iter()) {
            edge_codec.insert(*j, mass)
        }

        edges
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

/// Autoregressive approximation of the Polya urn model for graphs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyaUrnSliceCodecs<NodeC: Codec + Symbol, EdgeC: Codec + Symbol, Ty: EdgeType> {
    pub len: usize,
    pub node: NodeC,
    pub edge: EdgeC,
    pub loops: bool,
    pub phantom: PhantomData<Ty>,
}

impl<NodeC: Codec<Symbol: Default> + Symbol, EdgeC: Codec<Symbol: OrdSymbol> + Symbol, Ty: EdgeType> PrefixFn for PolyaUrnSliceCodecs<NodeC, EdgeC, Ty> {
    type Prefix = GraphPrefix<NodeC::Symbol, EdgeC::Symbol, Ty>;
    type Output = (NodeC, EdgesIID<EdgeC, PolyaUrnSliceCodec<Ty>>);

    fn apply(&self, x: &Self::Prefix) -> Self::Output {
        let indices = PolyaUrnSliceCodec::from_graph(x, self.loops);
        (self.node.clone(), EdgesIID::new(indices, self.edge.clone()))
    }

    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &Self::Prefix, slice: &<Self::Prefix as Prefix>::Slice) {
        let indices = &mut image.1.indices;
        indices.node -= 1;

        let new_node = indices.node + !self.loops as usize;

        let edge_codec = indices.first_edge_codec.get_mut();
        if new_node < x.graph.len() {
            edge_codec.insert(new_node, x.graph.nodes[new_node].1.len() + 1);
        }
        for ((node_, j), _) in slice.1.0.iter() {
            assert_eq!(indices.node, *node_);
            if *j != new_node {
                edge_codec.remove(j, 1);
            }
        }
    }

    fn update_after_push_slice(&self, image: &mut Self::Output, _: &Self::Prefix, slice: &<Self::Prefix as Prefix>::Slice) {
        let indices = &mut image.1.indices;
        let edge_codec = indices.first_edge_codec.get_mut();
        for ((node_, j), _) in slice.1.0.iter() {
            assert_eq!(indices.node, *node_);
            edge_codec.insert(*j, 1);
        }
        edge_codec.remove_all(&(indices.node + !self.loops as usize));
        indices.node += 1;
    }

    fn swap(&self, _: &mut Self::Output, _: usize, _: usize) {} // Invariant to swap.
}

impl<NodeC: Codec + Symbol, EdgeC: Codec<Symbol: OrdSymbol> + Symbol, Ty: EdgeType> Len for PolyaUrnSliceCodecs<NodeC, EdgeC, Ty> {
    fn len(&self) -> usize {
        self.len
    }
}