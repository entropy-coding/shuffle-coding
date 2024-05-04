//! Recursive shuffle coding for graphs. This file implements prefixes on graphs. 
use std::collections::BTreeSet;
use std::mem;
use std::ops::Range;

use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::codec::{OrdSymbol, Symbol};
use crate::graph::{Directed, Edge, EdgeType, Graph, GraphPlainPermutable};
use crate::multiset::Multiset;
use crate::permutable::{Permutable, Unordered};
use crate::plain::{Automorphisms, PlainPermutable};
use crate::recursive::joint::JointPrefix;
use crate::recursive::Prefix;

pub mod incomplete;
pub mod slice;

/// Graph where the edge structure between the last num_hidden_nodes nodes is unknown.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphPrefix<N: Symbol = (), E: Symbol = (), Ty: EdgeType = Directed> {
    pub graph: Graph<N, E, Ty>,
    pub num_unknown_nodes: usize,
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Permutable for GraphPrefix<N, E, Ty> {
    fn len(&self) -> usize {
        self.graph.len() - self.num_unknown_nodes
    }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len());
        assert!(j < self.len());
        self.graph.swap(i, j);
    }
}

impl<N: OrdSymbol, Ty: EdgeType> GraphPlainPermutable for GraphPrefix<N, (), Ty> {
    fn auts(&self) -> Automorphisms {
        let known_partitions = self.graph.node_labels().take(self.len()).
            collect_vec().orbits().into_iter();
        let unknown_partitions = self.unknown_nodes().
            map(|i| BTreeSet::from_iter(vec![i]));
        let partitions = known_partitions.chain(unknown_partitions);
        let mut a = self.graph.unlabelled_automorphisms(Some(partitions));
        a.group.adjust_len(self.len());
        a
    }
}

impl<N: Symbol + Default, E: Symbol, Ty: EdgeType> Prefix for GraphPrefix<N, E, Ty> {
    type Full = Graph<N, E, Ty>;
    type Slice = (N, Multiset<Edge<E>>);

    fn pop_slice(&mut self) -> Self::Slice {
        self.num_unknown_nodes += 1;
        let new_unknown_node = self.len();
        let mut edges = self.graph.nodes[new_unknown_node].1.iter().
            filter(|(j, _)| self.unknown_nodes().contains(j)).
            map(|(j, l)| ((new_unknown_node, *j), l.clone())).
            collect_vec();
        for (e, l) in &edges {
            assert_eq!(l, &self.graph.remove_edge(*e).unwrap());
        }

        if Ty::is_directed() {
            // TODO OPTIMIZE Store reverse edges to avoid quadratic runtime for sparse graphs:
            edges.extend(self.unknown_nodes().
                filter_map(|i| {
                    let edge = (i, new_unknown_node);
                    self.graph.remove_edge(edge).map(|label| (edge, label))
                }));
        }

        // TODO remove once MutCategorical tree is balanced on update:
        edges.shuffle(&mut thread_rng());
        (mem::take(&mut self.graph.nodes[new_unknown_node].0), Unordered(edges))
    }

    fn push_slice(&mut self, (node, Unordered(edges)): &Self::Slice) {
        let new_known_node = self.len();
        let old = mem::replace(&mut self.graph.nodes[new_known_node].0, node.clone());
        assert_eq!(old, N::default());

        for ((i, j), e) in edges {
            assert!(self.unknown_nodes().contains(i));
            assert!(self.unknown_nodes().contains(j));
            assert!(*i == new_known_node || *j == new_known_node);
            assert!(self.graph.insert_edge((*i, *j), e.clone()).is_none());
        }

        self.num_unknown_nodes -= 1;
    }

    fn from_full(graph: Self::Full) -> Self {
        Self { graph, num_unknown_nodes: 0 }
    }
    fn into_full(self) -> Self::Full {
        assert_eq!(0, self.num_unknown_nodes);
        self.graph
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> GraphPrefix<N, E, Ty> {
    pub fn unknown_nodes(&self) -> Range<usize> {
        self.len()..self.graph.len()
    }
}

impl<N: Symbol + Default, E: Symbol, Ty: EdgeType> GraphPlainPermutable for JointPrefix<Graph<N, E, Ty>>
where
    GraphPrefix<N, E, Ty>: PlainPermutable,
{
    fn auts(&self) -> Automorphisms {
        let mut prefix = GraphPrefix::from_full(self.full.clone());
        prefix.num_unknown_nodes = self.full.len() - self.len();
        prefix.automorphisms()
    }
}