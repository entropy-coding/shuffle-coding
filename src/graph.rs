use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::marker::PhantomData;
use std::os::raw::c_int;

use itertools::{Itertools, repeat_n};
use nauty_Traces_sys::{FALSE, nauty_check, NAUTYVERSIONID, optionblk, ran_init, SETWORDSNEEDED, SparseGraph,
                       sparsegraph, sparsenauty, statsblk, Traces, TracesOptions, TracesStats, TRUE, WORDSIZE};

use crate::codec::Symbol;
use crate::permutable::{Permutable, Permutation};
use crate::plain::{Automorphisms, Orbits, PermutationGroup, PlainPermutable};

pub type EdgeIndex = (usize, usize);
pub type Edge<E = ()> = (EdgeIndex, E);
/// We need deterministic results for getting the automorphism group generators from nauty.
/// For this reason, the list of neighbors we pass to nauty needs to be canonized, i.e. sorted.
/// Using BTreeMap instead of a HashMap avoids the cost for sorting within the codecs.
pub type NeighborMap<E> = BTreeMap<usize, E>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum AutomorphismsBackend { SparseNauty, Traces }

thread_local! {
    /// The backend used to compute undirected graph automorphisms and isomorphisms.
    /// Directed graphs are always handled by nauty.
    pub static AUTOMORPHISMS_BACKEND: RefCell<AutomorphismsBackend> = RefCell::new(AutomorphismsBackend::SparseNauty);
}

#[allow(unused)]
pub fn with_automorphisms_backend<F: FnOnce() -> R, R>(backend: AutomorphismsBackend, f: F) -> R {
    AUTOMORPHISMS_BACKEND.with(|b| {
        let old = b.replace(backend);
        let out = f();
        assert_eq!(backend, b.replace(old));
        out
    })
}

/// Marker type for a directed graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Directed {}

/// Marker type for an undirected graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Undirected {}

/// A graph's edge type determines whether it has directed edges or not.
pub trait EdgeType: Symbol {
    fn is_directed() -> bool;
}

impl EdgeType for Directed {
    #[inline]
    fn is_directed() -> bool {
        true
    }
}

impl EdgeType for Undirected {
    #[inline]
    fn is_directed() -> bool {
        false
    }
}

/// Ordered graph with optional node and edge labels.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Graph<N: Symbol = (), E: Symbol = (), Ty: EdgeType = Directed> {
    pub nodes: Vec<(N, NeighborMap<E>)>,
    pub phantom: PhantomData<Ty>,
}

pub type DiGraph<N = (), E = ()> = Graph<N, E, Directed>;
pub type UnGraph<N = (), E = ()> = Graph<N, E, Undirected>;
pub type PlainGraph<Ty = Directed> = Graph<(), (), Ty>;

impl<N: Symbol, E: Symbol> DiGraph<N, E> {
    pub fn into_undirected(self) -> UnGraph<N, E> {
        let out = UnGraph { nodes: self.nodes, phantom: PhantomData };
        out.verify_is_undirected();
        out
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Graph<N, E, Ty> {
    pub fn from_neighbors(neighbors: impl IntoIterator<Item=(N, NeighborMap<E>)>) -> Self {
        let out = Self { nodes: neighbors.into_iter().collect(), phantom: PhantomData };
        if !Ty::is_directed() {
            out.verify_is_undirected();
        }
        out
    }

    fn verify_is_undirected(&self) {
        assert!(self.nodes.iter().enumerate().all(|(i, (_, ne))|
        ne.iter().all(|(j, l)| self.nodes[*j].1.get(&i) == Some(l))));
    }

    pub fn edges(&self) -> Vec<Edge<E>> {
        if Ty::is_directed() {
            self.directed_edges().collect()
        } else {
            self.directed_edges().filter(|&((i, j), _)| i <= j).collect()
        }
    }

    pub fn insert_edge(&mut self, (i, j): EdgeIndex, e: E) -> Option<E> {
        let out = self.insert_directed_edge((i, j), e.clone());
        if !Ty::is_directed() && i != j {
            assert_eq!(out, self.insert_directed_edge((j, i), e));
        }
        out
    }

    pub fn remove_edge(&mut self, (i, j): EdgeIndex) -> Option<E> {
        let out = self.remove_directed_edge((i, j));
        if !Ty::is_directed() && i != j {
            assert_eq!(out, self.remove_directed_edge((j, i)));
        }
        out
    }

    pub fn empty(nodes: impl IntoIterator<Item=N>) -> Self {
        Self::from_neighbors(nodes.into_iter().map(|n| (n, NeighborMap::new())))
    }

    pub fn new(nodes: impl IntoIterator<Item=N>, edges: impl IntoIterator<Item=Edge<E>>) -> Self {
        let mut x = Self::empty(nodes);
        for (e, l) in edges {
            assert!(x.insert_edge(e, l).is_none(), "Duplicate edge {:?}", e);
        }
        x
    }

    #[allow(unused)]
    pub fn edge_indices(&self) -> Vec<EdgeIndex> {
        self.edges().into_iter().map(|(i, _)| i).collect()
    }

    pub fn edge_labels(&self) -> Vec<E> {
        self.edges().into_iter().map(|(_, e)| e).collect()
    }

    pub fn edge(&self, (i, j): EdgeIndex) -> Option<E> {
        self.nodes[i].1.get(&j).cloned()
    }

    fn insert_directed_edge(&mut self, (i, j): EdgeIndex, e: E) -> Option<E> {
        assert!(j < self.len());
        self.nodes[i].1.insert(j, e)
    }

    fn remove_directed_edge(&mut self, (i, j): EdgeIndex) -> Option<E> {
        self.nodes[i].1.remove(&j)
    }

    fn directed_edges(&self) -> impl Iterator<Item=Edge<E>> + '_ {
        self.nodes.iter().enumerate().flat_map(|(i, (_, ne))|
        ne.iter().map(move |(&j, l)| ((i, j), l.clone())))
    }

    pub fn node_labels(&self) -> impl Iterator<Item=N> + '_ {
        self.nodes.iter().map(|(n, _)| n.clone())
    }

    pub fn has_selfloops(&self) -> bool {
        self.nodes.iter().enumerate().any(|(i, (_, ne))| ne.contains_key(&i))
    }

    /// Automorphisms of the corresponding unlabelled graph (ignoring node and edge labels),
    /// respecting the given partitions.
    pub fn unlabelled_automorphisms(&self, partitions: Option<impl IntoIterator<Item=BTreeSet<usize>>>) -> Automorphisms {
        if self.len() == 0 { // Traces breaks on graph size 0.
            return Automorphisms {
                group: PermutationGroup::new(0, vec![]),
                canon: Permutation::identity(0),
                decanon: Permutation::identity(0),
                orbits: vec![],
                bits: 0.,
            };
        }

        let defaultptn = if partitions.is_none() { TRUE } else { FALSE };

        let (mut lab, mut ptn) = if let Some(partitions) = partitions {
            partitions.into_iter().flat_map(|p| {
                p.into_iter().enumerate().rev().
                    map(|(i, x)| (x as i32, if i == 0 { 0 } else { 1 }))
            }).unzip()
        } else { (vec![0; self.len()], vec![0; self.len()]) };
        let mut orbs = vec![0; self.len()];

        unsafe {
            nauty_check(WORDSIZE as c_int, SETWORDSNEEDED(self.len()) as c_int,
                        self.len() as c_int, NAUTYVERSIONID as c_int);
        }

        let sg = &mut self.to_nauty();
        let lab_ptr = lab.as_mut_ptr();
        let ptn_ptr = ptn.as_mut_ptr();
        let orbs_ptr = orbs.as_mut_ptr();

        thread_local! {
            /// Collect generators via static C callback function:
            static GENERATORS: RefCell<Vec<Permutation>> = RefCell::new(vec![]);
        }
        extern "C" fn push_generator(ordinal: c_int, perm: *mut c_int, n: c_int) {
            let generator = Permutation::from((0..n).map(
                |i| unsafe { *perm.offset(i as isize) } as usize).collect());
            GENERATORS.with(|g| {
                let mut generators = g.borrow_mut();
                generators.push(generator);
                assert_eq!(ordinal as usize, generators.len());
            });
        }
        extern "C" fn push_generator_from_nauty(ordinal: c_int, perm: *mut c_int, _orbits: *mut c_int,
                                                _numorbits: c_int, _stabnode: c_int, n: c_int) {
            push_generator(ordinal, perm, n);
        }

        let (grpsize1, grpsize2) = if
        AUTOMORPHISMS_BACKEND.with_borrow(|b| b == &AutomorphismsBackend::Traces) && !Ty::is_directed() {
            let options = &mut TracesOptions::default();
            options.getcanon = TRUE;
            options.userautomproc = Some(push_generator);
            options.defaultptn = defaultptn;
            let stats = &mut TracesStats::default();
            unsafe {
                ran_init(0);
                Traces(&mut sg.into(), lab_ptr, ptn_ptr, orbs_ptr,
                       options, stats, std::ptr::null_mut());
            }
            (stats.grpsize1, stats.grpsize2)
        } else {
            let options = &mut if Ty::is_directed() {
                optionblk::default_sparse_digraph()
            } else {
                optionblk::default_sparse()
            };
            options.getcanon = TRUE;
            options.userautomproc = Some(push_generator_from_nauty);
            options.defaultptn = defaultptn;
            let stats = &mut statsblk::default();
            thread_local! {
                /// Avoid canonized graph allocation for every call, nauty allows reuse:
                static CG: RefCell<sparsegraph> = RefCell::new(sparsegraph::default());
            }
            CG.with(|cg| unsafe {
                sparsenauty(&mut sg.into(), lab_ptr, ptn_ptr, orbs_ptr,
                            options, stats, &mut *cg.borrow_mut())
            });
            (stats.grpsize1, stats.grpsize2)
        };

        let generators = GENERATORS.with(|g| {
            let mut gens = g.borrow_mut();
            let out = gens.clone();
            gens.clear();
            out
        });

        let decanon = Permutation::from(lab.into_iter().map(|x| x as usize).collect());
        let canon = decanon.inverse();
        let orbits = orbs.orbits();
        let grpsize2: f64 = grpsize2.try_into().unwrap();
        let bits: f64 = grpsize1.log2() + grpsize2 * 10f64.log2();
        let group = PermutationGroup::new(self.len(), generators);
        Automorphisms { group, canon, decanon, orbits, bits }
    }

    fn to_nauty(&self) -> SparseGraph {
        let d = self.degrees().map(|x| x as i32).collect_vec();
        let v = d.iter().map(|d| *d as usize).scan(0, |acc, d| {
            let out = Some(*acc);
            *acc += d;
            out
        }).collect();
        let e = self.nodes.iter().map(|(_, ne)| ne.iter().map(
            |(i, _)| *i as i32).collect()).concat();
        SparseGraph { v, d, e }
    }

    pub fn degrees(&self) -> impl Iterator<Item=usize> + '_ {
        self.nodes.iter().map(|(_, ne)| ne.len())
    }

    pub fn num_edges(&self) -> usize {
        let e = self.degrees().sum();
        if Ty::is_directed() { e } else { (e + self.num_selfloops()) / 2 }
    }

    pub fn num_selfloops(&self) -> usize {
        self.nodes.iter().enumerate().filter(|(i, (_, ne))| ne.get(&i).is_some()).count()
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Permutable for Graph<N, E, Ty> {
    fn len(&self) -> usize { self.nodes.len() }

    fn swap(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }

        if Ty::is_directed() {
            *self = self.permuted(&Permutation::create_swap(self.len(), i, j));
            return;
        }

        self.nodes.swap(i, j);
        let nei = self.nodes[i].1.clone();
        let nej = self.nodes[j].1.clone();
        let mut neighbors = nei.keys().chain(nej.keys()).collect::<BTreeSet<_>>();
        neighbors.insert(&i);
        neighbors.insert(&j);
        for n in neighbors {
            let ne = &mut self.nodes[*n].1;
            let ei = ne.remove(&j);
            let ej = ne.remove(&i);
            if let Some(ei) = ei {
                ne.insert(i, ei);
            }
            if let Some(ej) = ej {
                ne.insert(j, ej);
            }
        }
    }

    fn permuted(&self, p: &Permutation) -> Self {
        assert_eq!(self.len(), p.len);
        return Graph::from_neighbors((p * &self.nodes).into_iter().
            map(|(n, ne)| (n, ne.into_iter().
                map(|(j, l)| (p * j, l)).collect())));
    }
}

thread_local! {
    static ISOMORPHISM_TEST_MAX_LEN: RefCell<usize> = RefCell::new(usize::MAX);
}

/// Runs the given function with isomorphism tests for graphs above this length disabled,
/// with is_isomorphic returning true for any such graphs of the same length.
/// Useful when testing unordered graph codecs where equality checks can become too slow.
pub fn with_isomorphism_test_max_len<R>(len: usize, f: impl FnOnce() -> R) -> R {
    ISOMORPHISM_TEST_MAX_LEN.with(|l| {
        let old = l.replace(len);
        let out = f();
        assert_eq!(l.replace(old), len);
        out
    })
}

pub trait GraphPlainPermutable: Permutable {
    fn auts(&self) -> Automorphisms;
}

impl<T: GraphPlainPermutable> PlainPermutable for T {
    fn automorphism_group(&self) -> PermutationGroup { self.auts().group }
    fn canon(&self) -> Permutation { self.auts().canon }
    fn orbits(&self) -> Orbits { self.auts().orbits }
    fn automorphisms(&self) -> Automorphisms { self.auts() }

    fn is_isomorphic(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        if ISOMORPHISM_TEST_MAX_LEN.with_borrow(|l| *l) < self.len() {
            return true;
        }

        self.canonized().0 == other.canonized().0
    }
}

impl<Ty: EdgeType> GraphPlainPermutable for PlainGraph<Ty> {
    fn auts(&self) -> Automorphisms { self.unlabelled_automorphisms(None::<Vec<_>>) }
}

impl<Ty: EdgeType> PlainGraph<Ty> {
    pub fn plain_empty(len: usize) -> Self {
        Self::empty(repeat_n((), len))
    }
}

impl PlainGraph {
    pub fn plain(len: usize, edges: impl IntoIterator<Item=EdgeIndex>) -> Self {
        Graph::new(repeat_n((), len), edges.into_iter().map(|i| (i, ())))
    }
}

impl<N: Symbol, Ty: EdgeType> Graph<N, (), Ty> {
    /// Returns true if edge was inserted, false if it already existed.
    pub fn insert_plain_edge(&mut self, e: (usize, usize)) -> bool {
        self.insert_edge(e, ()).is_none()
    }

    /// Returns true if edge was removed, false if it didn't exist.
    pub fn remove_plain_edge(&mut self, e: (usize, usize)) -> bool {
        self.remove_edge(e).is_some()
    }
}

impl UnGraph {
    #[allow(unused)]
    pub fn plain(len: usize, undirected_edges: impl IntoIterator<Item=EdgeIndex>) -> Self {
        DiGraph::plain(len, undirected_edges.into_iter().flat_map(|(i, j)| [(i, j), (j, i)])).into_undirected()
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Debug for Graph<N, E, Ty> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let edges = self.edges();

        fn universal_symbol<S: Symbol>(items: impl IntoIterator<Item=S>) -> Option<S> {
            let mut iter = items.into_iter();
            let first = iter.next()?;
            iter.all(|x| x == first).then(|| first)
        }

        let universal_node_label = universal_symbol(self.node_labels());
        let universal_edge_label = universal_symbol(edges.iter().map(|(_, l)| l.clone()));
        let nodes_str = if let Some(universal_node_label) = universal_node_label {
            format!("all {:?}", universal_node_label)
        } else {
            format!("{:?}", self.node_labels().collect_vec())
        };
        let edges_str = if let Some(universal_edge_label) = universal_edge_label {
            format!("all {:?} at {:?}", universal_edge_label, edges.iter().map(|(e, _)| e.clone()).collect_vec())
        } else {
            format!("{:?}", edges)
        };

        write!(f, "N={}: {nodes_str}, E={}{}: {edges_str}", self.len(), edges.len(), if Ty::is_directed() { "" } else { "u" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_automorphism() {
        assert_eq!(DiGraph::plain(0, []).automorphisms().group.generators, vec![]);
        assert_eq!(DiGraph::plain(1, []).automorphisms().group.generators, vec![]);
    }

    #[test]
    fn tiny_directed_automorphism() {
        let g1 = DiGraph::plain(3, [(0, 1)]);
        let g2 = DiGraph::plain(3, [(2, 1)]);
        assert!(g1.is_isomorphic(&g2));
    }
}