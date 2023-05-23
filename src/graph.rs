use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::iter::repeat;
use std::ops::Range;
use std::os::raw::c_int;

use itertools::Itertools;
use nauty_Traces_sys::{FALSE, nauty_check, NAUTYVERSIONID, optionblk, ran_init, SETWORDSNEEDED, SparseGraph, sparsegraph, sparsenauty, statsblk, Traces, TracesOptions, TracesStats, TRUE, WORDSIZE};

use crate::ans::Symbol;
use crate::permutable::{Automorphisms, GroupPermutable, GroupPermutableFromFused, Partition, Permutable, Permutation, PermutationGroup};
use crate::permutable::Partial;

pub type EdgeIndex = (usize, usize);
pub type Edge<E = ()> = (EdgeIndex, E);
/// We need deterministic results for getting the automorphism group generators from nauty.
/// For this reason, the list of neighbors we pass to nauty needs to be canonized, i. e. sorted.
/// Using BTreeMap instead of a HashMap avoid the cost for sorting within the codecs.
pub type NeighborMap<E> = BTreeMap<usize, E>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AutomorphismsBackend { SparseNauty, Traces }

/// Ordered directed graph with optional node and edge labels.
#[derive(Clone)]
pub struct Graph<N: Symbol = (), E: Symbol = ()> {
    pub nodes: Vec<(N, NeighborMap<E>)>,
    pub preferred_aut_backend: AutomorphismsBackend,
}

impl<N: Symbol, E: Symbol> PartialEq for Graph<N, E> {
    fn eq(&self, other: &Self) -> bool {
        self.nodes == other.nodes
    }
}

impl<N: Symbol, E: Symbol> Eq for Graph<N, E> {}

impl<N: Symbol, E: Symbol> Graph<N, E> {
    pub fn from_neighbors(neighbors: impl IntoIterator<Item=(N, NeighborMap<E>)>) -> Self {
        Graph { nodes: neighbors.into_iter().collect(), preferred_aut_backend: AutomorphismsBackend::SparseNauty }
    }

    pub fn new(nodes: impl IntoIterator<Item=N>, edges: impl IntoIterator<Item=Edge<E>>) -> Self {
        let mut x = Self::empty(nodes);
        for (e, l) in edges {
            x.insert_edge(e, l);
            // TODO: fails for many datasets including IMDB-BINARY:
            //assert!(x.insert_edge(e, l).is_none();, "Duplicate edge {:?}", e);
        }
        x
    }

    pub fn undirected(nodes: impl IntoIterator<Item=N>, undirected_edges: impl IntoIterator<Item=Edge<E>>) -> Self {
        Self::new(nodes, undirected_edges.into_iter().flat_map(|(e, l)| [(e, l.clone()), ((e.1, e.0), l)]))
    }

    pub fn empty(nodes: impl IntoIterator<Item=N>) -> Self {
        Self::from_neighbors(nodes.into_iter().map(|n| (n, NeighborMap::new())))
    }

    pub fn edge(&self, (i, j): EdgeIndex) -> Option<E> {
        self.nodes[i].1.get(&j).cloned()
    }

    pub fn insert_edge(&mut self, (i, j): EdgeIndex, e: E) -> Option<E> {
        assert!(j < self.len());
        self.nodes[i].1.insert(j, e)
    }

    pub fn remove_edge(&mut self, (i, j): EdgeIndex) -> Option<E> {
        self.nodes[i].1.remove(&j)
    }

    pub fn edges(&self) -> impl Iterator<Item=Edge<E>> + '_ {
        self.nodes.iter().enumerate().flat_map(|(i, (_, ne))|
            ne.iter().map(move |(&j, l)| ((i, j), l.clone())))
    }

    #[allow(unused)]
    pub fn edge_indices(&self) -> impl Iterator<Item=EdgeIndex> + '_ {
        self.edges().map(|(i, _)| i)
    }

    pub fn edge_labels(&self) -> impl Iterator<Item=E> + '_ {
        self.edges().map(|(_, e)| e)
    }

    pub fn undirected_edges(&self) -> impl Iterator<Item=(EdgeIndex, E)> + '_ {
        assert!(self.is_undirected());
        self.edges().filter(|&((i, j), _)| i <= j)
    }

    pub fn node_labels(&self) -> impl Iterator<Item=N> + '_ {
        self.nodes.iter().map(|(n, _)| n.clone())
    }

    pub fn is_undirected(&self) -> bool {
        self.nodes.iter().enumerate().all(|(i, (_, ne))|
            ne.iter().all(|(j, l)| self.nodes[*j].1.get(&i) == Some(l)))
    }

    pub fn has_selfloops(&self) -> bool {
        self.nodes.iter().enumerate().any(|(i, (_, ne))| ne.contains_key(&i))
    }

    /// Automorphisms of the corresponding unlabelled graph (ignoring node and edge labels),
    /// respecting the given partitions.
    pub fn unlabelled_automorphisms(&self, partitions: Option<impl IntoIterator<Item=Partition>>) -> Automorphisms {
        // TODO This is needed while Traces breaks on graphs with size 0.
        if self.len() == 0 {
            return Automorphisms {
                group: PermutationGroup::new(0, vec![]),
                canon: Permutation::identity(),
                decanon: Permutation::identity(),
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
                |i| unsafe { *perm.offset(i as isize) } as usize));
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

        // OPTIMIZE: precompute undirectedness:
        let is_undirected = self.is_undirected();
        let (grpsize1, grpsize2) = if
        self.preferred_aut_backend == AutomorphismsBackend::Traces && is_undirected {
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
            let options = &mut if is_undirected {
                optionblk::default_sparse()
            } else {
                optionblk::default_sparse_digraph()
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

        let decanon = Permutation::from(lab.into_iter().map(|x| x as usize));
        let canon = decanon.inverse();
        let orbits = orbs.orbits();
        let grpsize2: f64 = grpsize2.try_into().unwrap();
        let bits: f64 = grpsize1.log2() + grpsize2 * (10f64).log2();
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

    pub fn num_edges(&self, undirected: bool) -> usize {
        let e = self.degrees().sum();

        if undirected {
            assert!(self.is_undirected());
            (e + self.num_loops()) / 2
        } else { e }
    }

    pub fn num_loops(&self) -> usize {
        self.nodes.iter().enumerate().filter(|(i, (_, ne))| ne.get(&i).is_some()).count()
    }
}

impl<N: Symbol, E: Symbol> Permutable for Graph<N, E> {
    fn len(&self) -> usize { self.nodes.len() }

    fn permuted(&self, p: &Permutation) -> Self {
        // OPTIMIZE: Consider adding permutation to Graph struct.
        p.assert_len(self.len());
        Graph::from_neighbors((p * &self.nodes).into_iter().
            map(|(n, ne)| (n, ne.into_iter().
                map(|(j, l)| (p * j, l)).collect())))
    }
}

impl GroupPermutableFromFused for Graph {
    fn auts(&self) -> Automorphisms { self.unlabelled_automorphisms(None::<Vec<_>>) }
}

impl Graph {
    pub fn plain_empty(len: usize) -> Graph {
        Self::plain(len, vec![])
    }

    pub fn plain_undirected(len: usize, undirected_edges: impl IntoIterator<Item=EdgeIndex>) -> Graph {
        Self::plain(len, undirected_edges.into_iter().flat_map(|(i, j)| [(i, j), (j, i)]))
    }

    pub fn plain(len: usize, edges: impl IntoIterator<Item=EdgeIndex>) -> Self {
        Graph::new(repeat(()).take(len),
                   edges.into_iter().map(|i| (i, ())))
    }
}

impl<N: Symbol> Graph<N, ()> {
    /// Returns true if edge was inserted, false if it already existed.
    pub fn insert_plain_edge(&mut self, e: (usize, usize)) -> bool {
        self.insert_edge(e, ()).is_none()
    }
}

impl<N: Symbol, E: Symbol> Debug for Graph<N, E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let undirected = self.is_undirected();
        let edges = if undirected {
            self.edges().into_iter().filter_map(
                |(e, l)| if e.0 <= e.1 { Some((e, l)) } else { None }).collect_vec()
        } else { self.edges().collect_vec() };

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

        write!(f, "N={}: {nodes_str}, E={}{}: {edges_str}", self.len(), edges.len(), if undirected { "u" } else { "" })
    }
}


/// Graph where the edge structure between the last num_hidden_nodes nodes is unknown.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PartialGraph<N: Symbol = (), E: Symbol = ()> {
    pub graph: Graph<Option<N>, E>,
    pub num_unknown_nodes: usize,
}

impl<N: Symbol, E: Symbol> Permutable for PartialGraph<N, E> {
    fn len(&self) -> usize {
        self.graph.len() - self.num_unknown_nodes
    }

    fn permuted(&self, p: &Permutation) -> Self {
        p.assert_len(self.len());
        let graph = p * &self.graph;
        Self { graph, num_unknown_nodes: self.num_unknown_nodes }
    }
}

impl<N: Symbol + Ord + Hash> GroupPermutableFromFused for PartialGraph<N> {
    fn auts(&self) -> Automorphisms {
        let known_partitions = self.graph.node_labels().
            take(self.len()).map(|x| x.unwrap()).collect_vec().
            orbits().into_iter();
        let unknown_partitions = self.unknown_nodes().
            map(|i| Partition::from_iter(vec![i]));
        let partitions = known_partitions.chain(unknown_partitions);
        let mut a = self.graph.unlabelled_automorphisms(Some(partitions));
        a.group.adjust_len(self.len());
        a
    }
}

impl<N: Symbol, E: Symbol> Partial for PartialGraph<N, E> {
    type Complete = Graph<N, E>;
    type Diff = (N, Vec<Option<E>>);

    fn pop(&mut self) -> Self::Diff {
        let new_unknown_node = self.last();
        let mut edges = self.unknown_nodes().
            map(|i| self.graph.remove_edge((i, new_unknown_node))).collect_vec();
        edges.extend(self.unknown_nodes().
            map(|j| self.graph.remove_edge((new_unknown_node, j))));
        self.num_unknown_nodes += 1;
        let x = &mut self.graph.nodes[new_unknown_node];
        let node = x.0.clone().unwrap();
        x.0 = None;
        (node, edges)
    }

    fn push(&mut self, (node, edges): Self::Diff) {
        let new_known_node = self.len();
        let (n, _) = &mut self.graph.nodes[new_known_node];
        assert!(n.is_none());
        *n = Some(node);

        self.num_unknown_nodes -= 1;
        assert_eq!(2 * self.num_unknown_nodes, edges.len());
        for (i, edge) in self.unknown_nodes().zip_eq(edges.iter().take(self.num_unknown_nodes).cloned()) {
            if edge.is_some() {
                let prev = self.graph.insert_edge((i, new_known_node), edge.unwrap());
                assert!(prev.is_none());
            }
        }
        for (j, edge) in self.unknown_nodes().zip_eq(edges.iter().skip(self.num_unknown_nodes).cloned()) {
            if edge.is_some() {
                let prev = self.graph.insert_edge((new_known_node, j), edge.unwrap());
                assert!(prev.is_none());
            }
        }
    }

    fn empty(len: usize) -> Self {
        PartialGraph { graph: Graph::empty(repeat(None).take(len)), num_unknown_nodes: len }
    }
    fn from_complete(graph: Self::Complete) -> Self {
        let graph = Graph::from_neighbors(graph.nodes.into_iter().
            map(|(n, ne)| (Some(n), ne)));
        Self { graph, num_unknown_nodes: 0 }
    }
    fn into_complete(self) -> Self::Complete {
        assert_eq!(0, self.num_unknown_nodes);
        Graph::from_neighbors(self.graph.nodes.into_iter().map(|(n, ne)| (n.unwrap(), ne)))
    }
}


impl<N: Symbol, E: Symbol> PartialGraph<N, E> {
    pub fn unknown_nodes(&self) -> Range<usize> {
        self.len()..self.graph.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_automorphism() {
        assert_eq!(Graph::plain(0, []).automorphisms().group.generators, vec![]);
        assert_eq!(Graph::plain(1, []).automorphisms().group.generators, vec![]);
    }

    #[test]
    fn tiny_directed_automorphism() {
        let g1 = Graph::plain(3, [(0, 1)]);
        let g2 = Graph::plain(3, [(2, 1)]);
        assert!(g1.is_isomorphic(&g2));
    }
}
