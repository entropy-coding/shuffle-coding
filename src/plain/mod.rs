//! Plain shuffle coding.
use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;
use std::ops::Deref;

use crate::codec::{Codec, Message, Uniform, UniformCodec};
#[cfg(test)]
use crate::codec::assert_bits_eq;
use crate::permutable::{Orbit, Permutable, Permutation, PermutationUniform, Unordered};

pub mod coset_recursive;

pub mod labelled_graph;

/// Ordered object with a canonical ordering, and known automorphism group.
/// This is required for plain shuffle coding.
pub trait PlainPermutable: Permutable {
    /// Returns the automorphism group of the object.
    fn automorphism_group(&self) -> PermutationGroup;
    /// Returns the canonical ordering of the given object.
    fn canon(&self) -> Permutation;

    fn canonized(&self) -> (Self, Permutation) {
        let c = self.canon();
        (&c * self, c)
    }

    fn is_isomorphic(&self, other: &Self) -> bool {
        self.canonized().0 == other.canonized().0
    }

    /// Return the orbits of the object's automorphism group.
    fn orbits(&self) -> Orbits;

    /// Fused method for getting automorphism group, canonical permutation and orbits all at once,
    /// allowing optimized implementations.
    fn automorphisms(&self) -> Automorphisms {
        let group = self.automorphism_group();
        Automorphisms {
            group: group.clone(),
            canon: self.canon(),
            decanon: self.canon().inverse(),
            orbits: self.orbits(),
            bits: RCosetUniform::new(group).uni_bits(),
        }
    }

    #[cfg(test)]
    /// Any implementation should pass this test for any seed.
    fn test(&self, seed: usize) {
        self.test_left_group_action_axioms(seed);

        let p = &PermutationUniform { len: self.len() }.sample(seed);

        // Canonical labelling axiom, expressed in 5 equivalent ways:
        let y = p * self;
        let y_canon = y.canon();
        assert_eq!(y.permuted(&y_canon), self.permuted(&self.canon()));
        assert_eq!(&y_canon * &y, &self.canon() * self);
        assert_eq!(y.canonized().0, self.canonized().0);
        assert!(y.is_isomorphic(self));
        assert_eq!(Unordered(y.clone()), Unordered(self.clone()));

        // Fused method allowing optimized implementations:
        let y_aut = y.automorphisms();
        assert_eq!(y_aut.canon, y_canon);
        assert_eq!(y_aut.decanon, y_canon.inverse());

        let x_aut = self.automorphisms();

        // misc
        assert_eq!(y.orbits().len(), self.orbits().len());
        assert_eq!(y_aut.orbits.len(), x_aut.orbits.len());

        let y_aut_bits = AutomorphismUniform::from_group(y.automorphism_group()).uni_bits();
        let group = self.automorphism_group();
        let x_aut_bits = AutomorphismUniform::from_group(group).uni_bits();
        assert_bits_eq(y_aut_bits, x_aut_bits);
        assert_bits_eq(y_aut.bits, x_aut.bits);
        assert_bits_eq(x_aut.bits, x_aut_bits);

        // TODO test automorphism_group
    }
}

impl<P: PlainPermutable> PartialEq<Self> for Unordered<P> {
    fn eq(&self, other: &Self) -> bool {
        self.to_ordered().is_isomorphic(other.to_ordered())
    }
}

impl<P: PlainPermutable> Eq for Unordered<P> {}

/// Plain shuffle coding.
#[derive(Clone, Debug, Eq)]
pub struct PlainShuffleCodec<
    C: Codec<Symbol: PlainPermutable>,
    RCosetC: UniformCodec<Symbol=Permutation> = RCosetUniform,
    RCosetCFromP: Fn(&C::Symbol) -> RCosetC + Clone = fn(&<C as Codec>::Symbol) -> RCosetC> {
    pub ordered: C,
    pub rcoset_for: RCosetCFromP,
}

impl<
    C: Codec<Symbol: PlainPermutable>,
    RCosetCFromP: Fn(&C::Symbol) -> RCosetC + Clone,
    RCosetC: UniformCodec<Symbol=Permutation>>
Codec for PlainShuffleCodec<C, RCosetC, RCosetCFromP> {
    type Symbol = Unordered<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let x = x.canonized();
        let coset_min = (self.rcoset_for)(&x).pop(m);
        let x = &coset_min * &x;
        self.ordered.push(m, &x)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let (x, canon) = self.ordered.pop(m).canonized();
        let coset_min = canon.inverse();
        (self.rcoset_for)(&x).push(m, &coset_min);
        Unordered(x)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.ordered.bits(x.to_ordered()).map(|bits| bits - (self.rcoset_for)(x.to_ordered()).uni_bits())
    }
}

pub fn plain_shuffle_codec<C: Codec<Symbol: PlainPermutable>>(codec: C) -> PlainShuffleCodec<C> {
    PlainShuffleCodec { ordered: codec, rcoset_for: |x| RCosetUniform::new(x.automorphism_group()) }
}

/// Only for use of codec as symbol for parametrized models:
impl<
    C: Codec<Symbol: PlainPermutable>,
    RCosetCFromP: Fn(&C::Symbol) -> RCosetC + Clone,
    RCosetC: UniformCodec<Symbol=Permutation>>
PartialEq<Self> for PlainShuffleCodec<C, RCosetC, RCosetCFromP> {
    fn eq(&self, _: &Self) -> bool { true }
}

impl<P: PlainPermutable> PartialOrd<Self> for Unordered<P> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: PlainPermutable> Ord for Unordered<P> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_ordered().canon().cmp(&other.to_ordered().canon())
    }
}

impl<P: PlainPermutable> Unordered<P> {
    pub fn canonized(&self) -> P { self.to_ordered().canonized().0 }
}

pub type Orbits = Vec<Orbit>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrbitElementUniform {
    pub sorted_elements: Vec<usize>,
}

impl OrbitElementUniform {
    pub fn new(orbit: Orbit) -> Self {
        Self { sorted_elements: orbit.into_iter().collect() }
    }

    fn len(&self) -> usize { self.sorted_elements.len() }

    pub fn min(&self) -> usize { self.sorted_elements[0] }

    fn inner_codec(&self) -> Uniform {
        let size = self.len();
        Uniform::new(size)
    }
}

impl Codec for OrbitElementUniform {
    type Symbol = usize;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let index = self.sorted_elements.iter().position(|e| *e == *x).unwrap();
        self.inner_codec().push(m, &index);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let index = self.inner_codec().pop(m);
        self.sorted_elements[index]
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for OrbitElementUniform {
    fn uni_bits(&self) -> f64 { self.inner_codec().uni_bits() }
}

#[derive(Clone)]
pub struct AutomorphismUniform {
    chain: StabilizerChain,
}

impl Codec for AutomorphismUniform {
    type Symbol = Automorphism;

    fn push(&self, m: &mut Message, x: &Automorphism) {
        assert_eq!(x.min_elements.len(), self.chain.len());
        for ((orbit, _), min_element) in self.chain.iter().zip(&x.min_elements).rev() {
            OrbitElementUniform::new(orbit.clone()).push(m, &min_element);
        }
    }

    fn pop(&self, m: &mut Message) -> Automorphism {
        let (min_elements, element_to_min) = self.chain.iter().map(|(orbit, from)| {
            let min_element = OrbitElementUniform::new(orbit.clone()).pop(m);
            (min_element, from[&min_element].inverse())
        }).unzip();

        Automorphism { min_elements, base_to_min: element_to_min }
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for AutomorphismUniform {
    fn uni_bits(&self) -> f64 {
        self.chain.iter().map(|(orbit, _)| (orbit.len() as f64).log2()).sum()
    }
}

impl AutomorphismUniform {
    pub fn new(chain: StabilizerChain) -> Self { Self { chain } }

    #[allow(unused)]
    pub fn from_group(group: PermutationGroup) -> Self {
        Self::new(group.lex_stabilizer_chain())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RCosetUniform {
    chain: StabilizerChain,
}

impl Codec for RCosetUniform {
    type Symbol = Permutation;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let (coset_min, _) = self.chain.coset_min_and_restore_aut(x);
        let restore_aut = self.automorphism_codec().pop(m);
        let perm = restore_aut.apply_to(coset_min);
        self.permutation_codec().push(m, &perm);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let perm = self.permutation_codec().pop(m);
        let (coset_min, restore_aut) = self.chain.coset_min_and_restore_aut(&perm);
        self.automorphism_codec().push(m, &restore_aut);
        coset_min
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for RCosetUniform {
    fn uni_bits(&self) -> f64 {
        self.permutation_codec().uni_bits() - self.automorphism_codec().uni_bits()
    }
}

impl RCosetUniform {
    pub fn new(group: PermutationGroup) -> Self {
        Self { chain: group.lex_stabilizer_chain() }
    }

    fn automorphism_codec(&self) -> AutomorphismUniform {
        AutomorphismUniform::new(self.chain.clone())
    }

    fn permutation_codec(&self) -> PermutationUniform {
        PermutationUniform { len: self.chain.len() }
    }
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StabilizerChain(Vec<(Orbit, HashMap<usize, Permutation>)>);

impl Deref for StabilizerChain {
    type Target = Vec<(Orbit, HashMap<usize, Permutation>)>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

/// Automorphism that when applied to the min coset representative containing a permutation l
/// will recover l.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Automorphism {
    pub min_elements: Vec<usize>,
    pub base_to_min: Vec<Permutation>,
}

impl Automorphism {
    /// (a[n] * a[n-1] * ... a[1] * a[0])
    pub fn total(self) -> Permutation {
        let mut total = Permutation::identity(self.min_elements.len());
        for element_to_min in self.base_to_min.into_iter().rev() {
            total = element_to_min * total;
        }
        total
    }

    /// Restore permutation from lexicographical minimum of its coset.
    pub fn apply_to(self, coset_min: Permutation) -> Permutation {
        coset_min * self.total().inverse()
    }
}

impl StabilizerChain {
    /// Returns a pair of:
    /// - The lexicographical minimum of the coset containing the given labels.
    /// Can be used as a canonical representative of that coset.
    /// - The automorphism to be applied to this representative to restore the original labels.
    ///
    /// In short: (h, a) where h = min{labels * AUT}, a = h * labels^-1
    pub fn coset_min_and_restore_aut(&self, labels: &Permutation) -> (Permutation, Automorphism) {
        let mut min_elements = vec![];
        let mut base_to_min = vec![];
        let mut labels = labels.clone();

        for (base, (orbit, from)) in self.iter().enumerate() {
            let (min_label, min_element) = orbit.iter().
                map(|&e| (&labels * e, e)).
                min_by(|(l, _), (l_, _)| l.cmp(l_)).
                unwrap();
            let base_to_min_automorphism = from[&min_element].inverse();
            min_elements.push(min_element);
            base_to_min.push(base_to_min_automorphism.clone());
            // Unapply automorphism to get min coset representative, moving min -> base:
            let new_labels = labels * base_to_min_automorphism;
            assert_eq!(&new_labels * base, min_label);
            labels = new_labels;
        }
        (labels, Automorphism { min_elements, base_to_min })
    }
}

pub struct OrbitStabilizerInfo {
    pub orbit: Orbit,
    pub stabilizer: PermutationGroup,
    /// Permutation from any orbit element to the original element.
    pub from: HashMap<usize, Permutation>,
}

#[derive(Clone, Debug)]
pub struct Automorphisms {
    pub group: PermutationGroup,
    pub canon: Permutation,
    pub decanon: Permutation,
    pub orbits: Orbits,
    pub bits: f64,
}

impl PartialEq for Automorphisms {
    fn eq(&self, other: &Self) -> bool {
        self.group == other.group && self.canon == other.canon
    }
}

impl Eq for Automorphisms {}

impl Automorphisms {
    pub fn orbit(&self, index: usize) -> Orbit {
        self.orbits.iter().find(|o| o.contains(&index)).unwrap().clone()
    }
}

impl Permutable for Automorphisms {
    fn len(&self) -> usize {
        self.group.len
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.group.swap(i, j);
        self.canon.swap(i, j);
        self.decanon.swap(*self.canon.indices.get(&i).unwrap(), *self.canon.indices.get(&j).unwrap());
        for orbit in &mut self.orbits {
            let has_i = orbit.contains(&i);
            let has_j = orbit.contains(&j);
            if has_i && !has_j {
                orbit.remove(&i);
                orbit.insert(j);
            } else if has_j && !has_i {
                orbit.remove(&j);
                orbit.insert(i);
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PermutationGroup {
    pub len: usize,
    pub generators: Vec<Permutation>,
}

impl PermutationGroup {
    pub fn adjust_len(&mut self, len: usize) {
        for g in self.generators.iter_mut() {
            g.set_len(len);
        }
        self.len = len;
    }

    pub fn new(len: usize, generators: Vec<Permutation>) -> Self {
        for g in &generators {
            assert_eq!(len, g.len);
        }
        Self { len, generators }
    }

    /// Return an element's orbit and stabilizer group.
    /// See K. H. Rosen: Computational Group Theory, p. 79.
    pub fn orbit_stabilizer(&self, element: usize) -> OrbitStabilizerInfo {
        let mut orbit = vec![element];
        let mut from = HashMap::from([(element, Permutation::identity(self.len))]);
        let mut generators = BTreeSet::new();
        let mut i = 0;
        while let Some(&e) = orbit.get(i) {
            let from_e = from[&e].clone();
            for g in self.generators.iter() {
                let new = g * e;
                let from_new = from_e.clone() * g.inverse();
                assert_eq!(&from_new * new, element);
                if let Some(from_new_) = from.get(&new) {
                    let stab_g = from_new * from_new_.inverse();
                    if !stab_g.is_identity() {
                        assert_eq!(&stab_g * element, element);
                        generators.insert(stab_g);
                    }
                } else {
                    from.insert(new, from_new);
                    orbit.push(new);
                }
            }

            i += 1;
        }

        let stabilizer = PermutationGroup::new(self.len, generators.into_iter().collect());
        OrbitStabilizerInfo { orbit: Orbit::from_iter(orbit), stabilizer, from }
    }

    /// Stabilizer chain for base 0..self.len.
    /// OPTIMIZE: Use Schreier-Sims algorithm to compute stabilizer chain more quickly.
    pub fn lex_stabilizer_chain(&self) -> StabilizerChain {
        // OPTIMIZE: avoid this clone
        let mut group = self.clone();
        let mut chain = vec![];
        for element in 0..self.len {
            let OrbitStabilizerInfo { orbit, stabilizer, from } = group.orbit_stabilizer(element);
            chain.push((orbit, from));
            group = stabilizer;
        }
        StabilizerChain(chain)
    }

    /// Conjugated group with the set of elements relabeled as given by the permutation.
    pub fn permuted(self, p: &Permutation) -> PermutationGroup {
        let generators = self.generators.into_iter().map(|g| p.clone() * g * p.inverse().clone()).collect();
        PermutationGroup::new(self.len, generators)
    }
}

impl Permutable for PermutationGroup {
    fn len(&self) -> usize { self.len }

    fn swap(&mut self, i: usize, j: usize) {
        for g in self.generators.iter_mut() {
            g.swap(i, j);
        }
    }
}

#[cfg(test)]
pub mod test {
    use itertools::Itertools;

    use crate::benchmark::TestConfig;
    use crate::codec::Codec;
    use crate::codec::tests::test_and_print_vec;
    use crate::graph::Graph;
    use crate::graph_codec::tests::small_digraphs;
    use crate::permutable::{Permutable, PermutableCodec, Permutation};
    use crate::plain::coset_recursive::coset_recursive_shuffle_codec;

    use super::*;

    pub fn test_plain_shuffle_codecs<C: PermutableCodec<Symbol: PlainPermutable>>(
        codecs: &Vec<C>,
        unordered: &Vec<Unordered<C::Symbol>>,
        config: &TestConfig) {
        if let Some(seed) = config.axioms_seed {
            for x in unordered {
                x.to_ordered().test(seed);
            }
        }
        if config.plain {
            test_and_print_vec(codecs.iter().cloned().map(plain_shuffle_codec), unordered, &config.initial_message());
        }
        if config.plain_coset_recursive {
            test_and_print_vec(codecs.iter().cloned().map(coset_recursive_shuffle_codec), unordered, &config.initial_message());
        }
    }

    #[test]
    fn permutation_codec() {
        PermutationUniform { len: 5 }.test(&Permutation::from(vec![0, 2, 1, 3, 4]), &Message::random(0));
        PermutationUniform { len: 9 }.test_on_samples(100);
    }

    #[test]
    fn automorphism_codec() {
        for graph in small_digraphs() {
            test_automorphism_codec(graph);
        }
    }

    fn test_automorphism_codec(graph: Graph) {
        let group = graph.automorphism_group();
        let chain = group.lex_stabilizer_chain();
        let labels = PermutationUniform { len: graph.len() }.sample(0);
        let (canonized_labels, restore_aut) = chain.coset_min_and_restore_aut(&labels);
        let labels_ = restore_aut.clone().apply_to(canonized_labels.clone());
        assert_eq!(labels_, labels.clone());

        let mut isomorphic_labels = labels.clone();
        for g in group.generators.clone() {
            isomorphic_labels = isomorphic_labels * g;
            let (canonized_labels_, _) = chain.coset_min_and_restore_aut(&isomorphic_labels);

            assert_eq!(&canonized_labels, &canonized_labels_);
        }

        let aut_codec = AutomorphismUniform::new(chain);
        aut_codec.test(&restore_aut, &Message::random(0));
        aut_codec.test_on_samples(1000);
    }

    #[test]
    fn rcoset_codec() {
        for x in small_digraphs() {
            RCosetUniform::new(x.automorphism_group()).test_on_samples(100);
        }
    }

    #[test]
    fn orbit_stabilizer_trivial() {
        let trivial = PermutationGroup::new(3, vec![]);
        let r = trivial.orbit_stabilizer(2);
        assert_eq!(r.orbit.into_iter().collect_vec(), vec![2]);
        assert_eq!(r.stabilizer.generators, vec![]);
    }

    #[test]
    fn orbit_stabilizer() {
        let generators = vec![Permutation::from(vec![0, 1, 3, 2])];
        let g = PermutationGroup::new(4, generators);
        let r = g.orbit_stabilizer(2);
        assert_eq!(r.orbit, Orbit::from([2, 3]));
        assert_eq!(r.stabilizer.generators, vec![]);
    }

    #[test]
    fn orbit_stabilizer2() {
        let generators = vec![Permutation::from(vec![0, 1, 3, 2]), Permutation::from(vec![1, 0, 2, 3])];
        let g = PermutationGroup::new(4, generators);
        let r = g.orbit_stabilizer(2);
        assert_eq!(r.orbit, Orbit::from([2, 3]));
        assert_eq!(r.stabilizer.generators, vec![Permutation::from(vec![1, 0, 2, 3])]);
    }

    #[test]
    fn permuted_group() {
        let g = &PermutationGroup::new(4, vec![Permutation::from(vec![0, 1, 3, 2])]);
        let p = &Permutation::from(vec![3, 0, 1, 2]);
        let g_p = g.clone().permuted(p);

        for e in 0..3 {
            let orbit = g.orbit_stabilizer(e).orbit.iter().map(|x| p * *x).collect_vec();
            let orbit_ = g_p.orbit_stabilizer(p * e).orbit.into_iter().collect_vec();
            assert_eq!(orbit, orbit_)
        }
    }
}