//! Ordered objects / permutable classes.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::ops::Mul;
use std::vec;

use float_extras::f64::lgamma;
use itertools::Itertools;

use crate::codec::{Codec, Message, Symbol, Uniform, UniformCodec};

/// An ordered object. Each type implementing this represents a permutable class.
pub trait Permutable: Symbol {
    /// Length of the ordered object. The order of the corresponding permutable class.
    fn len(&self) -> usize;

    /// Swap two indices of the ordered object.
    fn swap(&mut self, i: usize, j: usize);

    /// Permutes the ordered object by a given permutation. 
    /// By default, this is implemented based on swap.
    fn permuted(&self, x: &Permutation) -> Self {
        self._permuted_from_swap(x)
    }

    fn _permuted_from_swap(&self, x: &Permutation) -> Self {
        assert_eq!(self.len(), x.len);
        let mut p = Permutation::identity(self.len());
        let mut p_inv = Permutation::identity(self.len());
        let mut out = self.clone();
        for i in (0..self.len()).rev() {
            let xi = x * i;
            let j = &p_inv * xi;
            let pi = &p * i;
            p_inv.swap(pi, xi);
            out.swap(pi, xi);
            p.swap(i, j);
        }

        out
    }

    /// Shuffles the ordered object by a random permutation uniformly sampled with the given seed.
    fn shuffled(&self, seed: usize) -> Self {
        self.permuted(&PermutationUniform { len: self.len() }.sample(seed))
    }

    /// Any implementation should pass this test for any ordered object and seed.
    fn test_left_group_action_axioms(&self, seed: usize) {
        let (p0, p1) = &PermutationUniform { len: self.len() }.samples(2, seed).into_iter().collect_tuple().unwrap();
        assert_eq!((p0 * self).len(), self.len());
        assert_eq!(&(&Permutation::identity(self.len()) * self), self);
        assert_eq!(p1 * &(p0 * self), &(p1.clone() * p0.clone()) * self);
    }
}

/// The unordered object corresponding to a given ordered object. 
/// The main difference is that equality is based on isomorphism, implemented in `plain/mod.rs`.
#[derive(Clone, Debug, Hash)]
pub struct Unordered<P: Permutable>(pub P);

impl<P: Permutable> Unordered<P> {
    pub fn into_ordered(self) -> P { self.0 }
    pub fn to_ordered(&self) -> &P { &self.0 }
    pub fn len(&self) -> usize { self.to_ordered().len() }
}

/// Permutation a given length, sparsely represented.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Permutation {
    pub len: usize,
    pub indices: BTreeMap<usize, usize>,
}

impl Permutation {
    pub fn identity(len: usize) -> Self { Self { len, indices: BTreeMap::new() } }

    pub fn inverse(&self) -> Self {
        Self { len: self.len, indices: self.indices.iter().map(|(&x, &y)| (y, x)).collect() }
    }

    pub fn from(items: Vec<usize>) -> Self {
        let len = items.len();
        Self {
            len,
            indices: items.into_iter().enumerate().filter(|(i, x)| {
                assert!(*x < len);
                i != x
            }).map(|(i, x)| (i, x)).collect(),
        }
    }

    pub fn to_iter(&self, len: usize) -> impl Iterator<Item=usize> + '_ {
        (0..len).map(move |i| self * i).into_iter()
    }

    pub fn is_identity(&self) -> bool { self.indices.is_empty() }

    fn normalize(&mut self) {
        self.indices.retain(|i, x| i != x);
    }

    pub fn set_len(&mut self, len: usize) {
        assert!(len >= self.len || self.indices.iter().all(|(&x, &y)| x < len && y < len));
        self.len = len;
    }

    pub fn create_swap(len: usize, i: usize, j: usize) -> Self {
        Self { len, indices: if i == j { BTreeMap::new() } else { BTreeMap::from_iter(vec![(i, j), (j, i)]) } }
    }
}

/// Permutations of a given length are ordered objects forming permutable class. 
impl Permutable for Permutation {
    fn len(&self) -> usize {
        self.len
    }

    fn swap(&mut self, i: usize, j: usize) {
        if i == j { return; }
        let p = self as &Self;
        let pi = p * i;
        let pj = p * j;
        if j == pi { self.indices.remove(&j); } else { self.indices.insert(j, pi); }
        if i == pj { self.indices.remove(&i); } else { self.indices.insert(i, pj); }
    }
}

impl Mul<Permutation> for Permutation {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self {
        assert_eq!(self.len, rhs.len);

        for x in rhs.indices.values_mut() {
            *x = self.indices.remove(x).unwrap_or(*x);
        }

        for (&i, &x) in self.indices.iter() {
            rhs.indices.entry(i).or_insert(x);
        }

        rhs.normalize();
        rhs
    }
}

impl Mul<usize> for &Permutation {
    type Output = usize;

    fn mul(self, rhs: usize) -> usize {
        *self.indices.get(&rhs).unwrap_or(&rhs)
    }
}

impl<P: Permutable> Mul<&P> for &Permutation {
    type Output = P;

    fn mul(self, rhs: &P) -> P { P::permuted(rhs, self) }
}

impl<P: Permutable, Q: Permutable> Permutable for (P, Q) {
    fn len(&self) -> usize {
        let s = self.0.len();
        assert_eq!(s, self.1.len());
        s
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
        self.1.swap(i, j);
    }
}

/// Pop is the modern version of the Fisher-Yates shuffle, and push its inverse. Both are O(len).
#[derive(Clone)]
pub struct PermutationUniform {
    pub len: usize,
}

impl Codec for PermutationUniform {
    type Symbol = Permutation;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let mut p = Permutation::identity(self.len);
        let mut p_inv = Permutation::identity(self.len);
        let mut js = vec![];
        for i in (0..self.len).rev() {
            let xi = x * i;
            let j = &p_inv * xi;
            p_inv.swap(&p * i, xi);
            p.swap(i, j);
            js.push(j);
        }

        for (i, j) in js.iter().rev().enumerate() {
            let size = i + 1;
            Uniform::new(size).push(m, j);
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut p = Permutation::identity(self.len);

        for i in (0..self.len).rev() {
            let size = i + 1;
            let j = Uniform::new(size).pop(m);
            p.swap(i, j);
        }

        p
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for PermutationUniform {
    fn uni_bits(&self) -> f64 {
        lgamma((self.len + 1) as f64) / 2f64.ln()
    }
}

pub type Orbit = BTreeSet<usize>;

/// Len determines the length of a permutation in a permutable object.
/// Needs to be known upfront in some places, for example for decoding with recursive shuffle coding.
pub trait Len {
    fn len(&self) -> usize;
}

/// Codec for objects from a single permutable class of known length.
pub trait PermutableCodec: Codec<Symbol: Permutable> + Len {}
impl<C: Codec<Symbol: Permutable> + Len> PermutableCodec for C {}

#[cfg(test)]
pub mod tests {
    use std::collections::BTreeMap;

    use itertools::Itertools;

    use crate::codec::Codec;

    use super::*;

    #[test]
    fn permutation_elements() {
        let p0 = Permutation { len: 5, indices: BTreeMap::from([(2, 3), (3, 4), (4, 2)]) };
        let p1 = Permutation { len: 5, indices: BTreeMap::from([(0, 1), (1, 3), (2, 0), (3, 2)]) };
        let p2 = p1 * p0;
        assert_eq!(p2.indices, BTreeMap::from([(0, 1), (1, 3), (3, 4), (4, 0)]));
    }

    #[test]
    fn permutation_axioms() {
        for i in 0..20 {
            let (p0, p1, p2) = PermutationUniform { len: i }.samples(3, i).into_iter().collect_tuple().unwrap();

            // Permutation group axioms:
            assert_eq!(Permutation::identity(i) * p0.clone(), p0);
            assert_eq!(p0.inverse() * p0.clone(), Permutation::identity(i));
            assert_eq!(p0.clone() * p0.inverse(), Permutation::identity(i));
            assert_eq!((p2.clone() * p1.clone()) * p0.clone(), p2.clone() * (p1.clone() * p0.clone()));

            // Compatibility with integer:
            assert_eq!(&Permutation::identity(i) * 2, 2);
            assert_eq!(&(p2.clone() * p1.clone()) * 2, &p2 * (&p1 * 2));
        }
    }
}