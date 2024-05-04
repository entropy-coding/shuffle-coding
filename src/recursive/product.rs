//! Recursive shuffle coding on the Cartesian product of two permutable classes.
//! This implementation is uncached (no adaptive entropy coding), leading to a quadratic runtime.
//! Mostly here to demonstrate that a generic product impl is feasible in polynomial-time.
use std::collections::HashMap;

use crate::codec::{Codec, Message, MutCategorical, OrdSymbol};
use crate::permutable::Len;
use crate::recursive::{OrbitCodec, Prefix, PrefixFn, UncachedPrefixFn};

impl<P: Prefix, Q: Prefix> Prefix for (P, Q) {
    type Full = (P::Full, Q::Full);
    type Slice = (P::Slice, Q::Slice);

    fn pop_slice(&mut self) -> Self::Slice {
        (self.0.pop_slice(), self.1.pop_slice())
    }

    fn push_slice(&mut self, slice: &Self::Slice) {
        self.0.push_slice(&slice.0);
        self.1.push_slice(&slice.1);
    }

    fn from_full(full: Self::Full) -> Self {
        (P::from_full(full.0), Q::from_full(full.1))
    }

    fn into_full(self) -> Self::Full {
        (self.0.into_full(), self.1.into_full())
    }
}

pub trait OrdOrbitCodec: OrbitCodec<Symbol: OrdSymbol + Default> + Len {}
impl<O: OrbitCodec<Symbol: OrdSymbol + Default> + Len> OrdOrbitCodec for O {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductOrbitCodec<A: OrdOrbitCodec, B: OrdOrbitCodec> {
    pub a: A,
    pub b: B,
    pub codec: MutCategorical<(A::Symbol, B::Symbol)>,
}

impl<A: OrdOrbitCodec, B: OrdOrbitCodec> Codec for ProductOrbitCodec<A, B> {
    type Symbol = (A::Symbol, B::Symbol);
    fn push(&self, m: &mut Message, x: &Self::Symbol) { self.codec.push(m, x) }
    fn pop(&self, m: &mut Message) -> Self::Symbol { self.codec.pop(m) }
    fn bits(&self, x: &Self::Symbol) -> Option<f64> { self.codec.bits(x) }
}

impl<A: OrdOrbitCodec, B: OrdOrbitCodec> OrbitCodec for ProductOrbitCodec<A, B> {
    fn id(&self, index: usize) -> Self::Symbol {
        (self.a.id(index), self.b.id(index))
    }

    fn index(&self, id: &Self::Symbol) -> usize {
        for i in 0..self.a.len() {
            if self.a.id(i) == id.0 && self.b.id(i) == id.1 {
                return i;
            }
        }
        panic!("Orbit id not found: {:?}", id);
    }
}

impl<A: OrdOrbitCodec, B: OrdOrbitCodec> ProductOrbitCodec<A, B> {
    fn new(a: A, b: B) -> Self {
        assert_eq!(a.len(), b.len());
        let mut counts = HashMap::new();
        for index in 0..a.len() {
            *counts.entry((a.id(index), b.id(index))).or_insert(0) += 1;
        }
        Self { a, b, codec: MutCategorical::new(counts) }
    }
}

pub trait OrdOrbitCodecs: PrefixFn<Output: OrdOrbitCodec> {}
impl<O: PrefixFn<Output: OrdOrbitCodec>> OrdOrbitCodecs for O {}

impl<A: OrdOrbitCodecs, B: OrdOrbitCodecs> UncachedPrefixFn for (A, B) {
    type Prefix = (A::Prefix, B::Prefix);
    type Output = ProductOrbitCodec<A::Output, B::Output>;

    fn apply(&self, x: &Self::Prefix) -> Self::Output {
        ProductOrbitCodec::new(self.0.apply(&x.0), self.1.apply(&x.1))
    }
}