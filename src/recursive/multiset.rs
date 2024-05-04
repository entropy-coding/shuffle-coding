//! Recursive shuffle coding for multisets.
use std::marker::PhantomData;

use itertools::{Itertools, repeat_n};

use crate::codec::{Codec, ConstantCodec, EnumCodec, IID, OrdSymbol, Symbol};
use crate::permutable::{Len, Permutable, PermutableCodec, Permutation};
use crate::plain::{Orbits, PermutationGroup, plain_shuffle_codec, PlainPermutable, PlainShuffleCodec};
use crate::recursive::{Prefix, PrefixFn, ShuffleCodec, SliceCodecs, UncachedPrefixFn};
use crate::recursive::joint::{JointPrefix, JointSliceCodecs};
use crate::recursive::prefix_orbit::{hash, PrefixOrbitCodec};

impl<T: Symbol> Prefix for Vec<T> {
    type Full = Vec<T>;
    type Slice = T;

    fn pop_slice(&mut self) -> Self::Slice { self.pop().unwrap() }
    fn push_slice(&mut self, slice: &Self::Slice) { self.push(slice.clone()) }
    fn from_full(full: Vec<T>) -> Self { full }
    fn into_full(self) -> Vec<T> { self }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IIDVecSliceCodecs<S: Codec + Symbol> {
    len: usize,
    slice: S,
}

impl<S: Codec + Symbol> UncachedPrefixFn for IIDVecSliceCodecs<S> {
    type Prefix = Vec<<S as Codec>::Symbol>;
    type Output = S;

    fn apply(&self, _: &Self::Prefix) -> Self::Output { self.slice.clone() }
}

impl<S: Codec + Symbol> Len for IIDVecSliceCodecs<S> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<S: Codec + Symbol> SliceCodecs for IIDVecSliceCodecs<S> {
    fn empty_prefix(&self) -> impl Codec<Symbol=Self::Prefix> {
        ConstantCodec(vec![])
    }
}

impl<S: Codec + Symbol> IIDVecSliceCodecs<S> {
    #[allow(unused)]
    pub fn new(len: usize, slice: S) -> Self {
        IIDVecSliceCodecs { len, slice }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VecOrbitCodecs<S: OrdSymbol> {
    len: usize,
    phantom: PhantomData<S>,
}

impl<S: OrdSymbol> PrefixFn for VecOrbitCodecs<S> {
    type Prefix = Vec<S>;
    type Output = PrefixOrbitCodec;

    fn apply(&self, x: &Self::Prefix) -> Self::Output {
        let mut ids = x.iter().map(hash).collect_vec();
        let num_ignored_ids = self.len - x.len();
        ids.extend(repeat_n(0, num_ignored_ids));
        PrefixOrbitCodec::new(ids, num_ignored_ids)
    }

    fn update_after_pop_slice(&self, orbits: &mut Self::Output, _x: &Self::Prefix, _slice: &S) {
        orbits.pop_id();
    }

    fn update_after_push_slice(&self, orbits: &mut Self::Output, x: &Self::Prefix, slice: &S) {
        orbits.update_id(x.last_index(), hash(slice));
        orbits.push_id();
    }

    fn swap(&self, orbits: &mut Self::Output, i: usize, j: usize) {
        orbits.swap(i, j)
    }
}

impl<S: OrdSymbol> VecOrbitCodecs<S> {
    #[allow(unused)]
    pub fn new(len: usize) -> Self {
        Self { len, phantom: PhantomData }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointVecOrbitCodecs<S: OrdSymbol> {
    len: usize,
    phantom: PhantomData<S>,
}

impl<S: OrdSymbol> JointVecOrbitCodecs<S> {
    pub fn new(len: usize) -> Self {
        Self { len, phantom: PhantomData }
    }
}

impl<S: OrdSymbol> PrefixFn for JointVecOrbitCodecs<S> {
    type Prefix = JointPrefix<Vec<S>>;
    type Output = PrefixOrbitCodec;

    fn apply(&self, x: &Self::Prefix) -> Self::Output {
        PrefixOrbitCodec::new(x.full.iter().map(hash).collect_vec(), x.full.len() - x.len)
    }

    fn update_after_pop_slice(&self, orbits: &mut Self::Output, _x: &Self::Prefix, _slice: &()) {
        orbits.pop_id();
    }

    fn update_after_push_slice(&self, orbits: &mut Self::Output, _x: &Self::Prefix, _slice: &()) {
        orbits.push_id()
    }

    fn swap(&self, orbits: &mut Self::Output, i: usize, j: usize) {
        orbits.swap(i, j)
    }
}

pub type MultisetShuffleCodec<S, C> = EnumCodec<PlainShuffleCodec<C>, ShuffleCodec<JointSliceCodecs<C>, JointVecOrbitCodecs<S>>>;

pub fn multiset_shuffle_codec<S: OrdSymbol, C: PermutableCodec<Symbol=Vec<S>>>(ordered: C) -> MultisetShuffleCodec<S, C> {
    let len = ordered.len();
    // Plain shuffle coding has lower fixed overhead, but joint shuffle coding scales better:
    if len <= 11 {
        EnumCodec::A(plain_shuffle_codec(ordered))
    } else {
        EnumCodec::B(ShuffleCodec::new(JointSliceCodecs::new(ordered), JointVecOrbitCodecs::new(len)))
    }
}

#[allow(unused)]
pub fn iid_multiset_shuffle_codec<S: Codec<Symbol: OrdSymbol> + Symbol>(codec: &IID<S>) -> ShuffleCodec<IIDVecSliceCodecs<S>, VecOrbitCodecs<S::Symbol>> {
    ShuffleCodec::new(IIDVecSliceCodecs::new(codec.len, codec.item.clone()), VecOrbitCodecs::new(codec.len))
}

/// Needed so that Unordered<JointPrefix<Vec<S>>> implements Eq and can be used as Symbol in PrefixShuffleCodec.
impl<S: Symbol> PlainPermutable for JointPrefix<Vec<S>> {
    fn automorphism_group(&self) -> PermutationGroup { unimplemented!() }
    fn canon(&self) -> Permutation { unimplemented!() }
    fn orbits(&self) -> Orbits { unimplemented!() }
}