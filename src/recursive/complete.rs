use std::marker::PhantomData;

use itertools::{Itertools, repeat_n};

use crate::ans::{Codec, Distribution, EnumCodec, IID, Symbol};
use crate::permutable::{Orbit, Permutable};
use crate::plain::{Automorphisms, plain_shuffle_codec, PlainShuffleCodec};
use crate::recursive::{FromPrefix, hash, IIDVecSliceCodecs, JointPrefix, JointSliceCodecs, OrbitCodec, OrdSymbol, PlainPrefix, Prefix, PrefixOrbitCodec, ShuffleCodec};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompleteOrbitCodecs<P: PlainPrefix> {
    phantom: PhantomData<P>,
}

impl<P: PlainPrefix> CompleteOrbitCodecs<P> {
    pub fn new() -> Self { Self { phantom: PhantomData } }
}

impl<P: PlainPrefix> FromPrefix for CompleteOrbitCodecs<P> {
    type Prefix = P;
    type Output = CompleteOrbitCodec;

    fn apply(&self, x: &P) -> Self::Output {
        CompleteOrbitCodec::new(x.automorphisms())
    }

    fn swap(&self, _: &mut Self::Output, _: usize, _: usize) {
        // Empty because result is never used.
    }
}

#[derive(Clone, Debug)]
pub struct VecOrbitCodecs<S: OrdSymbol> {
    len: usize,
    phantom: PhantomData<S>,
}

impl<S: OrdSymbol> VecOrbitCodecs<S> {
    #[allow(unused)]
    pub fn new(len: usize) -> Self {
        Self { len, phantom: PhantomData }
    }
}

impl<S: OrdSymbol> FromPrefix for VecOrbitCodecs<S> {
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
        orbits.update_id(x.last_(), hash(slice));
        orbits.push_id();
    }

    fn swap(&self, orbits: &mut Self::Output, i: usize, j: usize) {
        orbits.swap(i, j)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointVecOrbitCodecs<S: OrdSymbol> {
    len: usize,
    phantom: PhantomData<S>,
}

impl<S: OrdSymbol> JointVecOrbitCodecs<S> {
    pub fn new(len: usize) -> Self {
        Self { phantom: PhantomData, len }
    }
}

impl<S: OrdSymbol> FromPrefix for JointVecOrbitCodecs<S> {
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

pub trait Len {
    fn len(&self) -> usize;
}

pub type MultisetShuffleCodec<S, C> = EnumCodec<PlainShuffleCodec<C>, ShuffleCodec<JointSliceCodecs<C>, JointVecOrbitCodecs<S>>>;

pub fn multiset_shuffle_codec<S: OrdSymbol, C: Codec<Symbol=Vec<S>> + Len>(ordered: C) -> MultisetShuffleCodec<S, C> {
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompleteOrbitCodec {
    automorphisms: Automorphisms,
}

impl Distribution for CompleteOrbitCodec {
    type Symbol = Orbit;

    fn norm(&self) -> usize { self.automorphisms.group.len }
    fn pmf(&self, orbit: &Orbit) -> usize { orbit.len() }

    fn cdf(&self, orbit: &Orbit, i: usize) -> usize {
        *self.canonized(orbit).iter().nth(i).unwrap()
    }

    fn icdf(&self, cf: usize) -> (Orbit, usize) {
        let some_element = &self.automorphisms.decanon * cf;
        let orbit = self.automorphisms.orbit(some_element);
        let i = self.canonized(&orbit).iter().position(|&x| x == cf).unwrap();
        (orbit, i)
    }
}

impl CompleteOrbitCodec {
    fn new(automorphisms: Automorphisms) -> Self { Self { automorphisms } }

    fn canonized(&self, orbit: &Orbit) -> Orbit {
        Orbit::from_iter(orbit.iter().map(|&x| &self.automorphisms.canon * x))
    }
}

impl OrbitCodec for CompleteOrbitCodec {
    type OrbitId = Orbit;

    fn id(&self, element: usize) -> Orbit {
        self.automorphisms.orbit(element)
    }

    fn orbit<'a>(&self, orbit: &'a Orbit) -> &'a Orbit { orbit }
}