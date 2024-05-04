use std::collections::{BTreeSet, HashMap};
use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::mem;

use crate::ans::{Codec, ConstantCodec, Distribution, Message, MutCategorical, MutDistribution, Symbol};
use crate::graph::{EdgeType, Graph};
use crate::permutable::{Orbit, Permutable, Permutation, Unordered};
use crate::plain::{Automorphisms, GraphPlainPermutable, Orbits, PermutationGroup, PlainPermutable};
use crate::recursive::complete::Len;
use crate::recursive::graph::prefix::GraphPrefix;

pub mod graph;
pub mod complete;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixShuffleCodec<S: SliceCodecs<Prefix: PlainPrefix>, O: OrbitCodecs<Prefix=S::Prefix>> {
    pub slice_codecs: S,
    pub orbit_codecs: O,
}

impl<S: SliceCodecs<Prefix: PlainPrefix>, O: OrbitCodecs<Prefix=S::Prefix>> Codec for PrefixShuffleCodec<S, O> {
    type Symbol = Unordered<S::Prefix>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        assert_eq!(self.slice_codecs.len(), x.len());
        let mut x = x.clone();
        if self.slice_codecs.len() != 0 {
            let orbits = &mut self.orbit_codecs.apply(&x);
            let slice_codec = &mut self.slice_codecs.apply(&x);
            for _ in 0..self.slice_codecs.len() {
                let element = *orbits.orbit(&orbits.pop(m)).first().unwrap();
                let last = x.last_();
                x.swap(element, last);
                self.orbit_codecs.swap(orbits, element, last);
                self.slice_codecs.swap(slice_codec, element, last);
                let slice = x.pop_slice();
                self.orbit_codecs.update_after_pop_slice(orbits, &x, &slice);
                self.slice_codecs.update_after_pop_slice(slice_codec, &x, &slice);
                slice_codec.push(m, &slice);
            }
        }
        self.slice_codecs.empty_prefix().push(m, &x);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut x = self.slice_codecs.empty_prefix().pop(m);
        if self.slice_codecs.len() == 0 { return Unordered(x); }

        let orbits = &mut self.orbit_codecs.apply(&x);
        let slice_codec = &mut self.slice_codecs.apply(&x);
        for _ in 0..self.slice_codecs.len() {
            let slice = slice_codec.pop(m);
            x.push_slice(&slice);
            self.orbit_codecs.update_after_push_slice(orbits, &x, &slice);
            self.slice_codecs.update_after_push_slice(slice_codec, &x, &slice);
            orbits.push(m, &orbits.id(x.last_()));
        }
        Unordered(x)
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> {
        None
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShuffleCodec<S: SliceCodecs<Prefix: PlainPrefix>, O: OrbitCodecs<Prefix=S::Prefix>> {
    pub prefix: PrefixShuffleCodec<S, O>,
}

impl<S: SliceCodecs<Prefix: PlainPrefix>, O: OrbitCodecs<Prefix=S::Prefix>> Codec for ShuffleCodec<S, O> {
    type Symbol = Unordered<<S::Prefix as Prefix>::Full>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        self.prefix.push(m, &Unordered(S::Prefix::from_full(x.clone())));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Unordered(self.prefix.pop(m).0.into_full())
    }

    fn bits(&self, Unordered(x): &Self::Symbol) -> Option<f64> {
        self.prefix.bits(&Unordered(S::Prefix::from_full(x.clone())))
    }
}

impl<S: SliceCodecs<Prefix: PlainPrefix>, O: OrbitCodecs<Prefix=S::Prefix>> ShuffleCodec<S, O> {
    #[allow(unused)]
    pub fn new(slice_codecs: S, orbit_codecs: O) -> Self {
        Self { prefix: PrefixShuffleCodec { slice_codecs, orbit_codecs } }
    }
}

pub trait OrbitCodec: Codec<Symbol=Self::OrbitId> + Debug {
    type OrbitId;

    fn id(&self, element: usize) -> Self::OrbitId;

    fn orbit<'a>(&'a self, id: &'a Self::OrbitId) -> &'a Orbit;
}

/// A function defined on prefixes of a permutable class. If the prefix is the result of a pop or push operation
/// the output can be computed based on the original prefix and the popped/pushed slice, for efficiency.
pub trait FromPrefix: Clone {
    type Prefix: Prefix;
    type Output;

    fn apply(&self, x: &Self::Prefix) -> Self::Output;

    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &Self::Prefix, _slice: &<Self::Prefix as Prefix>::Slice) {
        *image = self.apply(x)
    }

    fn update_after_push_slice(&self, image: &mut Self::Output, x: &Self::Prefix, _slice: &<Self::Prefix as Prefix>::Slice) {
        *image = self.apply(x)
    }

    fn swap(&self, image: &mut Self::Output, i: usize, j: usize);
}

///A function returning a codec for the orbits of any prefix of a permutable class.
pub trait OrbitCodecs: FromPrefix<Output=Self::OrbitCodec> {
    type OrbitId: Default;
    type OrbitCodec: OrbitCodec<OrbitId=Self::OrbitId>;
}

///A function returning a codec for the next slice of any prefix of a permutable class.
pub trait SliceCodecs: FromPrefix<Output=Self::SliceCodec> + Len {
    type SliceCodec: Codec<Symbol=<Self::Prefix as Prefix>::Slice>;
    type EmptyPrefixCodec: Codec<Symbol=Self::Prefix>;

    fn empty_prefix(&self) -> Self::EmptyPrefixCodec;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Autoregressive<S: SliceCodecs>(pub S);

impl<S: SliceCodecs> Codec for Autoregressive<S> {
    type Symbol = <S::Prefix as Prefix>::Full;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let mut prefix = Prefix::from_full(x.clone());
        let slice_codec = &mut self.0.apply(&prefix);
        for _ in 0..self.0.len() {
            let slice = prefix.pop_slice();
            self.0.update_after_pop_slice(slice_codec, &prefix, &slice);
            slice_codec.push(m, &slice);
        }
        self.0.empty_prefix().push(m, &prefix);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut prefix = self.0.empty_prefix().pop(m);
        let slice_codec = &mut self.0.apply(&prefix);
        for _ in 0..self.0.len() {
            let slice = slice_codec.pop(m);
            prefix.push_slice(&slice);
            self.0.update_after_push_slice(slice_codec, &prefix, &slice);
        }
        prefix.into_full()
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

impl<S: SliceCodecs> Len for Autoregressive<S> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<Orbit: OrbitCodec<OrbitId: Symbol + Default>, O: FromPrefix<Output=Orbit>> OrbitCodecs for O {
    type OrbitId = Orbit::OrbitId;
    type OrbitCodec = Orbit;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SomeCodec<C: Codec>(pub C);

impl<C: Codec> Codec for SomeCodec<C> {
    type Symbol = Option<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.0.push(m, x.as_ref().unwrap())
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Some(self.0.pop(m))
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.0.bits(x.as_ref().unwrap())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmptyJointPrefixCodec<C: Codec<Symbol: Permutable>> {
    pub full: C,
}

impl<C: Codec<Symbol: Permutable>> Codec for EmptyJointPrefixCodec<C> {
    type Symbol = JointPrefix<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.full.push(m, &x.full)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        JointPrefix { len: 0, full: self.full.pop(m) }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.full.bits(&x.full)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointSliceCodecs<C: Codec<Symbol: Permutable> + Len> {
    pub empty: EmptyJointPrefixCodec<C>,
}

impl<C: Codec<Symbol: Permutable> + Len> FromPrefix for JointSliceCodecs<C> {
    type Prefix = JointPrefix<C::Symbol>;
    type Output = ConstantCodec<()>;

    fn apply(&self, _: &Self::Prefix) -> Self::Output {
        ConstantCodec(())
    }

    fn swap(&self, _: &mut Self::Output, _: usize, _: usize) {}
}

impl<C: Codec<Symbol: Permutable> + Len> Len for JointSliceCodecs<C> {
    fn len(&self) -> usize {
        self.empty.full.len()
    }
}

impl<C: Codec<Symbol: Permutable> + Len> SliceCodecs for JointSliceCodecs<C> {
    type SliceCodec = Self::Output;
    type EmptyPrefixCodec = EmptyJointPrefixCodec<C>;

    fn empty_prefix(&self) -> Self::EmptyPrefixCodec {
        self.empty.clone()
    }
}

impl<C: Codec<Symbol: Permutable> + Len> JointSliceCodecs<C> {
    pub fn new(full: C) -> Self {
        Self { empty: EmptyJointPrefixCodec { full } }
    }
}

#[derive(Clone)]
pub struct IIDVecSliceCodecs<S: Codec + Symbol> {
    len: usize,
    slice: S,
}

impl<S: Codec + Symbol> FromPrefix for IIDVecSliceCodecs<S> {
    type Prefix = Vec<<S as Codec>::Symbol>;
    type Output = S;

    fn apply(&self, _: &Self::Prefix) -> Self::Output { self.slice.clone() }

    fn swap(&self, _: &mut Self::Output, _: usize, _: usize) {}
}

impl<S: Codec + Symbol> Len for IIDVecSliceCodecs<S> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<S: Codec + Symbol> SliceCodecs for IIDVecSliceCodecs<S> {
    type SliceCodec = S;
    type EmptyPrefixCodec = ConstantCodec<Vec<S::Symbol>>;

    fn empty_prefix(&self) -> Self::EmptyPrefixCodec {
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
pub struct PrefixOrbitCodec<OrbitId: OrdSymbol + Default = usize> {
    pub ids: Vec<OrbitId>,
    pub num_ignored_ids: usize,
    pub orbits: HashMap<OrbitId, Orbit>,
    pub categorical: MutCategorical<OrbitId>,
}

impl<OrbitId: OrdSymbol + Default> Permutable for PrefixOrbitCodec<OrbitId> {
    fn len(&self) -> usize {
        self.ids.len() - self.num_ignored_ids
    }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(!self.is_ignored(i));
        assert!(!self.is_ignored(j));

        let id_i = self.id(i);
        let id_j = self.id(j);
        if id_i == id_j {
            return;
        }
        let pi = self.orbits.get_mut(&id_i).unwrap();
        assert!(pi.remove(&i));
        assert!(pi.insert(j));
        let pj = self.orbits.get_mut(&id_j).unwrap();
        assert!(pj.remove(&j));
        assert!(pj.insert(i));
        self.ids.swap(i, j);
    }
}

impl<OrbitId: OrdSymbol + Default> PrefixOrbitCodec<OrbitId> {
    pub fn new(ids: Vec<OrbitId>, num_ignored_ids: usize) -> Self {
        let orbits = orbits_by_id(ids.iter().cloned().take(ids.len() - num_ignored_ids));
        let categorical = MutCategorical::new(orbits.iter().map(
            |(id, p)| (id.clone(), p.len())));
        Self { ids, num_ignored_ids, orbits, categorical }
    }

    pub fn push_id(&mut self) {
        let element = self.len();
        let id = self.ids[element].clone();
        self.num_ignored_ids -= 1;

        self.orbits.entry(id.clone()).or_default().insert(element);
        self.categorical.insert(id, 1);
    }

    pub fn pop_id(&mut self) {
        self.num_ignored_ids += 1;
        let element = self.len();
        let id = self.ids[element].clone();

        let partition = self.orbits.get_mut(&id).unwrap();
        assert!(partition.remove(&element));
        if partition.is_empty() {
            self.orbits.remove(&id);
        }
        self.categorical.remove(&id, 1);
    }

    pub fn is_ignored(&self, element: usize) -> bool {
        element >= self.len()
    }

    pub fn update_id(&mut self, element: usize, new_id: OrbitId) -> OrbitId {
        let old_id = mem::replace(&mut self.ids[element], new_id.clone());
        if self.is_ignored(element) {
            return old_id;
        }

        let old_part = self.orbits.get_mut(&old_id).unwrap();
        assert!(old_part.remove(&element));
        if old_part.is_empty() {
            self.orbits.remove(&old_id);
        }
        assert!(self.orbits.entry(new_id.clone()).or_default().insert(element));

        self.categorical.remove(&old_id, 1);
        self.categorical.insert(new_id, 1);
        old_id
    }
}

impl<OrbitId: OrdSymbol + Default> Distribution for PrefixOrbitCodec<OrbitId> {
    type Symbol = OrbitId;

    fn norm(&self) -> usize {
        self.categorical.norm()
    }
    fn pmf(&self, x: &Self::Symbol) -> usize { self.categorical.pmf(&x) }

    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        self.categorical.cdf(&x, i)
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        self.categorical.icdf(cf)
    }
}

impl<Id: OrdSymbol + Default> OrbitCodec for PrefixOrbitCodec<Id> {
    type OrbitId = Id;

    /// Returns the orbit identifier of the element.
    fn id(&self, element: usize) -> Id {
        self.ids[element].clone()
    }

    fn orbit(&self, orbit: &Id) -> &Orbit {
        &self.orbits[orbit]
    }
}

pub fn orbits_by_id<Id: OrdSymbol>(orbit_ids: impl ExactSizeIterator<Item=Id>) -> HashMap<Id, Orbit> {
    let mut orbits = HashMap::with_capacity(orbit_ids.len());

    for (index, value) in orbit_ids.enumerate() {
        orbits.entry(value).or_insert_with(BTreeSet::new).insert(index);
    }

    orbits
}

pub fn hash<T: Hash>(obj: T) -> usize {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish() as usize
}

pub trait Prefix: Permutable {
    type Full: Permutable;
    type Slice;

    fn pop_slice(&mut self) -> Self::Slice;
    fn push_slice(&mut self, slice: &Self::Slice);
    fn from_full(full: Self::Full) -> Self;
    fn into_full(self) -> Self::Full;
    fn last_(&self) -> usize { self.len() - 1 }
}

pub trait PlainPrefix: Prefix<Full=Self::GFull> + PlainPermutable {
    type GFull: Permutable + PlainPermutable;
}

impl<T: Prefix<Full: PlainPermutable> + PlainPermutable> PlainPrefix for T {
    type GFull = T::Full;
}

/// Prefix type where the prefix of length 0 contains the complete permutable object, and all slices are empty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointPrefix<P: Permutable> {
    pub full: P,
    pub len: usize,
}

impl<P: Permutable> Prefix for JointPrefix<P> {
    type Full = P;
    type Slice = ();

    fn pop_slice(&mut self) {
        self.len -= 1;
    }

    fn push_slice(&mut self, _: &()) {
        self.len += 1;
    }

    fn from_full(full: Self::Full) -> Self {
        Self { len: full.len(), full }
    }

    fn into_full(self) -> Self::Full {
        self.full
    }
}

impl<P: Permutable> Permutable for JointPrefix<P> {
    fn len(&self) -> usize {
        self.len
    }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len());
        assert!(j < self.len());
        self.full.swap(i, j)
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> GraphPlainPermutable for JointPrefix<Graph<N, E, Ty>>
where
    Graph<N, E, Ty>: PlainPermutable,
    GraphPrefix<N, E, Ty>: PlainPermutable,
{
    fn auts(&self) -> Automorphisms {
        let mut prefix = GraphPrefix::from_full(self.full.clone());
        prefix.num_unknown_nodes = self.full.len() - self.len();
        prefix.automorphisms()
    }
}

/// Needed so that Unordered<JointPrefix<Vec<S>>> implements Eq and can be used as Symbol in PrefixShuffleCodec.
impl<S: Symbol> PlainPermutable for JointPrefix<Vec<S>> {
    fn automorphism_group(&self) -> PermutationGroup {
        unimplemented!()
    }

    fn canon(&self) -> Permutation {
        unimplemented!()
    }

    fn orbits(&self) -> Orbits {
        unimplemented!()
    }
}

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

pub trait OrdSymbol: Symbol + Ord + Hash {}

impl<T: Symbol + Ord + Hash> OrdSymbol for T {}
