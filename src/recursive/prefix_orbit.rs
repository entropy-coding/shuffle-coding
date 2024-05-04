use std::collections::{BTreeSet, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem;

use crate::codec::{Distribution, MutCategorical, MutDistribution, OrdSymbol};
use crate::permutable::{Orbit, Permutable};
use crate::recursive::OrbitCodec;

/// Orbit codec useful for recursive shuffle coding on multisets and graphs.
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
        let id = &self.ids[element];

        let partition = self.orbits.get_mut(id).unwrap();
        assert!(partition.remove(&element));
        if partition.is_empty() {
            self.orbits.remove(id);
        }
        self.categorical.remove(id, 1);
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

impl<OrbitId: OrdSymbol + Default> OrbitCodec for PrefixOrbitCodec<OrbitId> {
    fn id(&self, index: usize) -> OrbitId {
        self.ids[index].clone()
    }

    fn index(&self, id: &OrbitId) -> usize {
        *self.orbits[id].first().unwrap()
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