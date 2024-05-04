use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{stdout, Write};
use std::marker::PhantomData;
use std::ops::Deref;

use clap::Parser;
use itertools::Itertools;

use crate::codec::{Bernoulli, Categorical, Codec, CodecTestResults, Independent, Message, OrdSymbol, Symbol, Uniform, UniformCodec};
use crate::datasets::{Dataset, SourceInfo};
use crate::graph::{EdgeType, Graph, Undirected, with_isomorphism_test_max_len};
use crate::graph_codec::{EmptyCodec, erdos_renyi_indices, GraphIID, num_all_edge_indices, polya_urn};
use crate::param_codec::{AutoregressiveErdosReyniGraphDatasetParamCodec, AutoregressivePolyaUrnGraphDatasetParamCodec, CategoricalParamCodec, EmptyParamCodec, ErdosRenyiParamCodec, GraphDatasetParamCodec, ParametrizedIndependent, PolyaUrnParamCodec, UniformParamCodec, WrappedParametrizedIndependent};
use crate::permutable::{Permutable, PermutableCodec, Unordered};
use crate::permutable::PermutationUniform;
use crate::plain::{plain_shuffle_codec, PlainPermutable};
use crate::plain::coset_recursive::coset_recursive_shuffle_codec;
use crate::plain::labelled_graph::EdgeLabelledGraph;
use crate::recursive::{Autoregressive, ShuffleCodec, SliceCodecs};
use crate::recursive::graph::GraphPrefix;
use crate::recursive::graph::incomplete::WLOrbitCodecs;
use crate::recursive::graph::slice::{ErdosRenyiSliceCodecs, PolyaUrnSliceCodecs};
use crate::recursive::joint::{JointPrefix, JointSliceCodecs};
use crate::recursive::plain_orbit::PlainOrbitCodecs;

#[derive(Clone, Debug, Parser, PartialEq)]
pub struct Config {
    /// Ignore node and edge labels for all graph datasets, only use graph structure.
    #[arg(long)]
    pub nolabels: bool,

    /// Report the following graph dataset statistics: Permutation and symmetry bits, and time needed for retrieving automorphisms from nauty.
    #[clap(long)]
    pub symstats: bool,
    /// Report the following graph dataset statistics: graphs total_nodes total_edges selfloops edge_prob node_entropy node_labels edge_entropy edge_labels.
    #[arg(long)]
    pub stats: bool,

    /// Do not test ordered codecs. These are tested by default.
    #[clap(short = 'o', long)]
    pub no_ordered: bool,
    /// Do not test codecs for only the parameters (no data). These are tested by default.
    #[clap(long)]
    pub no_param: bool,
    /// Do not test joint codecs for data and parameters. These are tested by default.
    #[clap(long)]
    pub no_parametrized: bool,
    /// Test codecs for only the data (no parameters). By default, these are not tested.
    #[clap(short, long)]
    pub unparametrized: bool,

    /// Shuffle coding test configuration.
    #[structopt(flatten)]
    pub shuffle: TestConfig,

    /// Use Erdős–Rényi model with uniform node/edge labels.
    #[arg(long = "eru", default_value_t = false)]
    pub uniform_er: bool,
    /// Use Pólya urn model allowing redraws and self-loops.
    #[arg(long = "pur", default_value_t = false)]
    pub redraw_pu: bool,

    /// Add custom dataset source via '<name>:<index file URL>'. The defaults are
    /// TU:https://raw.githubusercontent.com/chrsmrrs/datasets/gh-pages/_docs/datasets.md,
    /// SZIP:https://raw.githubusercontent.com/juliuskunze/szip-graphs/main/datasets.md, and
    /// REC:https://raw.githubusercontent.com/entropy-coding/rec-graphs/main/datasets.md
    /// Custom index files must be in the same format as the defaults, and refer to zip files of
    /// datasets that follow the TUDataset format described at https://chrsmrrs.github.io/datasets/docs/format/,
    /// except that no graph labels are required. Node/edge/graph attributes are ignored.
    #[arg(name = "source", long, value_parser = parse_index)]
    pub sources: Vec<SourceInfo>,
}

impl Deref for Config {
    type Target = TestConfig;
    fn deref(&self) -> &Self::Target { &self.shuffle }
}

pub struct Model {
    pub name: &'static str,
    pub is_autoregressive: bool,
}

impl Config {
    pub fn pu_models_original(&self) -> Vec<bool> {
        let mut original = vec![];
        if self.pu {
            original.push(false);
        }
        if self.redraw_pu {
            original.push(true);
        }
        original
    }

    pub fn models(&self) -> Vec<Model> {
        let mut models = vec![];
        if self.er {
            models.push(Model { name: "ER", is_autoregressive: false });
        }
        if self.uniform_er {
            models.push(Model { name: "ERu", is_autoregressive: false });
        }
        if self.pu {
            models.push(Model { name: "PU", is_autoregressive: false });
        }
        if self.redraw_pu {
            models.push(Model { name: "PUr", is_autoregressive: false });
        }
        if self.ae {
            models.push(Model { name: "AE", is_autoregressive: true });
        }
        if self.ap {
            models.push(Model { name: "AP", is_autoregressive: true });
        }
        models
    }
}

fn parse_index(s: &str) -> Result<SourceInfo, String> {
    let (name, index_url) = s.splitn(2, ':').collect_tuple().expect("Expected '<name>:<index file URL>'.");
    Ok(SourceInfo::new(name, index_url))
}

pub fn print_symmetry(permutables: &Vec<impl PlainPermutable>) {
    let perm_bits = permutables.iter().map(|x| PermutationUniform { len: x.len() }.uni_bits()).sum::<f64>();
    let mut symmetry_bits = 0.;
    let symmetry_sec = timeit_loops!(1, { symmetry_bits = permutables.iter().map(|x| x.automorphisms().bits).sum::<f64>(); });
    print_flush!("{perm_bits} {symmetry_bits} {symmetry_sec} ");
}

pub struct Benchmark {
    pub datasets: Vec<Dataset>,
    pub config: Config,
}

impl Benchmark {
    pub fn timed_run(&self) {
        let time = timeit_loops!(1, { self.run(); });
        println!("Finished in {time:.1}s.");
    }

    pub fn run(&self) {
        self.print_headers();

        for dataset in &self.datasets {
            DatasetBenchmark { dataset, config: &self.config }.run();
        }
    }

    fn print_headers(&self) {
        print!("src ctg dataset labels ");

        if self.config.symstats {
            print!("permutation symmetry sym_sec ");
        }
        if self.config.stats {
            print!("graphs total_nodes total_edges selfloops edge_prob node_entropy node_labels edge_entropy edge_labels ")
        }

        let print_codec = |codec: &str| print!("{codec} net enc dec ");
        let incomplete = format!("wl{}{}", self.config.wl_iter, if self.config.wl_extra_half_iter { "" } else { "-" });

        for Model { name, is_autoregressive } in self.config.models() {
            if !self.config.no_param {
                print_codec(&format!("par_{name}"));
            }

            let print_codec_p = |prefix: &str| {
                if self.config.unparametrized {
                    print_codec(&format!("n_{prefix}{name}"));
                }
                if !self.config.no_parametrized {
                    print_codec(&format!("{prefix}{name}"));
                }
            };

            if !self.config.no_ordered {
                print_codec_p(&"ord_")
            }

            if self.config.shuffle.plain {
                print_codec_p(&"");
            }
            if self.config.shuffle.plain_coset_recursive {
                print_codec_p(&"pc_");
            }
            if self.config.joint {
                if self.config.shuffle.complete {
                    print_codec_p(&"cj_");
                }
                if !self.config.shuffle.no_incomplete {
                    print_codec_p(&format!("{incomplete}j_"));
                }
            }
            if is_autoregressive && self.config.recursive {
                if self.config.shuffle.complete {
                    print_codec_p(&"cr_");
                }
                if !self.config.shuffle.no_incomplete {
                    print_codec_p(&format!("{incomplete}r_"));
                }
            }
        }
        println!();
    }
}

pub struct DatasetBenchmark<'a> {
    pub dataset: &'a Dataset,
    pub config: &'a Config,
}

impl DatasetBenchmark<'_> {
    fn er_indices(&self) -> impl Fn(Vec<usize>, bool) -> ErdosRenyiParamCodec<Undirected> + '_ + Clone {
        |nums_nodes, loops| ErdosRenyiParamCodec::new(nums_nodes, loops)
    }
    fn pu_indices(&self, original: bool) -> impl Fn(Vec<usize>, bool) -> PolyaUrnParamCodec<Undirected> + '_ + Clone {
        move |nums_nodes, loops| PolyaUrnParamCodec::new(nums_nodes, loops || original, original)
    }

    fn verify_and_print_stats(&self, stats: &DatasetStats) {
        if !self.config.stats { return; }
        assert_eq!(self.dataset.num_graphs, stats.num_graphs);
        assert_eq!(format!("{:.2}", self.dataset.avg_nodes), format!("{:.2}", stats.avg_nodes()));
        assert_eq!(format!("{:.2}", self.dataset.avg_edges), format!("{:.2}", stats.avg_edges()));

        let (node_entropy, node_labels) = if let Some(n) = &stats.node_label { (n.entropy(), n.masses.len()) } else { (0., 0) };
        let (edge_entropy, edge_labels) = if let Some(n) = &stats.edge_label { (n.entropy(), n.masses.len()) } else { (0., 0) };
        print_flush!("{} {} {} {} {} {} {} {} {} ", stats.num_graphs, stats.total_nodes, stats.total_edges, stats.loops, stats.edge.prob(), node_entropy, node_labels, edge_entropy, edge_labels);
    }

    pub fn run(&self) {
        print_flush!("{} {} {} ", self.dataset.source.name, self.dataset.category[..3].to_owned(), self.dataset.name);

        if self.config.nolabels || !(self.dataset.has_node_labels || self.dataset.has_edge_labels) {
            print_flush!("none ");
            let graphs = self.sorted(self.dataset.unlabelled_graphs());
            self.verify_and_print_stats(&DatasetStats::unlabelled(&graphs));

            let er = || {
                self.run_joint_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: EmptyParamCodec::default(), edge: EmptyParamCodec::default(), indices: self.er_indices() },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, .. } = DatasetStats::unlabelled(&graphs);
                        Independent::new(graphs.iter().map(|x| {
                            GraphIID::new(x.len(), erdos_renyi_indices(x.len(), edge.clone(), loops), EmptyCodec::default(), EmptyCodec::default())
                        }).collect_vec())
                    },
                }, &graphs)
            };
            if self.config.uniform_er { er() }
            if self.config.er { er() }
            for original in self.config.pu_models_original() {
                self.run_joint_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: EmptyParamCodec::default(), edge: EmptyParamCodec::default(), indices: self.pu_indices(original) },
                    infer: |graphs| {
                        let DatasetStats { loops, .. } = DatasetStats::unlabelled(&graphs);
                        Independent::new(graphs.iter().map(|x| {
                            let edge_indices = polya_urn(x.len(), x.num_edges(), loops || original, original);
                            GraphIID::new(x.len(), edge_indices, EmptyCodec::default(), EmptyCodec::default())
                        }).collect_vec())
                    },
                }, &graphs);
            }

            if self.config.ae {
                self.run_autoregressive_model(ParametrizedIndependent {
                    param_codec: AutoregressiveErdosReyniGraphDatasetParamCodec { joint: GraphDatasetParamCodec { node: EmptyParamCodec::default(), edge: EmptyParamCodec::default(), indices: self.er_indices() } },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, .. } = DatasetStats::unlabelled(&graphs);
                        Independent::new(graphs.iter().map(|x| {
                            Autoregressive(ErdosRenyiSliceCodecs {
                                len: x.len(),
                                has_edge: edge.clone(),
                                node: EmptyCodec::default(),
                                edge: EmptyCodec::default(),
                                loops,
                                phantom: PhantomData,
                            })
                        }).collect_vec())
                    },
                }, &graphs)
            }
            if self.config.ap {
                self.run_autoregressive_model(ParametrizedIndependent {
                    param_codec: AutoregressivePolyaUrnGraphDatasetParamCodec { node: EmptyParamCodec::default(), edge: EmptyParamCodec::default(), phantom: PhantomData },
                    infer: |graphs| {
                        let DatasetStats { loops, .. } = DatasetStats::unlabelled(&graphs);
                        Independent::new(graphs.iter().map(|x| {
                            Autoregressive(PolyaUrnSliceCodecs {
                                len: x.len(),
                                node: EmptyCodec::default(),
                                edge: EmptyCodec::default(),
                                loops,
                                phantom: PhantomData,
                            })
                        }).collect_vec())
                    },
                }, &graphs)
            }
        } else if let Some(graphs) = self.dataset.edge_labelled_graphs() {
            print_flush!("edges ");
            let graphs = self.sorted(graphs.clone());
            self.verify_and_print_stats(&DatasetStats::edge_labelled(&graphs));

            if self.config.er {
                self.run_joint_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: CategoricalParamCodec, edge: CategoricalParamCodec, indices: self.er_indices() },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, node_label, edge_label, .. } = DatasetStats::edge_labelled(&graphs);
                        Independent::new(graphs.iter().map(|x| GraphIID::new(
                            x.len(), erdos_renyi_indices(x.len(), edge.clone(), loops), node_label.clone().unwrap(), edge_label.clone().unwrap())).collect_vec())
                    },
                }, &graphs);
            }
            if self.config.uniform_er {
                self.run_joint_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: UniformParamCodec::default(), edge: UniformParamCodec::default(), indices: self.er_indices() },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, node_label, edge_label, .. } = DatasetStats::edge_labelled(&graphs);
                        let size = node_label.unwrap().masses.len();
                        let node_label = Uniform::new(size);
                        let size = edge_label.unwrap().masses.len();
                        let edge_label = Uniform::new(size);
                        Independent::new(graphs.iter().map(|x| GraphIID::new(
                            x.len(), erdos_renyi_indices(x.len(), edge.clone(), loops), node_label.clone(), edge_label.clone())).collect_vec())
                    },
                }, &graphs);
            }
            for original in self.config.pu_models_original() {
                self.run_joint_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: CategoricalParamCodec, edge: CategoricalParamCodec, indices: self.pu_indices(original) },
                    infer: |graphs| {
                        let DatasetStats { loops, node_label, edge_label, .. } = DatasetStats::edge_labelled(&graphs);
                        Independent::new(graphs.iter().map(|x| {
                            let edge_indices = polya_urn(x.len(), x.num_edges(), loops || original, original);
                            GraphIID::new(x.len(), edge_indices, node_label.clone().unwrap(), edge_label.clone().unwrap())
                        }).collect_vec())
                    },
                }, &graphs);
            }
        } else if let Some(graphs) = self.dataset.node_labelled_graphs() {
            print_flush!("nodes ");
            let graphs = self.sorted(graphs.clone());
            self.verify_and_print_stats(&DatasetStats::node_labelled(&graphs));
            if self.config.er {
                self.run_joint_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: CategoricalParamCodec, edge: EmptyParamCodec::default(), indices: self.er_indices() },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, node_label, .. } = DatasetStats::node_labelled(&graphs);
                        Independent::new(graphs.iter().map(|x| {
                            let edge_indices = erdos_renyi_indices(x.len(), edge.clone(), loops);
                            GraphIID::new(x.len(), edge_indices, node_label.clone().unwrap(), EmptyCodec::default())
                        }).collect_vec())
                    },
                }, &graphs);
            }
            if self.config.uniform_er {
                self.run_joint_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: UniformParamCodec::default(), edge: EmptyParamCodec::default(), indices: self.er_indices() },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, node_label, .. } = DatasetStats::node_labelled(&graphs);
                        let size = node_label.unwrap().masses.len();
                        let node_label = Uniform::new(size);
                        Independent::new(graphs.iter().map(|x| {
                            let edge_indices = erdos_renyi_indices(x.len(), edge.clone(), loops);
                            GraphIID::new(x.len(), edge_indices, node_label.clone(), EmptyCodec::default())
                        }).collect_vec())
                    },
                }, &graphs);
            }
            for original in self.config.pu_models_original() {
                self.run_joint_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: CategoricalParamCodec, edge: EmptyParamCodec::default(), indices: self.pu_indices(original) },
                    infer: |graphs| {
                        let DatasetStats { loops, node_label, .. } = DatasetStats::node_labelled(&graphs);
                        Independent::new(graphs.iter().map(|x| {
                            let edge_indices = polya_urn(x.len(), x.num_edges(), loops || original, original);
                            GraphIID::new(x.len(), edge_indices, node_label.clone().unwrap(), EmptyCodec::default())
                        }).collect_vec())
                    },
                }, &graphs);
            }
        }
        println!()
    }

    fn run_autoregressive_model<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType, S: SliceCodecs<Prefix=GraphPrefix<N, E, Ty>> + Symbol>(
        &self,
        codec: ParametrizedIndependent<impl Codec<Symbol=Independent<Autoregressive<S>>>, impl Fn(&Vec<Graph<N, E, Ty>>) -> Independent<Autoregressive<S>> + Clone>,
        graphs: &Vec<Graph<N, E, Ty>>)
    where
        Graph<N, E, Ty>: PlainPermutable,
        GraphPrefix<N, E, Ty>: PlainPermutable,
    {
        with_isomorphism_test_max_len(self.config.isomorphism_test_max_len, || {
            self.run_joint_model(codec.clone(), graphs);

            if !self.config.recursive {
                return;
            }

            let param = (codec.infer)(&graphs);
            let unordered = graphs.iter().cloned().map(Unordered).collect_vec();
            let to_ordered = Box::new(|x: Vec<Unordered<_>>| x.into_iter().map(|x| x.into_ordered()).collect());

            if self.config.complete {
                let codec = WrappedParametrizedIndependent {
                    parametrized_codec: codec.clone(),
                    data_to_inner: to_ordered.clone(),
                    codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| ShuffleCodec::new(c.0, PlainOrbitCodecs::new())))),
                };
                if self.config.unparametrized {
                    test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, &self.config.initial_message());
                }
                if !self.config.no_parametrized {
                    test_and_print(&codec, &unordered, &self.config.initial_message());
                }
            }
            if !self.config.no_incomplete {
                let wl_iter = self.config.wl_iter;
                let extra_half_iter = self.config.wl_extra_half_iter;
                let codec = WrappedParametrizedIndependent {
                    parametrized_codec: codec.clone(),
                    data_to_inner: to_ordered.clone(),
                    codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| ShuffleCodec::new(c.0, WLOrbitCodecs::<GraphPrefix<N, E, Ty>, N, E, Ty>::new(wl_iter, extra_half_iter))))),
                };
                if self.config.unparametrized {
                    test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, &self.config.initial_message());
                }
                if !self.config.no_parametrized {
                    test_and_print(&codec, &unordered, &self.config.initial_message());
                }
            }
        })
    }

    fn run_joint_model<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType, C: PermutableCodec<Symbol=Graph<N, E, Ty>> + Symbol>(
        &self,
        codec: ParametrizedIndependent<impl Codec<Symbol=Independent<C>>, impl Fn(&Vec<C::Symbol>) -> Independent<C> + Clone>,
        graphs: &Vec<C::Symbol>)
    where
        Graph<N, E, Ty>: PlainPermutable,
        GraphPrefix<N, E, Ty>: PlainPermutable,
    {
        with_isomorphism_test_max_len(self.config.isomorphism_test_max_len, || {
            let param = (codec.infer)(&graphs);
            if !self.config.no_param {
                test_and_print(&codec.param_codec, &param, &self.config.initial_message());
            }
            if !self.config.no_ordered {
                if !self.config.no_parametrized {
                    test_and_print(&codec, &graphs, &self.config.initial_message());
                }
                if self.config.unparametrized {
                    test_and_print(&param, &graphs, &self.config.initial_message());
                }
            }

            let unordered = graphs.iter().cloned().map(Unordered).collect_vec();
            let to_ordered = Box::new(|x: Vec<Unordered<_>>| x.into_iter().map(|x| x.into_ordered()).collect());

            if self.config.plain {
                let codec = WrappedParametrizedIndependent {
                    parametrized_codec: codec.clone(),
                    data_to_inner: to_ordered.clone(),
                    codec_from_inner: Box::new(|x| Independent::new(x.codecs.iter().cloned().map(plain_shuffle_codec))),
                };
                if self.config.unparametrized {
                    test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, &self.config.initial_message());
                }
                if !self.config.no_parametrized {
                    test_and_print(&codec, &unordered, &self.config.initial_message());
                }
            }
            if self.config.plain_coset_recursive {
                let codec = WrappedParametrizedIndependent {
                    parametrized_codec: codec.clone(),
                    data_to_inner: to_ordered.clone(),
                    codec_from_inner: Box::new(|x| Independent::new(x.codecs.iter().cloned().map(coset_recursive_shuffle_codec))),
                };
                if self.config.unparametrized {
                    test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, &self.config.initial_message());
                }
                if !self.config.no_parametrized {
                    test_and_print(&codec, &unordered, &self.config.initial_message());
                }
            }

            if self.config.joint {
                if self.config.complete {
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| ShuffleCodec::new(JointSliceCodecs::new(c), PlainOrbitCodecs::new())))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, &self.config.initial_message());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, &self.config.initial_message());
                    }
                }

                if !self.config.no_incomplete {
                    let wl_iter = self.config.wl_iter;
                    let extra_half_iter = self.config.wl_extra_half_iter;
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| ShuffleCodec::new(JointSliceCodecs::new(c), WLOrbitCodecs::<JointPrefix<_>, _, _, _>::new(wl_iter, extra_half_iter))))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, &self.config.initial_message());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, &self.config.initial_message());
                    }
                }
            }
        })
    }

    fn sorted<N: Symbol, E: Symbol, Ty: EdgeType>(&self, mut graphs: Vec<Graph<N, E, Ty>>) -> Vec<Graph<N, E, Ty>>
    where
        Graph<N, E, Ty>: PlainPermutable,
    {
        graphs.sort_unstable_by_key(|x| -(x.len() as isize));
        if self.config.symstats {
            print_symmetry(&graphs);
        }
        graphs
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetStats {
    pub num_graphs: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub loops: bool,
    pub edge: Bernoulli,
    pub node_label: Option<Categorical>,
    pub edge_label: Option<Categorical>,
}

impl DatasetStats {
    pub fn unlabelled<N: Symbol, E: Symbol, Ty: EdgeType>(graphs: &Vec<Graph<N, E, Ty>>) -> Self {
        let num_graphs = graphs.len();
        let loops = graphs.iter().any(|x| x.has_selfloops());
        let total_nodes = graphs.iter().map(|x| x.len()).sum::<usize>();
        let total_edges = graphs.iter().map(|x| x.num_edges()).sum::<usize>();
        let total_possible_edges = graphs.iter().map(|x| num_all_edge_indices::<Ty>(x.len(), loops)).sum::<usize>();
        let edge = Bernoulli::new(total_edges, total_possible_edges);
        Self { num_graphs, total_nodes, total_edges, loops, edge, node_label: None, edge_label: None }
    }

    pub fn node_labelled<E: Symbol, Ty: EdgeType>(graphs: &Vec<Graph<usize, E, Ty>>) -> Self {
        Self { node_label: Some(Self::node_label_dist(graphs)), ..Self::unlabelled(graphs) }
    }

    pub fn edge_labelled<Ty: EdgeType>(graphs: &Vec<EdgeLabelledGraph<Ty>>) -> Self {
        Self { edge_label: Some(Self::edge_label_dist(graphs)), ..Self::node_labelled(graphs) }
    }

    fn edge_label_dist<Ty: EdgeType>(graphs: &Vec<EdgeLabelledGraph<Ty>>) -> Categorical {
        Self::dist(&graphs.iter().flat_map(|x| x.edge_labels()).counts())
    }

    fn node_label_dist<E: Symbol, Ty: EdgeType>(graphs: &Vec<Graph<usize, E, Ty>>) -> Categorical {
        Self::dist(&graphs.iter().flat_map(|x| x.node_labels()).counts())
    }

    fn dist(counts: &HashMap<usize, usize>) -> Categorical {
        let masses = (0..counts.keys().max().unwrap() + 1).map(|x| counts.get(&x).cloned().unwrap_or_default()).collect_vec();
        Categorical::new(masses)
    }

    fn avg_nodes(&self) -> f64 {
        self.total_nodes as f64 / self.num_graphs as f64
    }

    fn avg_edges(&self) -> f64 {
        self.total_edges as f64 / self.num_graphs as f64
    }
}

pub fn test_and_print<S: Symbol>(codec: &impl Codec<Symbol=S>, symbol: &S, initial: &Message) -> CodecTestResults {
    let out = codec.test(symbol, initial);
    let CodecTestResults { bits, amortized_bits, enc_sec, dec_sec } = &out;
    print_flush!("{bits} {amortized_bits} {enc_sec} {dec_sec} ");
    out
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::datasets::dataset;

    use super::*;

    #[test]
    fn test() {
        Benchmark {
            datasets: vec!(dataset("MUTAG")),
            config: Config::parse_from(["", "--stats", "--er"]),
        }.timed_run();
    }
}

#[derive(Clone, Debug, Eq, Parser, PartialEq)]
pub struct TestConfig {
    /// Run plain shuffle coding.
    #[clap(short, long)]
    pub plain: bool,
    /// Run plain shuffle coding with a recursive coset codec.
    #[clap(hide = true, long)]
    pub plain_coset_recursive: bool,

    /// Run joint shuffle coding.
    #[clap(short = 'j', long)]
    pub joint: bool,

    /// Run recursive shuffle coding.
    #[clap(short = 'r', long)]
    pub recursive: bool,

    /// Use complete variant of joint/recursive shuffle coding.
    #[clap(short, long)]
    pub complete: bool,
    /// Do not use incomplete variant (based on the Weisfeiler-Lehman hash) of joint/recursive shuffle coding.
    #[clap(long)]
    pub no_incomplete: bool,

    /// Number of iterations in the Weisfeiler-Lehman hash used for incomplete shuffle coding.
    #[clap(short, long, default_value = "1")]
    pub wl_iter: usize,

    /// Use an extra half iteration in the Weisfeiler-Lehman hash used for incomplete shuffle coding.
    #[clap(hide = true, long)]
    pub wl_extra_half_iter: bool,

    /// Seed for the message tail random generator providing initial bits.
    #[clap(long, default_value = "0")]
    pub seed: usize,
    /// If some, verify permute group action and canon labelling axioms with the given seed.
    #[cfg(test)]
    #[clap(skip)]
    pub axioms_seed: Option<usize>,

    /// Maximum number of graph vertices to verify isomorphism of unordered objects after decoding.
    /// By default, isomorphism is not verified. Can be very slow for large graphs.
    #[clap(short, long, default_value = "0")]
    pub isomorphism_test_max_len: usize,

    /// Use Erdős–Rényi model.
    #[clap(long)]
    pub er: bool,

    /// Use Pólya urn model.
    #[clap(long)]
    pub pu: bool,

    /// Use autoregressive Erdős–Rényi model.
    #[clap(long)]
    pub ae: bool,

    /// Use autoregressive model approximating Pólya urn.
    #[clap(long)]
    pub ap: bool,
}

impl TestConfig {
    #[cfg(test)]
    pub fn test(seed: usize) -> Self {
        Self {
            axioms_seed: Some(seed),
            plain: true,
            plain_coset_recursive: true,
            recursive: true,
            joint: true,
            complete: true,
            no_incomplete: false,
            er: true,
            pu: true,
            ae: true,
            ap: true,
            wl_iter: 1,
            wl_extra_half_iter: true,
            seed,
            isomorphism_test_max_len: usize::MAX,
        }
    }

    pub fn initial_message(&self) -> Message {
        Message::random(self.seed)
    }
}
