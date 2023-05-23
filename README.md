# Shuffle Coding

This is the anonymized implementation for the submission "Entropy Coding of Unordered Datastructures", which is currently under review.

## Run

Apart from [Rust](https://www.rust-lang.org/tools/install), you need a C compiler to build this project due to dependency on [nauty and Traces' Rust bindings](https://github.com/a-maier/nauty-Traces-sys).

Use the following commands to run all the experiments from the paper:

```shell
cargo run --release -- --symstats --stats --er --shuffle AIDS..mit_ct2 reddit_threads..TRIANGLES
cargo run --release -- --symstats --stats --er --eru --pu --pur --shuffle MUTAG PTC_MR
cargo run --release -- --symstats --stats --er --eru --pu --pur --shuffle --plain MUTAG PTC_MR ZINC_full..ZINC_val PROTEINS IMDB-BINARY IMDB-MULTI
cargo run --release -- --symstats --stats --er --pu --shuffle SZIP
```

Full results can be found [here](https://docs.google.com/spreadsheets/d/1YP0om6hNktaUhyFzFVVSh5SCTefZ8kNafYDYYIDolhM/edit#gid=1483743764).

To retrieve standard deviations for the stochastic Polya urn models, you can set a random seed (0, 1 and 2 were used for the paper):
```shell
cargo run --release -- MUTAG PTC_MR ZINC_full..ZINC_val PROTEINS IMDB-BINARY IMDB-MULTI --plain --pu --shuffle --seed=0
cargo run --release -- MUTAG PTC_MR --pu --shuffle --seed=0 
cargo run --release -- SZIP --pu --shuffle --seed=0
```

# SZIP Graph Data

The undirected graphs collected for evaluation of Structural ZIP in [table 1 of [1]](https://www.cs.purdue.edu/homes/spa/papers/structure10.pdf#page=15) are in the szip-graphs folder.
The data was kindly supplied by the authors of [1] and published with their permission.
The graphs are in [TUDataset format](https://chrsmrrs.github.io/datasets/docs/format/) and summarized in [datasets.md](szip/datasets.md).

[1] Yongwook Choi and Wojciech Szpankowski: [Compression of Graphical Structures: Fundamental Limits, Algorithms, and Experiments.](https://www.cs.purdue.edu/homes/spa/papers/structure10.pdf) (2011)