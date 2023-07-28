// TODO: Move dims out of the result structs and add directly into hashmap, removes unnecessary duplicates. (More difficult than anticipated...)
// TODO: Add seeding input !!

// Standard imports
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{env, fs};
use std::env::Args;

// External imports
use rand::seq::SliceRandom;
use rand::SeedableRng;
use threadpool::ThreadPool;
use rand_chacha::ChaCha8Rng;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// Local imports
mod router;
use router::Routing;

mod json_serializer;
use json_serializer::PrettyFormatter2;

// Constants
const Z: usize = 7;

const DEFAULT_SAVE: bool = false;
const DEFAULT_PERMUTATIONS: u32 = 200;
const DEFAULT_CORES: usize = 6;

fn check_empty<T>(v: &Vec<T>) -> bool {
    return v.is_empty();
}

// Add new impl
#[derive(Serialize, Debug)]
struct Result {
    dims: [usize; 2],
    npaths: Vec<u32>,
    #[serde(skip_serializing_if = "check_empty")]
    paths: Vec<Vec<Vec<usize>>>,
}

impl Result {
    pub fn new(dims: [usize; 2]) -> Self {
        Result { 
            dims, 
            npaths: Vec::new(), 
            paths: Vec::new() 
        }
    }
}

// Add new impl
#[derive(Serialize, Debug)]
struct ResultPerm {
    dims: [usize; 2],
    npaths: Vec<Vec<u32>>,
    #[serde(skip_serializing_if = "check_empty")]
    paths: Vec<Vec<Vec<Vec<usize>>>>,
}

impl ResultPerm {
    pub fn new(dims: [usize; 2]) -> Self {
        ResultPerm { 
            dims, 
            npaths: Vec::new(), 
            paths: Vec::new()
        }
    }
}

#[derive(Deserialize, Debug)]
struct Data {
    indices: HashMap<String, usize>,
    dims: Vec<usize>,
    pathlists: HashMap<String, Vec<Vec<[String; 2]>>>,
}

fn read_json<P: AsRef<Path>>(path: P) -> Value {
    let file: String = fs::read_to_string(path).expect("Unable to read file");
    serde_json::from_str(&file).expect("Error occurred while parsing JSON")
}

// TODO: Change name of function
fn nodes_to_indices(
    node_dict: &HashMap<String, usize>,
    t_pair_vecs: &Vec<Vec<[String; 2]>>,
) -> Vec<Vec<[usize; 2]>> {
    let mut return_vec: Vec<Vec<[usize; 2]>> = Vec::new();
    for t_pair_vec in t_pair_vecs {
        let mut index_tuples: Vec<[usize; 2]> = Vec::new();
        for tuple in t_pair_vec {
            index_tuples.push([
                *node_dict.get(&tuple[0]).unwrap(),
                *node_dict.get(&tuple[1]).unwrap(),
            ]);
        }
        return_vec.push(index_tuples);
    }
    return_vec
}

fn handle_npathlist(
    router: &mut Routing,
    t_pairs: &mut Vec<Vec<[usize; 2]>>,
    collect: bool,
    rng: &mut Option<&mut ChaCha8Rng>
) -> Result {
    let mut result: Result = Result::new([router.dims[1], router.dims[2]]);
    for t_pair in t_pairs {
        if rng.is_some() {
            t_pair.shuffle(rng.as_mut().unwrap());
        }
        let (nsuccessful, paths): (u32, Vec<Vec<usize>>) = router.run(t_pair.to_vec(), collect);
        result.npaths.push(nsuccessful);
        if collect {
            result.paths.push(paths);
        }
    }
    result
}

fn handle_perms(
    router: &mut Routing,
    t_pairs: &mut Vec<Vec<[usize; 2]>>,
    n: u32,
    collect: bool,
    seed: u32
) -> ResultPerm {
    let mut result_permuted: ResultPerm = ResultPerm::new([router.dims[1], router.dims[2]]);
    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(seed.into());
    for _ in 0..n {
        let result: Result = handle_npathlist(router, t_pairs, collect, &mut Some(&mut rng));
        result_permuted.npaths.push(result.npaths);
        if collect {
            result_permuted.paths.push(result.paths);
        }
    }
    result_permuted
}

#[derive(Debug)]
struct FormattedArgs {
    save_paths: bool,
    perms: u32,
    data_path: PathBuf,
    prefix: PathBuf,
    results_path: PathBuf,
    results_perm_path: PathBuf,
    cores: usize,
}
impl FormattedArgs {
    pub fn new(args: &mut Args) -> Self {
        args.next();
        // let path = args.next().unwrap();
        let data_path: PathBuf = Path::new(&args.next().unwrap()).to_owned();
        let results_path: String = data_path
            .parent()
            .unwrap()
            .to_str()
            .unwrap()
            .replace("data", "results");
        let prefix: PathBuf = Path::new(&results_path).to_owned();
        let mut results_path: PathBuf = prefix.clone();
        let mut results_perm_path: PathBuf = prefix.clone();

        let mut save_paths: bool = DEFAULT_SAVE;
        let mut perms: u32 = DEFAULT_PERMUTATIONS;
        let mut cores: usize = DEFAULT_CORES;

        let mut extension: String = String::new();

        while let Some(current) = args.next() {
            match &current[..] {
                "--cores" | "-c" => cores = args.next().unwrap().parse().unwrap(),
                "--save" | "-s" => save_paths = args.next().unwrap() == "1",
                "--permutations" | "-p" => {
                    perms = args.next().unwrap().parse().unwrap()
                }
                "--extension" | "-e" => {
                    extension = args.next().unwrap().clone();
                }
                _ => continue,
            }
        }
        
        if save_paths {
            results_path.push(format!("np_results_ext_{}.json", extension));
            results_perm_path.push(format!("p_results_ext_{}.json", extension));
        } else {
            results_path.push(format!("np_results_{}.json", extension));
            results_perm_path.push(format!("p_results_{}.json", extension));
        }

        Self {
            save_paths,
            perms,
            data_path,
            prefix,
            results_path,
            results_perm_path,
            cores,
        }
    }
}

// Fix naming
fn write_to_json<T: std::fmt::Debug + Serialize>(
    output: Arc<Mutex<HashMap<String, T>>>,
    path: PathBuf,
) {
    let out: HashMap<String, T> = Arc::try_unwrap(output).unwrap().into_inner().unwrap();
    let json_result: Value = json!(out);
    let formatter = PrettyFormatter2::new();
    let buf: Vec<u8> = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(buf, formatter);
    json_result.serialize(&mut serializer).unwrap();
    // let json_result_string: String = serde_json::to_string_pretty(&json_result).unwrap();
    let json_result_string = String::from_utf8(serializer.into_inner()).unwrap();
    fs::write(path, &json_result_string).unwrap();
}

fn main() {
    // let argv = env::args();
    let args: FormattedArgs = FormattedArgs::new(&mut env::args());
    if !Path::exists(&args.prefix) {
        fs::create_dir_all(&args.prefix).unwrap();
    }

    let data: Data = serde_json::from_value(read_json(args.data_path)).unwrap();
    let (y, x): (usize, usize) = (data.dims[0], data.dims[1]);
    let terminal_dict: HashMap<String, usize> = data.indices;
    let terminals: Vec<usize> = terminal_dict.values().cloned().collect();
    let pathlists: HashMap<String, Vec<Vec<[String; 2]>>> = data.pathlists;

    let pool: ThreadPool = ThreadPool::new(args.cores);

    let results: HashMap<String, Result> = HashMap::new();
    let results_permuted: HashMap<String, ResultPerm> = HashMap::new();

    let output: Arc<Mutex<HashMap<String, Result>>> = Arc::new(Mutex::new(results));
    let output_permuted: Arc<Mutex<HashMap<String, ResultPerm>>> =
        Arc::new(Mutex::new(results_permuted));

    for i in (10..91).rev() {
        let output: Arc<Mutex<HashMap<String, Result>>> = output.clone();
        let output_permuted: Arc<Mutex<HashMap<String, ResultPerm>>> = output_permuted.clone();
        let terminals: Vec<usize> = terminals.clone();

        let key: String = format!("N{i}");
        let mut t_pairs: Vec<Vec<[usize; 2]>> = nodes_to_indices(&terminal_dict, &pathlists[&key]);

        pool.execute(move || {
            println!("{key}");
            let mut router: Routing = Routing::new([Z, y, x], terminals);

            let result: Result =
                handle_npathlist(&mut router, &mut t_pairs, args.save_paths, & mut None);
            let result_permuted: ResultPerm = 
                handle_perms(&mut router, &mut t_pairs, args.perms, args.save_paths, i);

            let mut out = output.lock().unwrap();
            out.insert(key.clone(), result);
            let mut out_permuted = output_permuted.lock().unwrap();
            out_permuted.insert(key, result_permuted);
        });
    }
    pool.join();

    write_to_json(output, args.results_path);
    if args.perms > 0 {
        write_to_json(output_permuted, args.results_perm_path);
    }
}
