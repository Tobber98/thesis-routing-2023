// TODO: Move dims out of the result structs and add directly into hashmap, removes unnecessary duplicates. (More difficult than anticipated...) loading into pandas does not work
// TODO: Add seeding input !! (Added but not working properly)

// Standard imports
use std::collections::{BTreeMap, HashMap};
use std::env::Args;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{env, fs};

// External imports
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use threadpool::ThreadPool;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use indicatif::{ProgressBar, ProgressStyle};

// Local imports
mod router;
use router::Router;

mod json_serializer;
use json_serializer::PrettyFormatter2;

// Constants
const DEFAULT_SAVE: bool = false;
const DEFAULT_PERMUTATIONS: u32 = 200;
const DEFAULT_CORES: usize = 6;

fn check_empty<T>(v: &Vec<T>) -> bool {
    return v.is_empty();
}

#[derive(Serialize, Debug)]
struct Result {
    shape: Vec<usize>,
    length: usize,
    nroutable: Vec<Vec<u32>>,
    #[serde(skip_serializing_if = "check_empty")]
    paths: Vec<Vec<Vec<Vec<usize>>>>,
}

impl Result {
    pub fn new(shape: Vec<usize>, length: usize) -> Self {
        Self {
            shape,
            length,
            nroutable: Vec::new(),
            paths: Vec::new(),
        }
    }
}

#[derive(Deserialize, Debug)]
struct Data {
    shapes: HashMap<String, Vec<Vec<usize>>>,
    netlists: BTreeMap<String, Vec<Vec<[usize; 2]>>>,
}

impl Data {
    pub fn map_netlists(&self, terminals: &Vec<usize>, nlength: &String) -> Vec<Vec<[usize; 2]>> {
        let t_pairs_vec: &Vec<Vec<[usize; 2]>> = self.netlists.get(nlength).unwrap();
        let mut index_netlists: Vec<Vec<[usize; 2]>> = Vec::with_capacity(t_pairs_vec.len());
        for pairs in t_pairs_vec {
            let mut index_netlist: Vec<[usize; 2]> = Vec::with_capacity(pairs.len());
            for pair in pairs {
                index_netlist.push([terminals[pair[0]], terminals[pair[1]]])
            }
            index_netlists.push(index_netlist);
        }
        index_netlists
    }

    pub fn str_to_shape(&self, shape_str: &String) -> Vec<usize> {
        let stripped: &str = &shape_str[1..shape_str.len() - 1];
        let shape: Vec<usize> = stripped.split(", ").map(|x| x.parse().unwrap()).collect();
        shape
    }
}

fn read_json<P: AsRef<Path>>(path: P) -> Value {
    let file: String = fs::read_to_string(path).expect("Unable to read file");
    serde_json::from_str(&file).expect("Error occurred while parsing JSON")
}

// TODO: Create router here?
fn handle_netlists(
    router: &mut Router,
    netlists: &mut Vec<Vec<[usize; 2]>>,
    length: usize,
    collect: bool,
    seed: u64,
    permutations: u32,
) -> Result {
    let mut result: Result = Result::new(router.shape.clone(), length);
    let rng: &mut ChaCha8Rng = &mut ChaCha8Rng::seed_from_u64(seed);
    for netlist in netlists {
        let mut routable_vec: Vec<u32> = Vec::new();
        let mut paths_vec: Vec<Vec<Vec<usize>>> = Vec::new();

        let (nroutable, paths): (u32, Vec<Vec<usize>>) = router.run(netlist.to_vec(), collect);
        routable_vec.push(nroutable);
        paths_vec.push(paths);
        for _ in 1..permutations {
            netlist.shuffle(rng);
            let (nroutable, paths): (u32, Vec<Vec<usize>>) = router.run(netlist.to_vec(), collect);
            routable_vec.push(nroutable);
            paths_vec.push(paths);
        }
        result.nroutable.push(routable_vec);
        if collect {
            result.paths.push(paths_vec);
        }
    }
    result
}

#[derive(Debug)]
struct FormattedArgs {
    save_paths: bool,
    perms: u32,
    data_path: PathBuf,
    prefix: PathBuf,
    results_path: PathBuf,
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

        let mut save_paths: bool = DEFAULT_SAVE;
        let mut perms: u32 = DEFAULT_PERMUTATIONS;
        let mut cores: usize = DEFAULT_CORES;

        let mut extension: String = String::new();

        while let Some(current) = args.next() {
            match &current[..] {
                "--cores" | "-c" => cores = args.next().unwrap().parse().unwrap(),
                "--save" | "-s" => save_paths = args.next().unwrap() == "1",
                "--permutations" | "-p" => perms = args.next().unwrap().parse().unwrap(),
                "--extension" | "-e" => {
                    extension = args.next().unwrap().clone();
                }
                _ => continue,
            }
        }

        if save_paths {
            results_path.push(format!("results_ext{}.json", extension));
        } else {
            results_path.push(format!("results{}.json", extension));
        }

        Self {
            save_paths,
            perms,
            data_path,
            prefix,
            results_path,
            cores,
        }
    }
}

// Fix naming
fn write_to_json(output: HashMap<String, Vec<HashMap<String, Result>>>, path: PathBuf) {
    let json_result: Value = json!(output);
    let formatter: PrettyFormatter2 = PrettyFormatter2::new();
    let buf: Vec<u8> = Vec::new();
    let mut serializer: serde_json::Serializer<Vec<u8>, PrettyFormatter2> =
        serde_json::Serializer::with_formatter(buf, formatter);
    json_result.serialize(&mut serializer).unwrap();
    let json_result_string = String::from_utf8(serializer.into_inner()).unwrap();
    fs::write(path, &json_result_string).unwrap();
}

fn main() {
    let args: FormattedArgs = FormattedArgs::new(&mut env::args());
    if !Path::exists(&args.prefix) {
        fs::create_dir_all(&args.prefix).unwrap();
    }

    let pool: ThreadPool = ThreadPool::new(args.cores);
    let data: Data = serde_json::from_value(read_json(args.data_path)).unwrap();
    let mut results: HashMap<String, Vec<HashMap<String, Result>>> = HashMap::new();
    for shape_str in data.shapes.keys() {
        println!("_______________\n{shape_str}");
        let mut results_shape: Vec<HashMap<String, Result>> = Vec::new();
        for layout in 0..data.shapes[shape_str].len() {
            let terminals: &Vec<usize> = &data.shapes.get(shape_str).unwrap()[layout];
            let output: Arc<Mutex<HashMap<String, Result>>> = Arc::new(Mutex::new(HashMap::new()));

            let bar: ProgressBar = ProgressBar::new(data.netlists.len().try_into().unwrap());
            bar.set_style(ProgressStyle::with_template("[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>5}/{len:5}").unwrap().progress_chars("#>-"));
            bar.tick();
            let bar: Arc<Mutex<ProgressBar>> = Arc::new(Mutex::new(bar));
            for nlength in data.netlists.keys().rev() {
                let nlength = nlength.clone();
                let length: usize = nlength.strip_prefix("N").unwrap().parse().unwrap();
                let mut netlists: Vec<Vec<[usize; 2]>> = data.map_netlists(terminals, &nlength);
                let shape: Vec<usize> = data.str_to_shape(shape_str);
                let terminals: Vec<usize> = terminals.clone();

                let output = output.clone();
                let bar = bar.clone();
                pool.execute(move || {
                    // print!("  | {nlength}");
                    let mut router: Router = Router::new(&shape, terminals);
                    let result: Result = handle_netlists(
                        &mut router,
                        &mut netlists,
                        length,
                        args.save_paths,
                        9,
                        args.perms,
                    );
                    let mut locked_output = output.lock().unwrap();
                    locked_output.insert(nlength, result);

                    bar.lock().unwrap().inc(1);
                });
            }
            pool.join();
            bar.lock().unwrap().finish();
            let out: HashMap<String, Result> =
                Arc::try_unwrap(output).unwrap().into_inner().unwrap();
            results_shape.push(out);
        }
        results.insert(shape_str.to_string(), results_shape);
    }
    write_to_json(results, args.results_path);
}
