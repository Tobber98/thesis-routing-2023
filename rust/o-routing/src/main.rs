// TODO: Move dims out of the result structs and add directly into hashmap, removes unnecessary duplicates. (More difficult than anticipated...) loading into pandas does not work
// TODO: Add seeding input !! (Added but not working properly)

// Standard imports
use std::collections::{BTreeMap, HashMap};
use std::env::Args;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{env, fs};

use rand::seq::SliceRandom;
use rand::thread_rng;
// External imports
// use rand::seq::SliceRandom;
// use rand::SeedableRng;
// use rand_chacha::ChaCha8Rng;
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
// const DEFAULT_PERMUTATIONS: u32 = 200;
const DEFAULT_CORES: usize = 11;

#[derive(Serialize, Debug)]
struct OrderResult {
    shape: [usize; 3],
    length: usize,
    min_distances: Vec<u32>,

    nroutable: HashMap<String, Vec<u32>>,
    completed: HashMap<String, Vec<Vec<bool>>>,
    path_length: HashMap<String, Vec<usize>>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    paths: Vec<Vec<Vec<Vec<usize>>>>,
}

impl OrderResult {
    pub fn new(shape: [usize; 3], length: usize) -> Self {
        Self {
            shape,
            length,
            min_distances:  Vec::new(),
            
            nroutable:      HashMap::new(),
            completed:      HashMap::new(),
            path_length:    HashMap::new(),

            paths:          Vec::new(),
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

fn update_order_result(
    name: &str,
    result: &mut OrderResult,
    npaths: u32,
    path_length: usize,
    completed: Vec<bool>
) { 
    result.nroutable.entry(name.to_string()).or_insert_with(Vec::new).push(npaths);
    result.path_length.entry(name.to_string()).or_insert_with(Vec::new).push(path_length);
    result.completed.entry(name.to_string()).or_insert_with(Vec::new).push(completed);
}


fn run_orders(
    router: &mut Router,
    netlists: &mut Vec<Vec<[usize; 2]>>,
    length: usize,
    collect: bool
) -> OrderResult {
    let mut result: OrderResult = OrderResult::new(router.shape.clone(), length);

    let rng = &mut thread_rng();
    for netlist in netlists {
        // netlist.shuffle(rng);
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("random", &mut result, npaths, path_length, completed);

        netlist.sort_by_key(|k| router.min_distance(k));
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("shortest", &mut result, npaths, path_length, completed);

        netlist.reverse();
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("longest", &mut result, npaths, path_length, completed);

        netlist.sort_by_key(|k| router.min_z(k));
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("min_z", &mut result, npaths, path_length, completed);

        netlist.reverse();
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("max_z", &mut result, npaths, path_length, completed);

        netlist.sort_by_key(|k| router.min_y(k));
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("min_y", &mut result, npaths, path_length, completed);

        netlist.reverse();
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("max_y", &mut result, npaths, path_length, completed);

        netlist.sort_by_key(|k| router.min_x(k));
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("min_x", &mut result, npaths, path_length, completed);

        netlist.reverse();
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("max_x", &mut result, npaths, path_length, completed);

        netlist.sort_by_key(|k| router.min_volume(k));
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("min_volume", &mut result, npaths, path_length, completed);

        router.set_inaccesible();
        netlist.shuffle(rng);
        netlist.sort_by_key(|k| router.min_density(k));
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("min_density", &mut result, npaths, path_length, completed);

        router.set_inaccesible();
        netlist.shuffle(rng);
        netlist.sort_by_key(|k| router.max_density(k));
        let (npaths, path_length, completed) = router.run(netlist.to_vec(), collect);
        update_order_result("max_density", &mut result, npaths, path_length, completed);


        // Adaptive routing
        netlist.shuffle(rng);
        let (npaths, path_length, completed) = router.run_adaptive_changer(netlist.to_vec(), collect, false);
        update_order_result("adaptive_max_density_random", &mut result, npaths, path_length, completed);
        
        netlist.shuffle(rng);
        let (npaths, path_length, completed) = router.run_adaptive_changer(netlist.to_vec(), collect, true);
        update_order_result("adaptive_min_density_random", &mut result, npaths, path_length, completed);


        // Best adaptive combination test
        netlist.sort_by_key(|k| router.min_distance(k));
        let (npaths, path_length, completed) = router.run_adaptive_changer(netlist.to_vec(), collect, false);
        update_order_result("adaptive_max_density_shortest", &mut result, npaths, path_length, completed);

        netlist.sort_by_key(|k| router.min_distance(k));
        netlist.reverse();
        let (npaths, path_length, completed) = router.run_adaptive_changer(netlist.to_vec(), collect, false);
        update_order_result("adaptive_max_density_longest", &mut result, npaths, path_length, completed);

        netlist.sort_by_key(|k| router.min_volume(k));
        let (npaths, path_length, completed) = router.run_adaptive_changer(netlist.to_vec(), collect, false);
        update_order_result("adaptive_max_density_min_volume", &mut result, npaths, path_length, completed);
 

        let distance_list: Vec<u32> = netlist.iter().map(|x| router.min_distance(x)).collect();
        result.min_distances = distance_list;
    }
    result
}



// TODO: Create router here?
// 


#[derive(Debug)]
struct FormattedArgs {
    _save_paths: bool,
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
        let mut cores: usize = DEFAULT_CORES;

        let mut extension: String = String::new();

        while let Some(current) = args.next() {
            match &current[..] {
                "--cores" | "-c" => cores = args.next().unwrap().parse().unwrap(),
                "--save" | "-s" => save_paths = args.next().unwrap() == "1",
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
            _save_paths: save_paths,
            data_path,
            prefix,
            results_path,
            cores,
        }
    }
}

// Fix naming
fn write_to_json(output: HashMap<String, Vec<HashMap<String, OrderResult>>>, path: PathBuf) {
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
    let mut results: HashMap<String, Vec<HashMap<String, OrderResult>>> = HashMap::new();
    for shape_str in data.shapes.keys() {
        println!("_______________\n{shape_str}");
        let mut results_shape: Vec<HashMap<String, OrderResult>> = Vec::new();
        for layout in 0..data.shapes[shape_str].len() {
            let terminals: &Vec<usize> = &data.shapes.get(shape_str).unwrap()[layout];
            let output: Arc<Mutex<HashMap<String, OrderResult>>> = Arc::new(Mutex::new(HashMap::new()));

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
                    let shape2 = [shape[0], shape[1], shape[2]];
                    let mut router: Router = Router::new(shape2, terminals);
                    let result: OrderResult = run_orders(&mut router, &mut netlists, length, true);
                    let mut locked_output = output.lock().unwrap();
                    locked_output.insert(nlength, result);

                    bar.lock().unwrap().inc(1);
                });
            }
            pool.join();
            bar.lock().unwrap().finish();
            let out: HashMap<String, OrderResult> =
                Arc::try_unwrap(output).unwrap().into_inner().unwrap();
            results_shape.push(out);

        }
        results.insert(shape_str.to_string(), results_shape);
    }
    write_to_json(results, args.results_path);
}
