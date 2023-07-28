// Standard library imports
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::vec;

// Consider using precalc h for grid/mesh (all indices)
pub struct Routing {
    pub dims: [usize; 3],
    terminals: Vec<usize>,
    coordinates: Vec<[usize; 3]>,
    g: Vec<i32>,
    accessible: Vec<bool>,
    predecessor: Vec<usize>,
}

impl Routing {
    pub fn new(dims: [usize; 3], terminals: Vec<usize>) -> Self {
        // let (z, y, x): (usize, usize, usize) = (dims[0], dims[1], dims[2]);
        let [z, y, x] = dims;
        let mut coordinates: Vec<[usize; 3]> = Vec::new();
        for i in 0..z * y * x + 1 {
            coordinates.push([i / (y * x), i / x % y, i % x]);
        }
        Self {
            dims,
            terminals,
            coordinates,
            g: vec![i32::MAX; z * y * x],
            accessible: vec![true; z * y * x],
            predecessor: vec![usize::MAX; z * y * x],
        }
    }

    fn reset_arrays(&mut self) {
        let [z, y, x] = self.dims;
        // let (z, y, x) = (self.dims[0], self.dims[1], self.dims[2]);
        self.g = vec![i32::MAX; z * y * x];
        self.predecessor = vec![usize::MAX; z * y * x];
    }

    fn reset_accessible(&mut self) {
        let [z, y, x] = self.dims;
        // let (z, y, x) = (self.dims[0], self.dims[1], self.dims[2]);
        self.accessible = vec![true; z * y * x];
    }

    // fn index_unravel3d(&self, i: usize) -> [usize; 3] {
    //     let (ylim, xlim) = (self.dims[1], self.dims[2]);
    //     // [i / (ylim * xlim), i / xlim % ylim, i % xlim]
    //     [i % xlim, i / xlim % ylim, i / (ylim * xlim)]
    // }

    fn get_successors(&self, q: usize, goal: usize) -> Vec<usize> {
        let [zlim, ylim, xlim] = self.dims;
        // let (zlim, ylim, xlim) = (self.dims[0], self.dims[1], self.dims[2]);
        let z_offset: usize = ylim * xlim; // Stays constant within run, see if better performance if lookup in struct
        let [z, y, x]: [usize; 3] = self.coordinates[q];
        // let (z, y, x) = (coordinates[0], coordinates[1], coordinates[2]);

        let mut successors: Vec<usize> = Vec::with_capacity(6);
        if z > 0 && (self.accessible[q - z_offset] || q == goal + z_offset) {
            successors.push(q - z_offset);
        }
        if y > 0 && (self.accessible[q - xlim] || q == goal + xlim) {
            successors.push(q - xlim);
        }
        if x > 0 && (self.accessible[q - 1] || q == goal + 1) {
            successors.push(q - 1);
        }
        if x < xlim - 1 && (self.accessible[q + 1] || q + 1 == goal) {
            successors.push(q + 1);
        }
        if y < ylim - 1 && (self.accessible[q + xlim] || q + xlim == goal) {
            successors.push(q + xlim);
        }
        if z < zlim - 1 && (self.accessible[q + z_offset] || q + z_offset == goal) {
            successors.push(q + z_offset);
        }
        successors
    }

    fn absolute_distance(&self, current: usize, (gz, gy, gx): (usize, usize, usize)) -> i32 { // goal tuple can be turned into goal array or lookup as well
        // let current_coordinates: [usize; 3] = self.coordinates[current];
        let [qz, qy, qx] =  self.coordinates[current];
        // let (qz, qy, qx) = (
        //     current_coordinates[0],
        //     current_coordinates[1],
        //     current_coordinates[2],
        // );
        (gz.abs_diff(qz) + gy.abs_diff(qy) + gx.abs_diff(qx)) as i32
    }

    pub fn a_star(&mut self, start: usize, goal: usize) -> u32 {
        let goal_coordinates: [usize; 3] = self.coordinates[goal];
        let goal_tuple = (
            goal_coordinates[0],
            goal_coordinates[1],
            goal_coordinates[2],
        );

        if self.get_successors(goal, goal).len() == 0 {
            if self.absolute_distance(start, goal_tuple) == 1
            {
                self.predecessor[goal] = start;
                return 1
            }
            return 0;
        }

        self.g[start] = 0;
        // let mut closedset: HashSet<usize> = HashSet::new();
        let mut openheap: BinaryHeap<(Reverse<i32>, Reverse<i32>, usize)> =
            BinaryHeap::new();
        openheap.push((Reverse(0), Reverse(0), start));
        // closedset.insert(start);
        let mut closedvec: Vec<bool> = vec![false; self.dims.into_iter().product()];
        closedvec[start] = true;

        while openheap.len() > 0 {
            let q: usize = openheap.pop().unwrap().2;

            let successors: Vec<usize> = self.get_successors(q, goal);
            let gc: i32 = self.g[q];
            for successor in successors {
                if successor == goal {
                    self.predecessor[goal] = q;
                    return 1;
                }
                // if !closedset.contains(&successor) {
                if !closedvec[successor] {
                    // seems only possible when "h(x) <= d(x,y) + h(y)" holds
                    // closedset.insert(successor);
                    closedvec[successor] = true;
                    self.g[successor] = gc + 1;
                    let f: i32 = self.g[successor] + self.absolute_distance(successor, goal_tuple);
                    self.predecessor[successor] = q;
                    openheap.push((Reverse(f), Reverse(gc + 1), successor));
                }
            }
        }
        return 0;
    }

    pub fn no_path(&mut self, start: usize, goal: usize) -> Vec<usize> {
        let path: Vec<usize> = Vec::new();
        if self.predecessor[goal] == usize::MAX {
            return path;
        }
        let mut current: usize = goal;
        while current != start {
            self.accessible[current] = false;
            current = self.predecessor[current];
        }
        path
    }

    pub fn get_path(&mut self, start: usize, goal: usize) -> Vec<usize> {
        let mut path: Vec<usize> = Vec::new();
        if self.predecessor[goal] == usize::MAX {
            return path;
        }
        let mut current: usize = goal;
        while current != start {
            self.accessible[current] = false;
            path.push(current);
            current = self.predecessor[current];
        }
        path.push(current);
        path
    }

    fn set_inaccesible(&mut self) {
        for terminal in &self.terminals {
            self.accessible[*terminal] = false;
        }
    }

    pub fn run(
        &mut self,
        terminal_pairs: Vec<[usize; 2]>,
        collect: bool,
    ) -> (u32, Vec<Vec<usize>>) {
        let mut paths: Vec<Vec<usize>> = Vec::new();
        let mut npaths: u32 = 0;

        let path_func: &dyn Fn(&mut Routing, usize, usize) -> Vec<usize> = if collect {
            &Routing::get_path
        } else {
            &Routing::no_path
        };

        self.reset_accessible();
        self.set_inaccesible();
        for terminal_pair in terminal_pairs {
            let (start, goal) = (terminal_pair[0], terminal_pair[1]);
            npaths += self.a_star(start, goal);

            // If only checking for fully routable use commented out code below to improve performance (a lot)!
            // let success: u32 = self.a_star(start, goal);
            // if success == 0{break;}
            // npaths += 1;

            let path: Vec<usize> = path_func(self, start, goal);
            if path.len() > 0 {
                paths.push(path);
            }
            self.reset_arrays();
        }
        (npaths, paths)
    }
}
