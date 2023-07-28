// Standard library imports
use std::cmp::Reverse;
// use std::collections::{BinaryHeap, HashSet};
use std::collections::BinaryHeap;
use std::vec;
use core::iter::zip;

// Consider using precalc h for grid/mesh (all indices)


// Easier to work with shape if it ranges from x to z ... instead of reverse?
fn i_to_nd(i: usize, shape: &Vec<usize>, n: usize) -> Vec<usize>
{
    let mut result: Vec<usize> = Vec::new();
    let mut base: usize = n;
    for limit in shape {
        base /= limit;
        result.push(i / base % limit);
    }
    result
}


pub struct Router {
    n: usize,
    pub shape: Vec<usize>,
    terminals: Vec<usize>,
    coordinates: Vec<Vec<usize>>,
    g: Vec<i32>,
    accessible: Vec<bool>,
    predecessor: Vec<usize>,
}

impl Router {
    pub fn new(shape: &Vec<usize>, terminals: Vec<usize>) -> Self {
        let n: usize = shape.into_iter().product();
        let mut coordinates: Vec<Vec<usize>> = Vec::new();
        for i in 0..n + 1 {
            coordinates.push(i_to_nd(i, shape, n));
        }
        let shape = shape.to_vec();
        Self {
            n,
            shape,
            terminals,
            coordinates,
            g: vec![i32::MAX; n],
            accessible: vec![true; n],
            predecessor: vec![usize::MAX; n],
        }
    }

    fn reset_arrays(&mut self) {
        self.g = vec![i32::MAX; self.n];
        self.predecessor = vec![usize::MAX; self.n];
    }

    fn reset_accessible(&mut self) {
        self.accessible = vec![true; self.n];
    }

    // fn index_unravel3d(&self, i: usize) -> [usize; 3] {
    //     let (ylim, xlim) = (self.dims[1], self.dims[2]);
    //     // [i / (ylim * xlim), i / xlim % ylim, i % xlim]
    //     [i % xlim, i / xlim % ylim, i / (ylim * xlim)]
    // }

    #[inline]
    fn get_successors(&self, q: usize, goal: usize) -> Vec<usize> {
        let mut successors: Vec<usize> = Vec::with_capacity(8);
        let mut offset = self.n;

        // let coordinates: &Vec<usize> = &self.coordinates[q];
        for (coordinate, limit) in zip(&self.coordinates[q], &self.shape) {
            offset /= limit;
            if *coordinate > 0 && (self.accessible[q - offset] || q == goal + offset) {
                successors.push(q - offset)
            }
            if *coordinate < limit - 1 && (self.accessible[q + offset] || q + offset == goal) {
                successors.push(q + offset)
            }
            // offset *= coordinate;
        }
        successors


        // let [zlim, ylim, xlim] = self.shape;
        // let z_offset: usize = ylim * xlim; // Stays constant within run, see if better performance if lookup in struct
        // let [z, y, x]: [usize; 3] = self.coordinates[q];

        // let mut successors: Vec<usize> = Vec::with_capacity(6);
        // if z > 0 && (self.accessible[q - z_offset] || q == goal + z_offset) {
        //     successors.push(q - z_offset);
        // }
        // if y > 0 && (self.accessible[q - xlim] || q == goal + xlim) {
        //     successors.push(q - xlim);
        // }
        // if x > 0 && (self.accessible[q - 1] || q == goal + 1) {
        //     successors.push(q - 1);
        // }
        // if x < xlim - 1 && (self.accessible[q + 1] || q + 1 == goal) {
        //     successors.push(q + 1);
        // }
        // if y < ylim - 1 && (self.accessible[q + xlim] || q + xlim == goal) {
        //     successors.push(q + xlim);
        // }
        // if z < zlim - 1 && (self.accessible[q + z_offset] || q + z_offset == goal) {
        //     successors.push(q + z_offset);
        // }
        // successors
    }

    #[inline]
    fn absolute_distance(&self, current: usize, goal_coordinates: &Vec<usize>) -> i32 { // goal tuple can be turned into goal array or lookup as well
        let mut total = 0;
        let current_coordinates: &Vec<usize> = &self.coordinates[current];
        for (cc, gc) in zip(current_coordinates, goal_coordinates)
        {
            total += cc.abs_diff(*gc);
        }
        total as i32
    }

    pub fn a_star(&mut self, start: usize, goal: usize) -> u32 {
        let goal_coordinates: &Vec<usize> = &self.coordinates[goal];

        if self.get_successors(goal, goal).len() == 0 {
            if self.absolute_distance(start, goal_coordinates) == 1
            {
                self.predecessor[goal] = start;
                return 1
            }
            return 0;
        }

        self.g[start] = 0;
        // let mut closedset: HashSet<usize> = HashSet::new();
        let mut openheap: BinaryHeap<(Reverse<i32>, Reverse<i32>, usize)> = 
            BinaryHeap::with_capacity(1000);
        openheap.push((Reverse(0), Reverse(0), start));
        // closedset.insert(start);


        let mut closed_vec: Vec<bool> = vec![false; self.n];
        closed_vec[start] = true;

        while openheap.len() > 0 {
            let q: usize = openheap.pop().unwrap().2;
            // let successors: Vec<usize> = self.get_successors(q, goal);
            let gn: i32 = self.g[q] + 1;
            for successor in self.get_successors(q, goal) {
                if successor == goal {
                    self.predecessor[goal] = q;
                    return 1;
                }
                // if !closedset.contains(&successor) {
                    // closedset.insert(successor);
                if !closed_vec[successor] {
                    closed_vec[successor] = true;
                    self.g[successor] = gn;
                    let f: i32 = gn + self.absolute_distance(successor, goal_coordinates);
                    self.predecessor[successor] = q;
                    openheap.push((Reverse(f), Reverse(gn), successor));
                }
            }
        }
        return 0;
    }

    #[inline]
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

    #[inline]
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
        let mut paths: Vec<Vec<usize>> = Vec::with_capacity(terminal_pairs.len());
        let mut npaths: u32 = 0;

        let path_func: &dyn Fn(&mut Router, usize, usize) -> Vec<usize> = if collect {
            &Router::get_path
        } else {
            &Router::no_path
        };

        self.reset_accessible();
        self.set_inaccesible();
        for terminal_pair in terminal_pairs {
            let (start, goal) = (terminal_pair[0], terminal_pair[1]);
            // npaths += self.a_star(start, goal);

            // If only checking for fully routable use commented out code below to improve performance (a lot)!
            let success: u32 = self.a_star(start, goal);
            if success == 0{break;}
            npaths += 1;

            let path: Vec<usize> = path_func(self, start, goal);
            if path.len() > 0 {
                paths.push(path);
            }
            self.reset_arrays();
        }
        (npaths, paths)
    }
}
