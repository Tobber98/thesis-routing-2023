// Standard library imports
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::vec;

// Consider using precalc h for grid/mesh (all indices)
pub struct Router {
    pub shape: [usize; 3],
    terminals: Vec<usize>,
    pub coordinates: Vec<[usize; 3]>,
    g: Vec<u32>,
    accessible: Vec<bool>,
    predecessor: Vec<usize>,
    // octants: [usize; 3]
}

impl Router {
    pub fn new(shape: [usize; 3], terminals: Vec<usize>) -> Self {
        let [z, y, x] = shape;
        let mut coordinates: Vec<[usize; 3]> = Vec::new();
        for i in 0..z * y * x + 1 {
            coordinates.push([i / (y * x), i / x % y, i % x]);
        }
        // let mut octants: [usize; 3] = [0, 0, 0];
        // for i in 0..shape.len() {
        //     octants[i] = shape[i] / 2;
        // }
        Self {
            shape,
            terminals,
            coordinates,
            g: vec![u32::MAX; z * y * x],
            accessible: vec![true; z * y * x],
            predecessor: vec![usize::MAX; z * y * x],
            // octants
        }
    }

    fn reset_arrays(&mut self) {
        let [z, y, x] = self.shape;
        self.g = vec![u32::MAX; z * y * x];
        self.predecessor = vec![usize::MAX; z * y * x];
    }

    fn reset_accessible(&mut self) {
        let [z, y, x] = self.shape;
        self.accessible = vec![true; z * y * x];
    }

    // fn index_unravel3d(&self, i: usize) -> [usize; 3] {
    //     let (ylim, xlim) = (self.dims[1], self.dims[2]);
    //     // [i / (ylim * xlim), i / xlim % ylim, i % xlim]
    //     [i % xlim, i / xlim % ylim, i / (ylim * xlim)]
    // }

    fn get_successors(&self, q: usize, goal: usize) -> Vec<usize> {
        let [zlim, ylim, xlim] = self.shape;
        let z_offset: usize = ylim * xlim; // Stays constant within run, see if better performance if lookup in struct
        let [z, y, x]: [usize; 3] = self.coordinates[q];

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

    fn absolute_distance(&self, current: usize, [gz, gy, gx]: [usize; 3]) -> u32 { // goal tuple can be turned into goal array or lookup as well
        let [qz, qy, qx] =  self.coordinates[current];
        (gz.abs_diff(qz) + gy.abs_diff(qy) + gx.abs_diff(qx)) as u32
    }

    pub fn a_star(&mut self, start: usize, goal: usize) -> u32 {
        let goal_coordinates: [usize; 3] = self.coordinates[goal];

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
        let mut openheap: BinaryHeap<(Reverse<u32>, Reverse<u32>, usize)> =
            BinaryHeap::new();
        openheap.push((Reverse(0), Reverse(0), start));
        // closedset.insert(start);
        let mut closedvec: Vec<bool> = vec![false; self.shape.into_iter().product()];
        closedvec[start] = true;

        while openheap.len() > 0 {
            let q: usize = openheap.pop().unwrap().2;

            let successors: Vec<usize> = self.get_successors(q, goal);
            let gc: u32 = self.g[q];
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
                    let f: u32 = self.g[successor] + self.absolute_distance(successor, goal_coordinates);
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

    pub fn get_path_length(&mut self, start: usize, goal: usize) -> Vec<usize> {
        let mut path: Vec<usize> = Vec::new();
        if self.predecessor[goal] == usize::MAX {
            path.push(0);
            return path;
        }
        let mut current: usize = goal;
        let mut length: usize = 0;
        while current != start {
            self.accessible[current] = false;
            current = self.predecessor[current];
            length += 1;
        }
        path.push(length);
        path
    }

    // pub fn get_path(&mut self, start: usize, goal: usize) -> Vec<usize> {
    //     let mut path: Vec<usize> = Vec::new();
    //     if self.predecessor[goal] == usize::MAX {
    //         return path;
    //     }
    //     let mut current: usize = goal;
    //     while current != start {
    //         self.accessible[current] = false;
    //         path.push(current);
    //         current = self.predecessor[current];
    //     }
    //     path.push(current);
    //     path
    // }

    pub fn set_inaccesible(&mut self) {
        for terminal in &self.terminals {
            self.accessible[*terminal] = false;
        }
    }

    pub fn run(
        &mut self,
        terminal_pairs: Vec<[usize; 2]>,
        collect: bool,
    ) -> (u32, usize, Vec<bool>) {
        // let mut paths: Vec<Vec<usize>> = Vec::new();
        let mut npaths: u32 = 0;
        let mut path_length: usize = 0;
        let mut completed: Vec<bool> = Vec::new();

        let path_func: &dyn Fn(&mut Router, usize, usize) -> Vec<usize> = if collect {
            &Router::get_path_length
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
            if success == 0 {
                completed.push(false);
                // break;
            }
            else {
                completed.push(true);
                npaths += 1;
            }

            let path: Vec<usize> = path_func(self, start, goal);
            path_length += path[0];
            // if path.len() > 0 {
            //     paths.push(path);
            // }
            self.reset_arrays();
        }
        (npaths, path_length, completed)
    }

    pub fn max_density(&self, pair: &[usize; 2]) -> i32
    {
        (self.get_successors(pair[0], pair[1]).len() + self.get_successors(pair[1], pair[0]).len()) as i32
    }

    pub fn min_density(&self, pair: &[usize; 2]) -> i32
    {
        -((self.get_successors(pair[0], pair[1]).len() + self.get_successors(pair[1], pair[0]).len()) as i32)
    }

    pub fn run_adaptive_changer(
        &mut self,
        terminal_pairs: Vec<[usize; 2]>,
        collect: bool,
        reverse: bool,
    ) -> (u32, usize, Vec<bool>) {
        // let mut paths: Vec<Vec<usize>> = Vec::new();
        let mut npaths: u32 = 0;
        let mut path_length: usize = 0;
        let mut completed: Vec<bool> = Vec::new();

        let path_func: &dyn Fn(&mut Router, usize, usize) -> Vec<usize> = if collect {
            &Router::get_path_length
        } else {
            &Router::no_path
        };

        self.reset_accessible();
        self.set_inaccesible();
        let mut pairs = terminal_pairs.clone();
        while !pairs.is_empty() {
            pairs.sort_by_key(|x| self.max_density(x));
            if !reverse {
                pairs.reverse();
            }
            let [start, goal] = pairs.pop().unwrap();

        // }
    
        // for terminal_pair in terminal_pairs {
        //     let [start, goal] = terminal_pair;
            // npaths += self.a_star(start, goal);

            // If only checking for fully routable use commented out code below to improve performance (a lot)!
            let success: u32 = self.a_star(start, goal);
            if success == 0 {
                // path_length = 0;
                completed.push(false);
                // break;
            }
            else {
                completed.push(true);
                npaths += 1;
            }

            let path: Vec<usize> = path_func(self, start, goal);
            path_length += path[0];
            // if path.len() > 0 {
            //     paths.push(path);
            // }
            self.reset_arrays();
        }
        (npaths, path_length, completed)
    }




    // fn l_to_r(&self, terminals: &[usize; 2]) -> [usize; 2] {
    //     let [ylim, xlim] = [self.shape[1], self.shape[2]];
    //     let [i, j] = terminals;
    //     if i / (ylim * xlim) + i / xlim % ylim + i % xlim > j / (ylim * xlim) + j / xlim % ylim + j % xlim {
    //         return [*j, *i]
    //     }
    //     *terminals
    // }

    // pub fn run_lr(
    //     &mut self,
    //     terminal_pairs: Vec<[usize; 2]>,
    //     collect: bool,
    // ) -> (u32, usize, Vec<bool>) {
    //     // let mut paths: Vec<Vec<usize>> = Vec::new();
    //     let mut npaths: u32 = 0;
    //     let mut path_length: usize = 0;
    //     let mut completed: Vec<bool> = Vec::new();

    //     let path_func: &dyn Fn(&mut Router, usize, usize) -> Vec<usize> = if collect {
    //         &Router::get_path_length
    //     } else {
    //         &Router::no_path
    //     };

    //     self.reset_accessible();
    //     self.set_inaccesible();
    //     for terminal_pair in terminal_pairs {
    //         let [start, goal] = self.l_to_r(&terminal_pair);
    //         // let (start, goal) = (terminal_pair[0], terminal_pair[1]);
    //         // npaths += self.a_star(start, goal);

    //         // If only checking for fully routable use commented out code below to improve performance (a lot)!
    //         let success: u32 = self.a_star(start, goal);
    //         if success == 0 {
    //             completed.push(false);
    //             // break;
    //         }
    //         else {
    //             completed.push(true);
    //             npaths += 1;
    //         }

    //         let path: Vec<usize> = path_func(self, start, goal);
    //         path_length += path[0];
    //         // if path.len() > 0 {
    //         //     paths.push(path);
    //         // }
    //         self.reset_arrays();
    //     }
    //     (npaths, path_length, completed)
    // }




    pub fn min_distance(&self, terminals: &[usize; 2]) -> u32{
        let [start, goal] = terminals;
        // let start_coordinate = self.coordinates[start];
        let goal_coordinate = self.coordinates[*goal];
        
        self.absolute_distance(*start, goal_coordinate)
    }


    pub fn min_x(&self, terminals: &[usize; 2]) -> i32 {
        let [_zlim, ylim, xlim] = self.shape;
        (terminals[0] / (ylim * xlim) + terminals[1] / (ylim * xlim)) as i32
    }

    pub fn min_y(&self, terminals: &[usize; 2]) -> i32 {
        let [_zlim, ylim, xlim] = self.shape;
        (terminals[0] / xlim % ylim + terminals[1] / xlim % ylim) as i32
    }

    pub fn min_z(&self, terminals: &[usize; 2]) -> i32 {
        let xlim = self.shape[2];
        (terminals[0] % xlim + terminals[1] % xlim) as i32
    }

    // pub fn get_octant(&self, terminals: &[usize; 2])
    // {
    //     let [zlim, ylim, xlim] = self.shape;
    //     let start_coordinates: [usize; 3] = self.coordinates[terminals[0]];
    //     let stop_coordinates: [usize; 3] = self.coordinates[terminals[1]];
        
    // }

    // pub fn min_different_axes(&self, terminals: &[usize; 2]) -> i32
    // {
    //     let [start, goal] = terminals;
    //     let sc = self.coordinates[*start];
    //     let gc = self.coordinates[*goal];

    //     (sc[0] != gc[0]) as i32 + (sc[1] != gc[1]) as i32 + (sc[1] != gc[1]) as i32
    // }

    fn calculate_volume(&self, current: usize, [gz, gy, gx]: [usize; 3]) -> u32 { // goal tuple can be turned into goal array or lookup as well
        let [qz, qy, qx] =  self.coordinates[current];
        (gz.abs_diff(qz) * gy.abs_diff(qy) * gx.abs_diff(qx)) as u32
    }

    pub fn min_volume(&self, terminals: &[usize; 2]) -> u32
    {
        let [start, goal] = terminals;
        // let start_coordinate = self.coordinates[start];
        let goal_coordinate = self.coordinates[*goal];
        
        self.calculate_volume(*start, goal_coordinate)
    }    



    // pub fn sort_max_distance(&self, terminals: &[usize; 2]) -> Reverse<i32>{
    //     let [start, goal] = terminals;
    //     // let start_coordinate = self.coordinates[start];
    //     let goal_coordinate = self.coordinates[*goal];
        
    //     Reverse(self.absolute_distance(*start, goal_coordinate))
    // }
}
