# School Bus Route Optimization

This Python code repository contains an implementation of different algorithms for optimizing school bus routes to transport students efficiently while adhering to capacity constraints. The algorithms covered in this repository include Genetic Algorithm, Simulated Annealing, Nearest Neighbor, and Clarke-Wright Savings Algorithm.

## Introduction

Transporting students to school efficiently is a crucial task, and this repository provides solutions to the School-Bus Routing Problem with Single Load Plan (SBRP-SLP). The goal is to find the shortest routes for school buses while ensuring that the total number of students on each route does not exceed the bus's capacity.

## Key Features

- **Multiple Optimization Algorithms**: This repository includes implementations of several optimization algorithms, allowing you to compare their performance in solving the school bus routing problem.

- **Easy Configuration**: You can generate random bus stops and customize parameters such as the number of stops, bus capacity, and more.

- **Algorithm Output**: The code outputs the best routes found by each algorithm and their respective distances, making it easy to evaluate their effectiveness.

- **Flexible Input**: You can load bus stop data from files, making it possible to analyze real-world scenarios.

## Usage

This codebase can be utilized by researchers, transportation planners, and anyone interested in optimizing school bus routes. You can customize the parameters and algorithms used to address your specific routing problems.

## Algorithms

1. **Genetic Algorithm**: Genetic algorithms are used to evolve a population of bus routes over multiple generations to find the best possible route.

2. **Simulated Annealing**: Simulated Annealing is a stochastic optimization technique that explores the solution space by accepting or rejecting new routes based on a temperature schedule.

3. **Nearest Neighbor (Graph Theory)**: This algorithm constructs routes by selecting the nearest neighbor at each stop, taking into account capacity constraints.

4. **Clarke-Wright Savings Algorithm**: This heuristic algorithm calculates savings for merging routes and progressively combines them to form efficient routes.

## How to Run

To run the code and optimize school bus routes, follow these steps:

1. Clone this repository to your local machine.
2. Make sure you have Python 3 installed.
3. Run the `main()` function in the provided Python script.

## Contributing

Contributions to this repository are welcome! If you have suggestions, improvements, or bug fixes, feel free to open issues or create pull requests. Please follow the existing code style and include clear commit messages.

## License

This project is licensed under the MIT License. You can find the license details in the [LICENSE](LICENSE) file.
