# Prerequisite Knowledge Graph


## Easy concepts

### Can analyze bandwidth usage of Ricart-Agrawala
> The Ricart-Agrawala Algorithm has a bandwidth usage of 2*(N-1) messages per enter operation, which can be high for large N.
- **Topic:** Mutual Exclusion in Distributed Systems
- **Pages:** 7, 25
- **Keywords:** bandwidth, usage, ricart-agrawala, analyze, algorithm, messages

### Can explain Maekawa's Algorithm
> Maekawa's Algorithm reduces the number of replies needed for mutual exclusion by using voting sets, allowing only a subset of processes to grant permission.
- **Topic:** Mutual Exclusion in Distributed Systems
- **Pages:** 9, 10, 12
- **Keywords:** maekawa's, algorithm, explain, reduces, number, replies

### Can explain Ricart-Agrawala Algorithm
> The Ricart-Agrawala Algorithm is a distributed mutual exclusion algorithm that uses Lamport timestamps to order requests for critical section access.
- **Topic:** Mutual Exclusion in Distributed Systems
- **Pages:** 3, 4, 5
- **Keywords:** algorithm, ricart-agrawala, explain, distributed, mutual, exclusion


## Medium concepts

### Can analyze performance of Maekawa's Algorithm
> Maekawa's Algorithm has improved bandwidth usage compared to Ricart-Agrawala, with O(√N) messages per enter operation, making it more efficient for large systems.
- **Topic:** Mutual Exclusion in Distributed Systems
- **Pages:** 18, 25
- **Keywords:** maekawa's, algorithm, analyze, performance, improved, bandwidth
- **Prerequisites:** `Can analyze bandwidth usage of Ricart-Agrawala`, `Can explain Maekawa's Algorithm`

### Can identify safety in Maekawa's Algorithm
> Safety in Maekawa's Algorithm is ensured because a process can only receive replies from its voting set, preventing multiple processes from entering the critical section simultaneously.
- **Topic:** Mutual Exclusion in Distributed Systems
- **Pages:** 19, 21
- **Keywords:** safety, maekawa's, algorithm, identify, ensured, because
- **Prerequisites:** `Can explain Ricart-Agrawala Algorithm`, `Can explain Maekawa's Algorithm`


## Hard concepts

### Can describe the Ring Election Algorithm
> The Ring Election Algorithm organizes processes in a logical ring and uses message passing to elect a leader based on attributes.
- **Topic:** Mutual Exclusion in Distributed Systems
- **Pages:** 35, 48
- **Keywords:** ring, election, algorithm, describe, organizes, processes
- **Prerequisites:** `Can identify safety in Maekawa's Algorithm`, `Can analyze performance of Maekawa's Algorithm`, `Can explain leader election in distributed systems`

### Can explain leader election in distributed systems
> Leader election is a process in distributed systems to select a single process as the coordinator or leader among a group of processes.
- **Topic:** Mutual Exclusion in Distributed Systems
- **Pages:** 26, 29
- **Keywords:** leader, election, distributed, systems, process, explain
- **Prerequisites:** `Can explain Ricart-Agrawala Algorithm`, `Can explain Maekawa's Algorithm`

### Can identify safety and liveness in election algorithms
> Election algorithms must ensure safety by electing one leader and liveness by guaranteeing that the election process terminates successfully.
- **Topic:** Mutual Exclusion in Distributed Systems
- **Pages:** 31, 32
- **Keywords:** election, safety, liveness, algorithms, identify, must
- **Prerequisites:** `Can describe the Ring Election Algorithm`
