# Concept Map: myslides.pdf

_Pages processed: 51_

## 1. Mutual Exclusion in Distributed Systems

### 1.1 Classical Algorithms

#### Can explain Ricart-Agrawala Algorithm

> The Ricart-Agrawala Algorithm is a distributed mutual exclusion algorithm that uses Lamport timestamps to order requests for critical section access.

**Keywords:** Ricart-Agrawala, mutual exclusion, Lamport timestamps, distributed systems

**Pages:** 3, 4, 5

**Check your understanding:**
- What is the role of Lamport timestamps?
- How does the algorithm ensure safety?

#### Can analyze bandwidth usage of Ricart-Agrawala

> The Ricart-Agrawala Algorithm has a bandwidth usage of 2*(N-1) messages per enter operation, which can be high for large N.

**Keywords:** bandwidth, messages, Ricart-Agrawala, performance

**Pages:** 7, 25

**Check your understanding:**
- What is the bandwidth usage for an exit operation?
- How does this compare to other algorithms?

#### Can explain Maekawa's Algorithm

> Maekawa's Algorithm reduces the number of replies needed for mutual exclusion by using voting sets, allowing only a subset of processes to grant permission.

**Keywords:** Maekawa, voting sets, mutual exclusion, distributed systems

**Pages:** 9, 10, 12

**Check your understanding:**
- What is a voting set?
- How does Maekawa's Algorithm differ from Ricart-Agrawala?

#### Can identify safety in Maekawa's Algorithm

> Safety in Maekawa's Algorithm is ensured because a process can only receive replies from its voting set, preventing multiple processes from entering the critical section simultaneously.

**Keywords:** safety, Maekawa, voting sets, critical section

**Pages:** 19, 21

**Check your understanding:**
- How does the intersection of voting sets contribute to safety?
- What happens if two processes request access simultaneously?

#### Can analyze performance of Maekawa's Algorithm

> Maekawa's Algorithm has improved bandwidth usage compared to Ricart-Agrawala, with O(√N) messages per enter operation, making it more efficient for large systems.

**Keywords:** performance, bandwidth, Maekawa, efficiency

**Pages:** 18, 25

**Check your understanding:**
- What is the client delay in Maekawa's Algorithm?
- How does the performance scale with N?

#### Can explain leader election in distributed systems

> Leader election is a process in distributed systems to select a single process as the coordinator or leader among a group of processes.

**Keywords:** leader election, distributed systems, coordinator, election algorithms

**Pages:** 26, 29

**Check your understanding:**
- What is the purpose of leader election?
- What happens if the leader fails?

#### Can describe the Ring Election Algorithm

> The Ring Election Algorithm organizes processes in a logical ring and uses message passing to elect a leader based on attributes.

**Keywords:** Ring Election, algorithm, logical ring, message passing

**Pages:** 35, 48

**Check your understanding:**
- How are messages passed in the Ring Election Algorithm?
- What happens when multiple processes call for an election?

#### Can identify safety and liveness in election algorithms

> Election algorithms must ensure safety by electing one leader and liveness by guaranteeing that the election process terminates successfully.

**Keywords:** safety, liveness, election algorithms, leader

**Pages:** 31, 32

**Check your understanding:**
- What does safety guarantee in an election?
- What is meant by liveness in this context?
