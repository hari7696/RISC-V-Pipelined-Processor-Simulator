# RISC-V-Pipelined-Processor-Simulator

## Project Overview
This academic project[CDA 5155] implements a simulator for a pipelined processor based on the RISC-V architecture. The simulator processes RISC-V instruction sets in a cycle-by-cycle manner, simulating the behavior of a pipelined processor with five functional stages (Fetch, Decode, Execute, Memory, Write-Back). It outputs a detailed trace of processor states, including register values, queue contents, and memory data at each cycle.

The goal of this simulator is to demonstrate pipeline operations, including handling dependencies, managing instruction queues, and synchronizing the execution of different instruction types.

## Features
* Supports standard RISC-V instruction formats and operations.
* Implements instruction fetch, decode, execution, memory access, and write-back stages.
* Handles structural, data, and control hazards within the pipeline.
* Produces a detailed simulation trace for debugging and evaluation.
* Simulates queues and memory as part of the pipeline.

## Input
The simulator accepts a RISC-V assembly text file as input. Each instruction in the file follows the RISC-V format. The input file is specified as a command-line argument when running the simulator.


## Output
The simulator generates a simulation.txt file containing the pipeline trace for each cycle. The output includes:

1. Instruction Fetch and Decode Status: Displays instructions waiting or executed.
2. Queue Contents: Shows the state of pre-issue, pre-execution, and post-execution queues.
3. Register File: Displays the contents of the 32 general-purpose registers.
4. Memory Contents: Shows the memory state after each cycle.

## Functional Description
- **Instruction Fetch/Decode:** Fetches up to two instructions per cycle, decodes them, and places them in the pre-issue queue. Handles branch and break instructions.
- **Issue Unit:** Uses the scoreboard algorithm to issue instructions while avoiding RAW, WAW, and WAR hazards.
- **Execution Units:**
    ALU1 for memory address calculations.
    ALU2 for arithmetic operations.
    ALU3 for logical operations.
- **Memory Access:** Handles lw and sw instructions.
- **Write-Back:** Updates the register file with results from memory and ALU operations.

## How to run

```bash
python ./Vsim input_instructions.txt
```
- Post run check the generated simulation.txt for the output traced

## limitations
- Data forwarding is not implemented.
- assumes all instructions are RISC-V
- Does not handle exceptions.