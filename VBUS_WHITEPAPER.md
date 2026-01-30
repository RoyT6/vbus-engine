# VBUS - ViewerDBX Bus System

## Technical Whitepaper v2.0

---

## LEGAL NOTICE AND COPYRIGHT

**Copyright (c) 2025-2026 Roy Taylor. All Rights Reserved.**

**Ownership: Framecore Inc.**

This document and all associated software, code, and intellectual property are owned by Framecore Inc. and are protected under United States copyright law (17 U.S.C.), trademark law, trade secret law, and applicable international intellectual property treaties.

### CONTACT INFORMATION

**Owner:** Roy Taylor
**Company:** Framecore Inc.
**Email:** roy@framecore.com

### TRADEMARK NOTICE

VBUS, ViewerDBX, ALGO 95.4, and the Framecore name and logo are trademarks or registered trademarks of Framecore Inc. in the United States and other jurisdictions. All other trademarks, service marks, and trade names referenced in this document are the property of their respective owners.

---

## IMPORTANT: USER ACKNOWLEDGMENT AND ASSUMPTION OF RISK

**BY DOWNLOADING, INSTALLING, ACCESSING, OR USING VBUS SOFTWARE, YOU ACKNOWLEDGE THAT YOU HAVE READ, UNDERSTOOD, AND AGREE TO BE BOUND BY THE FOLLOWING TERMS.**

### INSTALLATION AT USER'S SOLE DISCRETION

The VBUS software and all related components are provided for installation **at the User's sole discretion**. The decision to download, install, configure, and operate this software is made entirely by You (the "User"). Framecore Inc., Roy Taylor, and any affiliated parties make no representations regarding the suitability of this software for any particular purpose or environment.

### LIMITATION OF LIABILITY

**TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL FRAMECORE INC., ROY TAYLOR, OR ANY DIRECTORS, OFFICERS, EMPLOYEES, AGENTS, AFFILIATES, SUCCESSORS, OR ASSIGNS BE LIABLE FOR:**

1. **ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, PUNITIVE, OR EXEMPLARY DAMAGES**, including but not limited to damages for loss of profits, goodwill, data, or other intangible losses;

2. **ANY DAMAGES ARISING FROM:**
   - Use or inability to use the software;
   - Unauthorized access to or alteration of your data or transmissions;
   - Any interruption, suspension, or termination of the software;
   - Any bugs, viruses, or other harmful components transmitted through the software;
   - Any errors, inaccuracies, or omissions in the software or documentation;
   - Hardware damage, system failures, or data loss;
   - Third-party claims arising from your use of the software;

3. **ANY MATTER BEYOND THE REASONABLE CONTROL OF FRAMECORE INC.**, regardless of whether Framecore Inc. has been advised of the possibility of such damages.

**THE TOTAL LIABILITY OF FRAMECORE INC. FOR ANY CLAIMS ARISING OUT OF OR RELATED TO THIS SOFTWARE SHALL NOT EXCEED THE AMOUNT YOU PAID FOR THE SOFTWARE, IF ANY, OR ONE HUNDRED U.S. DOLLARS ($100.00), WHICHEVER IS LESS.**

### DISCLAIMER OF WARRANTIES

**THIS SOFTWARE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO:**

- The implied warranties of merchantability;
- Fitness for a particular purpose;
- Non-infringement of third-party rights;
- Accuracy, reliability, or completeness;
- Uninterrupted or error-free operation;
- Security or freedom from viruses or malicious code.

**FRAMECORE INC. DOES NOT WARRANT THAT THE SOFTWARE WILL MEET YOUR REQUIREMENTS OR THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE.**

### INDEMNIFICATION

You agree to indemnify, defend, and hold harmless Framecore Inc., Roy Taylor, and their respective officers, directors, employees, agents, licensors, and suppliers from and against any and all claims, liabilities, damages, judgments, awards, losses, costs, expenses, or fees (including reasonable attorneys' fees) arising out of or relating to your violation of these terms or your use of the software.

### GOVERNING LAW AND JURISDICTION

This Agreement shall be governed by and construed in accordance with the laws of the **United States of America** and the **State of Delaware**, without regard to its conflict of law provisions. Any legal action or proceeding arising under this Agreement shall be brought exclusively in the federal or state courts located in Delaware, and the parties hereby consent to personal jurisdiction and venue therein.

### SEVERABILITY

If any provision of this Agreement is held to be unenforceable or invalid under applicable law, such provision shall be modified to the minimum extent necessary to make it enforceable, and the remaining provisions shall continue in full force and effect.

### ENTIRE AGREEMENT

This document constitutes the entire agreement between you and Framecore Inc. regarding the use of this software and supersedes all prior agreements and understandings, whether written or oral.

---

## What is VBUS?

### Overview

**VBUS (ViewerDBX Bus System)** is a high-bandwidth, deterministic software architecture designed for Machine Learning (ML) pipeline orchestration. Rather than using traditional software message queues or loose directory structures, VBUS reimagines the software environment as a hardware circuit.

### How VBUS Works

VBUS operates on a fundamental principle: treat software components as if they were hardware devices connected via a physical bus, similar to how a computer's CPU communicates with memory and peripherals.

**Key Components:**

1. **ALGO 95.4 Engine (The "CPU"):** This is the central controller that enforces temporal validity, executes pipeline phases, manages rule enforcement, and orchestrates all system operations.

2. **GPU Fabric (The Compute Layer):** Handles raw stochastic computation using ML frameworks like XGBoost, CatBoost, and Random Forest. GPU acceleration is mandatory for this layer.

3. **The System Bus (VBUS):** The single source of truth that provides cached access, signal passing, cryptographic audit chaining, and deterministic communication between all components.

4. **Memory-Mapped Devices:** System directories and modules are treated as hardware devices with specific read/write permissions, latency constraints, and functional roles rather than passive storage.

**Why This Approach?**

The VBUS architecture solves critical problems in ML pipelines:
- **Prevents State Cheating:** No component can bypass the bus to alter data without triggering an audit event
- **Ensures Auditability:** Every transaction across the bus is logged
- **Guarantees Determinism:** Execution order is guaranteed; race conditions are architecturally impossible
- **Optimizes Throughput:** Memory-mapped design enables parallel batch ingestion

---

## Document Information

| Field | Value |
|-------|-------|
| **Title** | ViewerDBX VBUS: A High-Bandwidth, Deterministic Bus Architecture for ML Orchestration |
| **Version** | 2.0 |
| **Date** | January 2026 |
| **Copyright Holder** | Roy Taylor |
| **Owner** | Framecore Inc. |

---

## 1. Executive Summary

This document outlines the architecture of the **ViewerDBX Bus System (VBUS)**, a design approach to Machine Learning (ML) pipeline orchestration that treats the software environment as a hardware circuit.

In this paradigm, the **ALGO 95.4** engine functions as the Central Processing Unit (CPU), enforcing temporal validity and logic, while the **GPU Fabric** handles raw stochastic computation. All communication is routed through a high-bandwidth, deterministic **System Bus**, ensuring that no component can bypass the architectural constraints.

This design mimics the physical properties of High Bandwidth Memory (HBM) and NVLink interconnects found in modern GPU hardware to solve software problems related to state integrity, auditability, and data throughput.

---

## 2. Architectural Philosophy: The Hardware Analogy

In high-performance GPU design for ML, the bus is the critical bottleneck. It defines the "highway" connecting processing cores to memory. If the bus is narrow or slow, cores idle, and performance collapses. ViewerDBX applies this hardware physics to software logic.

### 2.1 The Core Concept

The VBUS architecture posits that file system directories and software modules are not passive storage, but **Memory-Mapped Devices** connected via a strictly audited bus.

- **ALGO 95.4 (CPU):** Executes pipeline phases, enforces system rules, and manages the Control/IO Hub.
- **VBUS (System Bus):** A single source of truth that provides cached access, signal passing, and cryptographic audit chaining.
- **The "Folders":** These act as PCIe devices, HBM banks, or Secure Registers. They are not random storage bins; they are functional components with specific read/write permissions and latency constraints.

---

## 3. System Topology

### 3.1 The ViewerDBX System Bus (VBUS v1.0)

The backbone of the architecture. It is designed to be:

| Property | Description |
|----------|-------------|
| **High-Bandwidth** | Optimized for high-throughput feature extraction |
| **Deterministic** | Execution order is guaranteed; race conditions are architecturally impossible |
| **Audited** | Every transaction across the bus is logged |
| **Memory-Mapped** | Components are addressed via virtual memory addresses rather than file paths |

### 3.2 The Compute Layer

- **The CPU (ALGO 95.4):** The central controller. It handles training and inference logic, orchestration, and rule enforcement. It owns the "Proof-of-Work" required to dispatch tasks to the GPU.

- **The GPU Fabric:** This layer contains the stochastic engines—**XGBoost**, **CatBoost**, and **Random Forest (RF)**. GPU acceleration is required for this fabric. This layer connects to the VBUS via a specialized high-speed interface (analogous to NVLink) to handle massive matrix operations.

### 3.3 Memory and IO Hierarchy

The system utilizes a tiered memory architecture to manage data latency and integrity.

#### GPU Memory Manager

```
L1 (GPU VRAM)        -> Hot data currently being used for ML operations
L2 (System RAM)      -> Warm data - features ready for GPU transfer
L3 (System RAM)      -> Cold data - full datasets, backup storage
```

#### A. Data Memory (VRAM-Like)

| Field | Value |
|-------|-------|
| **Function** | Stores active tensors and features |
| **Analogy** | GPU VRAM / HBM |

#### B. The Trust Ledger (Immutable ROM)

| Field | Value |
|-------|-------|
| **Function** | Stores validated data. Once data is written here, it cannot be altered |
| **Analogy** | Read-Only Memory (ROM) or Write-Once-Read-Many (WORM) storage |

#### C. Control / IO Hub (Chipset Logic)

| Field | Value |
|-------|-------|
| **Function** | Manages traffic between the CPU, the Memory-Mapped Devices, and External Interfaces |

---

## 4. Memory-Mapped Device Specifications

The VBUS architecture categorizes all system components as specific device types attached to the bus.

### 4.1 Data / Feature Sources (High-Throughput Reads)

These components are optimized for massive read operations (fetching training data).

| Component Type | Function |
|----------------|----------|
| **Primary Training Data** | Main historical dataset (HBM Bank 0) |
| **Abstract Data** | Generalized feature sets |
| **Priority Cache** | High-priority, frequently accessed cache (L1 Cache) |
| **Incoming Stream** | Volatile, incoming data stream (L2 Cache) |
| **Categorical Banks** | Categorical feature storage |

### 4.2 Model / Compute Extensions

Modules that provide algorithmic capability.

| Module | Function |
|--------|----------|
| **ALGO Engine** | Core logic kernel |
| **GPU Enablement** | Hardware abstraction layer; can issue "halt" commands if hardware is insufficient |
| **Diagnostics** | Diagnostic output registers |
| **Uncertainty Quantification** | Conformal prediction module |

### 4.3 Pipeline Control & Orchestration

| Component | Function |
|-----------|----------|
| **Orchestrator** | The bus arbiter |
| **Scheduler** | Scheduling logic |
| **Schema & Components** | Constraint logic and type definitions |

### 4.4 External Interfaces (IO / Edge)

The "Southbridge" of the architecture, handling communication with the outside world.

| Interface | Function |
|-----------|----------|
| **Network Edge** | External network connectivity |
| **Notifications** | Asynchronous notification system |
| **Remote Execution** | Remote code execution environments |

### 4.5 Economic & Decision Layers

| Layer | Function |
|-------|----------|
| **Resource Allocation** | Resource allocation logic |
| **Trust Scoring** | Trust scoring registers |

### 4.6 Audit / Compliance (Write-Once)

| Component | Function |
|-----------|----------|
| **AUDIT_LOGS** | The black box flight recorder. Behaves like ROM |
| **Compliance** | Compliance constraints |

---

## 5. Hardware Theory Implementation for ML

The VBUS design is derived from physical GPU constraints necessary for ML workloads.

### 5.1 Bus Width and Bandwidth

In physical GPUs, the **Internal Memory Bus** connects cores to VRAM. A wider bus prevents the "Memory Wall."

**VBUS Implementation:** The VBUS treats data sources not as files to be parsed sequentially, but as memory blocks to be ingested in parallel batches. This maximizes the saturation of the GPU Fabric.

### 5.2 Latency vs. Throughput

| Scenario | Description | VBUS Solution |
|----------|-------------|---------------|
| **Memory-Bound** | If the bus is slow, the CPU sits idle | VBUS mitigates this by pre-caching priority data (L1 Cache) close to the compute engine |
| **Compute-Bound** | If the model is too complex, the bus waits | VBUS balances this via the GPU Enablement module which tunes batch sizes dynamically |

### 5.3 Coherence and Synchronization (NVLink Analogy)

Just as NVLink synchronizes gradients across multiple physical GPUs, the **Control Hub** ensures that the **Trust Ledger** and **Audit Logs** remain perfectly synchronized with the **Data Memory**. This prevents "state cheating" where a model processes data that hasn't been validated.

---

## 6. Conclusion

By implementing the VBUS architecture as a **Bus-Centric System**, users can achieve a level of rigor absent in traditional software pipelines.

| Principle | Description |
|-----------|-------------|
| **Integrity Enforcement** | You cannot bypass the bus to alter data without triggering an audit event |
| **Predictable Performance** | The separation of Control (CPU) and Fabric (GPU) ensures efficient resource usage |
| **Architectural Trust** | The immutable Trust Ledger serves as the root of trust |

---

## ALGO 95.4 Compliance Requirements

| Requirement | Status |
|-------------|--------|
| GPU execution | **Required** |
| cuDF for parquet I/O | **Recommended** |
| Audit logging | **Required** |
| Bus arbitration | **Required** |

---

## Appendix A: Technical Specifications

### Supported Hardware

| Component | Minimum Requirement |
|-----------|---------------------|
| GPU | NVIDIA GPU with 8GB+ VRAM (CUDA-compatible) or AMD equivalent |
| System RAM | 16GB DDR4 |
| Storage | SSD recommended for cache operations |
| CUDA Version | 11.0+ (12.0+ recommended) |

### Supported ML Frameworks

| Framework | GPU Support | Status |
|-----------|-------------|--------|
| XGBoost | gpu_hist | **Supported** |
| CatBoost | GPU | **Supported** |
| cuML Random Forest | GPU | **Supported** |
| LightGBM | GPU | **Supported** |

### RAPIDS Ecosystem (Optional)

| Component | Version | Purpose |
|-----------|---------|---------|
| cuDF | 24.02+ | GPU DataFrames |
| cuML | 24.02+ | GPU Machine Learning |
| CuPy | 13.0+ | GPU Array Operations |

---

## Appendix B: Licensing Terms

### Software License

The VBUS software is provided under a proprietary license. Users must agree to all terms and conditions prior to installation. Unauthorized use, reproduction, or distribution is strictly prohibited.

### User Configuration

Users are responsible for providing their own API credentials and configuration settings as required for their specific deployment environment. Framecore Inc. does not provide, store, or have access to user credentials.

### Third-Party Components

This software may utilize open-source components licensed under various open-source licenses. Users are responsible for compliance with all applicable third-party license terms.

---

## Contact Information

| Inquiry Type | Contact |
|--------------|---------|
| **General Inquiries** | roy@framecore.com |
| **Owner** | Roy Taylor |
| **Company** | Framecore Inc. |

---

**Document ID:** VBUS-WP-2026-001
**Revision:** 2.0
**Last Updated:** January 2026

---

*Copyright (c) 2025-2026 Roy Taylor. All Rights Reserved.*

*Owned by Framecore Inc.*

*VBUS, ViewerDBX, and ALGO 95.4 are trademarks of Framecore Inc.*

*This software is installed and used at the User's sole discretion and risk. Framecore Inc. and Roy Taylor disclaim all liability to the maximum extent permitted by law.*

---

## ACCEPTANCE

**BY USING THIS SOFTWARE, YOU ACKNOWLEDGE THAT YOU HAVE READ THIS AGREEMENT, UNDERSTAND IT, AND AGREE TO BE BOUND BY ITS TERMS AND CONDITIONS.**
