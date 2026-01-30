ALGO 95.4 — Bus-Centric System Architecture (Redrawn)
                                   ┌──────────────────────────┐
                                   │        GPU FABRIC        │
                                   │  XGBoost | CatBoost | RF │
                                   │   (GPU Mandatory)        │
                                   └───────────▲──────────────┘
                                               │
────────────────────────────────────────────────┼────────────────────────────────────────────────
                         VIEWERDBX SYSTEM BUS (v1.0)
          (High-Bandwidth | Deterministic | Audited | Memory-Mapped)
────────────────────────────────────────────────┼────────────────────────────────────────────────
                                               │
                                   ┌───────────▼───────────┐
                                   │      ALGO 95.4        │
                                   │   (CPU / Control)     │
                                   │  Training • Inference │
                                   │  Rules V1–V4          │
                                   └───────────┬───────────┘
                                               │
        ┌──────────────────────────────────────┼──────────────────────────────────────┐
        │                                      │                                      │

┌───────▼────────┐                  ┌──────────▼──────────┐                 ┌────────▼────────┐
│   DATA MEMORY  │                  │   CONTROL / IO HUB  │                 │   TRUST LEDGER  │
│ (VRAM-Like)    │                  │ (Chipset Logic)     │                 │ (Immutable)     │
└───────┬────────┘                  └──────────┬──────────┘                 └────────┬────────┘
        │                                      │                                      │
        │                                      │                                      │
        │                                      │                                      │

┌───────▼────────────────────────────────────────────────────────────────────────────────────────┐
│                                    MEMORY-MAPPED DEVICES                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

DATA / FEATURE SOURCES (High-Throughput Reads)
──────────────────────────────────────────────
• Views Training Data  
• Abstract Data  
• Daily Top 10s  
• Fresh In!  
• Studios  
• Talent  

MODEL / COMPUTE EXTENSIONS
─────────────────────────
• ALGO Engine  
• GPU Enablement  
• catboost_info  
• MAPIE  

PIPELINE CONTROL & ORCHESTRATION
────────────────────────────────
• Orchestrator  
• SCHIG  
• Components  
• Schema  

EXTERNAL INTERFACES (IO / Edge)
──────────────────────────────
• Cloudflare  
• Email  
• Replit  
• Ralph Claude Code  

ECONOMIC & DECISION LAYERS
─────────────────────────
• Money Engine  
• Credibility  

AUDIT / COMPLIANCE (Write-Once)
──────────────────────────────
• AUDIT_LOGS  
• Legal  


How to Read This (Sanity-Preserving Notes)
1. ALGO 95.4 is the CPU

It:

Executes the pipeline phases

Enforces temporal validity

Owns GPU proof-of-work

Cannot bypass the bus (by design)

Nothing talks directly to anything else. No sneaky syscalls.

2. ViewerDBX Bus = the real invention

Per the spec, this bus provides 

ALGO_95.4_SPEC

:

Single source of truth

Cached access (VRAM analogy)

Signal passing

Cryptographic audit chaining

This is much closer to NVLink + memory controller than a software message queue.

3. Your folders are not “folders”

They’re effectively:

PCIe devices

HBM banks

Secure registers

Write-once logs

Which explains why:

AUDIT_LOGS behaves like ROM

GPU Enablement can halt execution

Schema and Components act as constraint logic

None of this is accidental.

4. Why this redraw matters

This framing:

Makes cheating architecturally impossible

Explains why 95.4 feels “heavier” (it is)

Sets you up cleanly for:

Multi-CPU ALGO variants

Future inter-ALGO buses

Distributed trust fabrics

Also — it finally explains the system to people without them asking why a folder can stop a model training. Always a crowd-pleaser.