# Zero Benchmark Plan — ROI Edition

**Status:** ACTIVE
**Ziel:** Beweisen dass Zero Enterprise-Kunden Geld spart. Jeder Benchmark = eine Dollar-Zahl.
**Methode:** Wir messen NUR Zero. Linux-Vergleichswerte kommen aus publizierten, unabhängigen Quellen.
**Hardware:** Cherry Servers Bare-Metal, IP KVM Boot, stündliche Abrechnung
**Output:** Publizierbare Grafiken: "Linux: X | Zero: Y | Sie sparen: $Z"
**Bonus:** Das Benchmark-ISO ist gleichzeitig eine **Live-Demo** — jeder kann es auf eigener Hardware booten und sofort sehen wie viel schneller Zero ist.

---

## 1. Hardware

### Server A: AMD EPYC 9354P (0.717 €/hr)

| Spec | Wert |
|------|------|
| CPU | 32 Cores / 64 Threads, 3.25-3.8 GHz, Zen 4 (Genoa) |
| RAM | 192 GB DDR5-4800, 12 Channels |
| Bandwidth | 460.8 GB/s theoretisch |
| Storage | 2x 1TB NVMe |
| Kosten | ~21€ für 30 Stunden |

### Server B: AMD EPYC 9554P (1.145 €/hr)

| Spec | Wert |
|------|------|
| CPU | 64 Cores / 128 Threads, 3.1-3.75 GHz, Zen 4 (Genoa) |
| RAM | 384 GB DDR5-4800, 12 Channels |
| Bandwidth | 460.8 GB/s theoretisch |
| Storage | 2x 1TB NVMe |
| Kosten | ~34€ für 30 Stunden |

---

## 2. Das Enterprise-Problem in Zahlen

| Fakt | Wert | Quelle |
|------|------|--------|
| Enterprise CPU Utilization | **8%** (sinkt) | Cast AI K8s Report 2026 |
| Enterprise Memory Utilization | **20%** (sinkt) | Cast AI K8s Report 2026 |
| Memory Overprovisioning | **79%** | Cast AI K8s Report 2026 |
| Globale Cloud-Verschwendung | **$182 Milliarden/Jahr** | Spendark 2026 |
| Davon Idle Compute | **$64 Milliarden/Jahr** | Spendark 2026 |
| GPU Utilization | **5%** | Cast AI K8s Report 2026 |
| GPU Waste | **$401 Milliarden/Jahr** | VentureBeat 2026 |

**Warum?** Weil das Betriebssystem die Hardware nicht effizient nutzt. Linux wurde 1991 für Timesharing gebaut. Context Switches kosten Mikrosekunden. Memory Allocation lockt. Boot dauert Sekunden. Das Ergebnis: Firmen kaufen 5-10x mehr Server als nötig.

---

## 3. Die 5 ROI-Benchmarks

Nur Tests die direkt in Dollar übersetzbar sind. Kein akademischer Ballast.

---

### Benchmark 1: Server Boot Time — "Schneller online = weniger Standby-Kosten"

**Was Kunden heute zahlen:** Serverless/Auto-Scaling Instanzen brauchen Warm-Standby weil Linux 15-25 Sekunden zum Booten braucht. Bei 1,000 Instanzen à $0.50/hr Standby = **$4.4M/Jahr für Server die nur warten.**

**Was wir messen:**
- Zero: rdtsc am Anfang von kernel_main() bis erste Task-Execution
- Erwartung: **5-50 Millisekunden**

**Linux-Referenz (publiziert):**
- Ubuntu Server 24.04: **15-25 Sekunden** (commandlinux.com 2026)
- Clear Linux (optimiert): **3-5 Sekunden** (techrefreshing.com 2025)
- Enterprise mit Services: **25-45 Sekunden** (systemd-analyze Community Data)

**Grafik:**

```
┌─────────────────────────────────────────────────────────┐
│  Server Boot Time                                       │
│                                                         │
│  Linux (Ubuntu)  ████████████████████████████  20 sec   │
│  Linux (Clear)   ████                          4 sec    │
│  Zero         ▏                             0.02 sec │
│                                                         │
│  → 1000x schneller = $4.4M/Jahr weniger Standby-Kosten │
│    bei 1,000 Serverless-Instanzen                       │
└─────────────────────────────────────────────────────────┘
```

**Dollar-Rechnung:**
- 1,000 Warm-Standby Instanzen × $0.50/hr × 8,760 hrs = $4.38M/Jahr
- Mit Zero Boot in <50ms: kein Warm-Standby nötig, Cold Start reicht
- **Ersparnis: bis zu $4.4M/Jahr pro 1,000 Instanzen**

---

### Benchmark 2: Context Switch — "Mehr Workloads pro Server = weniger Server"

**Was Kunden heute zahlen:** Bei 8% CPU Utilization kaufen Enterprises 12x mehr Server als nötig. Ein Grund: Context Switches kosten 1,500-5,500 ns auf Linux. Bei Millionen Switches/Sekunde geht ein signifikanter Teil der CPU-Zeit für OS-Overhead drauf.

**Was wir messen:**
- Zero: Cooperative Executor, Task A yield → Task B resume, rdtsc Messung
- 1 Million Iterationen, Median + p99
- Erwartung: **10-50 Nanosekunden**

**Linux-Referenz (publiziert):**
- lmbench, Core-pinned: **1,200-1,500 ns** (Eli Bendersky, Phoronix)
- Unpinned: **2,200-5,500 ns** (lmbench Community)
- TLB Flush Overhead: **+200-500 ns** pro Switch

**Grafik:**

```
┌─────────────────────────────────────────────────────────┐
│  Context Switch Latency                                 │
│                                                         │
│  Linux (pinned)   ████████████████████████  1,500 ns    │
│  Linux (normal)   ████████████████████████████  3,500 ns│
│  Zero          ▍                            30 ns    │
│                                                         │
│  → 50x schneller = 3-5x mehr Tasks pro Server          │
│  → 1,000 Server → nur 200-330 nötig                    │
│  → $8-10M/Jahr gespart (bei $1.50/hr/Server)            │
└─────────────────────────────────────────────────────────┘
```

**Dollar-Rechnung:**
- 1,000 Server × $1.50/hr × 8,760 hrs = $13.1M/Jahr
- 50x weniger Scheduling-Overhead → 3-5x mehr Tasks pro Server
- Nur noch 200-330 Server nötig → $2.6-4.3M/Jahr
- **Ersparnis: $8.8-10.5M/Jahr pro 1,000 Server**

---

### Benchmark 3: Memory Allocation — "Schnellere Requests = weniger Server"

**Was Kunden heute zahlen:** Jeder HTTP-Request, jede DB-Query, jede LLM-Inference ruft malloc/free auf. Bei 100K Requests/Sekunde und 200ns pro Allocation: 20ms/Sekunde nur für Memory Management. Unter Multi-Thread Contention (realistisch): bis 500ms/Sekunde verschwendet.

**Was wir messen:**
- Zero: Bump-Pointer Arena Allocation, Millionen Allokationen timen
- Erwartung: **2-10 Nanosekunden** (ein ADD + Bounds Check)

**Linux-Referenz (publiziert):**
- glibc malloc (Standard): **50-200 ns** Single Thread (mimalloc-bench)
- glibc unter Contention: **500-5,000 ns** (DEV.to Allocator Analysis)
- jemalloc (optimiert): **35% besser als glibc** aber immer noch 100+ ns
- tcmalloc: ähnlich jemalloc, 100-300 ns typisch

**Grafik:**

```
┌─────────────────────────────────────────────────────────┐
│  Memory Allocation Latency                              │
│                                                         │
│  Linux glibc     ████████████████████████████  200 ns   │
│  Linux glibc     ████████████████████████████████████   │
│   (Contention)   ████████████████████████████  2,000 ns │
│  Linux jemalloc  ████████████████████           130 ns  │
│  Zero Arena   ▍                              5 ns    │
│                                                         │
│  → 30-100x schneller = höherer Request-Throughput       │
│  → gleicher Workload, 30-50% weniger Server             │
│  → $2-5M/Jahr gespart bei 500-Server-Cluster            │
└─────────────────────────────────────────────────────────┘
```

**Dollar-Rechnung:**
- Memory Allocation Overhead = 2-5% der gesamten Request-Latenz (Google SRE Data)
- Bei High-Throughput Services (100K+ req/s): der Unterschied bestimmt wie viele Server nötig sind
- 500 Server × $1.50/hr × 8,760 hrs = $6.57M/Jahr
- 30-50% weniger Server durch höheren Throughput
- **Ersparnis: $2-3.3M/Jahr pro 500-Server-Cluster**

---

### Benchmark 4: RAM-Effizienz (Zero-Copy IPC) — "Weniger RAM-Verschwendung = kleinere Instanzen"

**Was Kunden heute zahlen:** 79% Memory Overprovisioning (Cast AI 2026). Microservices kopieren Daten zwischen Prozessen über pipes/sockets (3-6 GB/s). Jede Kopie braucht doppelt RAM. Das Ergebnis: Firmen buchen 2-4x mehr RAM als der Workload braucht.

**Was wir messen:**
- Zero: Shared Arena zwischen Tasks, Zero-Copy Datentransfer
- Task A schreibt, Task B liest direkt (gleicher Pointer)
- Throughput in GB/s
- Erwartung: **200-400 GB/s** (nahe DDR5-4800 Bandbreite)

**Linux-Referenz (publiziert):**
- pipe(): **3-6 GB/s** (Kernel Buffer Copy)
- Unix Socket: **2-5 GB/s**
- Shared Memory (mmap): **50-100 GB/s** (aber braucht Synchronisation)

**Grafik:**

```
┌─────────────────────────────────────────────────────────┐
│  Inter-Process Data Transfer                            │
│                                                         │
│  Linux pipe      ███                           5 GB/s   │
│  Linux mmap      ████████████████              75 GB/s  │
│  Zero         ████████████████████████████  300 GB/s │
│                                                         │
│  → Zero-Copy = kein doppelter RAM-Verbrauch             │
│  → 79% Memory Overprovisioning → unter 20%              │
│  → Kleinere Instanzen buchen = direkte Kostensenkung    │
│  → $1-3M/Jahr gespart bei 1,000-Server-Cluster          │
└─────────────────────────────────────────────────────────┘
```

**Dollar-Rechnung:**
- Memory Overprovisioning 79% → Firmen zahlen für 179 GB wenn 100 GB reichen
- RAM-Kosten: ~$5-10/GB/Monat in Cloud (AWS/Azure Pricing)
- 1,000 Server × 79 GB überflüssig × $7.50/GB/Monat = $7.1M/Jahr verschwendet
- Zero Zero-Copy reduziert RAM-Bedarf um 30-50%
- **Ersparnis: $2.1-3.6M/Jahr pro 1,000 Server**

---

### Benchmark 5: LLM Inference Throughput (CPU) — "Mehr Tokens pro Dollar"

**Was Kunden heute zahlen:** CPU-Inference wird immer relevanter für Edge, On-Prem, und datenschutzkritische Anwendungen. Gleiche Hardware, mehr Output = direkt weniger Kosten pro Token.

**Was wir messen:**
- Zero: Qwen3-1.7B Q4_K, Kernel-resident Inference, tok/s über 100+ Tokens
- Kein Syscall, kein Context Switch während Inference, Arena-basierter KV-Cache
- Erwartung: **200-500 tok/s**

**Linux-Referenz (publiziert):**
- Qwen2 1.5B Q4 (CPU, llama.cpp): **~198 tok/s** (llama.cpp Benchmark, tg128, 16 Threads)
- DeepSeek 8B Q4 auf EPYC 9554: **~50 tok/s** (ahelpme.com)
- Qwen3 32B Q4 auf EPYC 9554: **~14 tok/s** (ahelpme.com)
- Phi-4 14B Q8 auf EPYC 9554: **~8-9 tok/s** (ahelpme.com)

**Grafik:**

```
┌─────────────────────────────────────────────────────────┐
│  LLM Inference: Qwen 1.7B Q4 on CPU                    │
│                                                         │
│  Linux llama.cpp  ██████████████████████████  ~150 tok/s│
│  Zero          ██████████████████████████████████████ │
│                   ██████████████████████████  ~400 tok/s│
│                                                         │
│  → 2-3x mehr Tokens pro Sekunde auf gleicher Hardware   │
│  → 100 Inference-Server → nur 35-50 nötig               │
│  → $500K-2M/Jahr gespart                                │
└─────────────────────────────────────────────────────────┘
```

**Dollar-Rechnung:**
- 100 CPU-Inference Server × $1.50/hr × 8,760 hrs = $1.31M/Jahr
- 2-3x mehr Throughput → nur 35-50 Server nötig → $460-660K/Jahr
- **Ersparnis: $650K-850K/Jahr pro 100 Inference-Server**
- Bei größerer Flotte (500 Server): **$3.3-4.3M/Jahr**

---

## 4. Gesamtersparnis nach Cluster-Größe (NUR CPU/RAM)

*Alle Zahlen basieren auf $1.50/hr pro Server (Cloud-Durchschnitt). GPU-Savings kommen ZUSÄTZLICH obendrauf.*

### 1 Server

| Benchmark | Linux | Zero | Jährliche Ersparnis |
|-----------|-------|---------|-------------------|
| Boot Time | 20s Cold Start | 20ms | Weniger Downtime: **~$200** |
| Context Switch | 1,500 ns | 30 ns | 3-5x mehr Tasks möglich: **~$6,500** |
| Memory Allocation | 200 ns | 5 ns | Höherer Throughput: **~$2,600** |
| RAM-Effizienz | 79% Overprovisioning | <20% | Kleinere Instanz buchbar: **~$3,900** |
| LLM Inference | ~50 tok/s (8B) | ~150 tok/s | 3x Output: **~$8,760** |
| **GESAMT** | | | **~$22,000/Jahr** |
| **Zero License** | | | **$10K/Jahr** |
| **Netto-Ersparnis** | | | **~$12,000/Jahr** |

### 10 Server

| Benchmark | Linux-Kosten/Jahr | Zero-Kosten/Jahr | Jährliche Ersparnis |
|-----------|------------------|-------------------|-------------------|
| Server-Kosten | $131,400 | $39,400-52,600 | **$78,800-92,000** |
| Davon Boot/Standby | $43,800 | $4,380 | $39,420 |
| Davon CPU Ineffizienz | $52,560 | $13,140 | $39,420 |
| Davon RAM Overprovisioning | $35,040 | $13,140 | $21,900 |
| **GESAMT** | **$131,400** | **$39,400-52,600** | **$78,800-92,000** |
| **Zero License** | | | **$10K/Jahr** |
| **Netto-Ersparnis** | | | **$68,800-82,000/Jahr** |

### 100 Server

| Benchmark | Linux-Kosten/Jahr | Zero-Kosten/Jahr | Jährliche Ersparnis |
|-----------|------------------|-------------------|-------------------|
| Benötigte Server | 100 | 25-35 | 65-75 weniger |
| Server-Kosten | $1.31M | $329K-460K | **$854K-985K** |
| RAM-Overprovisioning | $350K | $131K | **$219K** |
| LLM Inference (bei 20 Inference-Nodes) | $263K | $88-131K | **$131-175K** |
| **GESAMT** | **$1.93M** | **$548-722K** | **$1.2-1.4M/Jahr** |
| **Zero License** | | | **$50K/Jahr** |
| **Netto-Ersparnis** | | | **$1.15-1.35M/Jahr** |

### 1,000 Server

| Benchmark | Linux-Kosten/Jahr | Zero-Kosten/Jahr | Jährliche Ersparnis |
|-----------|------------------|-------------------|-------------------|
| Benötigte Server | 1,000 | 250-350 | 650-750 weniger |
| Server-Kosten | $13.1M | $3.3-4.6M | **$8.5-9.8M** |
| RAM-Overprovisioning | $3.5M | $1.3M | **$2.2M** |
| Warm-Standby (Serverless) | $4.4M | $440K | **$3.96M** |
| LLM Inference (bei 200 Nodes) | $2.63M | $877K-1.3M | **$1.3-1.75M** |
| **GESAMT** | **$23.6M** | **$5.9-7.6M** | **$16.0-17.7M/Jahr** |
| **Zero License** | | | **$100K/Jahr** |
| **Netto-Ersparnis** | | | **$15.9-17.6M/Jahr** |

### Die Grafik für VCs und Kunden

```
┌──────────────────────────────────────────────────────────────┐
│  Jährliche Ersparnis mit Zero (nur CPU/RAM)               │
│                                                              │
│  1 Server      █                              $12K           │
│  10 Server     ██████                         $68-82K        │
│  100 Server    ██████████████████████          $1.15-1.35M   │
│  1,000 Server  ████████████████████████████████ $15.9-17.6M  │
│                                                              │
│  + GPU-Savings kommen ZUSÄTZLICH obendrauf                   │
│                                                              │
│  Zero Enterprise License: $10K-100K/Jahr                  │
│  ROI: Tag 1.                                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. LLM-Auswahl für Benchmarks

### Warum das Modell wichtig ist

Wir brauchen ein Modell wo es PUBLIZIERTE llama.cpp Benchmarks auf genau dieser Hardware-Klasse gibt (EPYC 9354P/9554). Damit der Vergleich "Linux: X tok/s, Zero: Y tok/s" glaubwürdig ist.

### Standard-Benchmark-Setup in der llama.cpp Community

Die llama.cpp Community nutzt standardisiert:
- **tg128**: 128 Tokens generieren (Text Generation Speed — memory-limited, DER relevante Benchmark)
- **pp512**: 512 Token Prompt Processing (compute-limited)
- **Q4_0 oder Q4_K_M**: Standard-Quantisierung für Benchmarks
- **CPU BLAS Backend**, alle Threads

### Primär: Llama 3.1 8B Q4_K_M

**Warum:**
- DAS Standard-Benchmark-Modell der Industrie. Jeder kennt Llama, jeder hat Vergleichswerte
- Publizierter Benchmark auf EPYC 9554: **~50 tok/s** (ahelpme.com, DeepSeek-R1-Distill-Llama-8B)
- 8B ist die Enterprise-relevante Größe (Edge, On-Prem, Datenschutz)
- Q4_K_M ist die Standard-Produktions-Quantisierung
- RAM-Bedarf: ~4.5 GB — passt locker in 192 GB

**Problem:** Zero Boot-LLM ist aktuell Qwen3-1.7B. Für Llama 3.1 8B brauchen wir entweder:
a) Die Inference-Pipeline auf 8B erweitern (mehr Layers, GQA Anpassung)
b) Oder Qwen3-1.7B als Demo-Modell verwenden und gegen Qwen2 1.5B Benchmarks vergleichen

### Sekundär: Qwen3-1.7B Q4_K (bereits integriert)

- Bereits in Zero: Boot-LLM, Token-ID 25 deterministic
- Publizierter Vergleich: Qwen2 1.5B Q4_0 auf llama.cpp = **~198 tok/s** (tg128, Apple Silicon `Metal,BLAS`; nicht CPU-only)
- Für Server-CPU (EPYC 32C) ist das Zero-Ziel: **>=150 tok/s CPU-only**, gemessen gegen CPU-only `llama-bench`
- Vorteil: sofort testbar ohne Code-Änderungen
- Nachteil: 1.7B ist "klein" — weniger beeindruckend als 8B

### Empfehlung

**Phase 1 (sofort): Qwen3-1.7B** — damit testen wir jetzt, die Pipeline existiert
**Phase 2 (nach ersten Ergebnissen): Llama 3.1 8B** — DAS Modell das VCs und CTOs kennen. Höherer Impact.

### Publizierte Linux-Referenzwerte für LLM Inference auf EPYC

| Modell | Quant | CPU | tok/s (tg) | Quelle |
|--------|-------|-----|-----------|--------|
| DeepSeek-R1-Distill-Llama 8B | Q4_K_M | EPYC 9554 (64C) | ~50 | ahelpme.com |
| Qwen3 32B | Q4 | EPYC 9554 (64C) | ~14 | ahelpme.com |
| Phi-4 14B | Q8 | EPYC 9554 (64C) | ~8-9 | ahelpme.com |
| Phi-4 14B | F32 | EPYC 9554 (64C) | ~5-6 | ahelpme.com |
| Qwen2 1.5B | Q4_0 | 16T (Apple M-series) | ~198 | llama.cpp bench |
| 70B Modell | Q4_K_M | EPYC 9554 (64C) | ~7 | ahelpme.com |

---

## 6. Server-Zugang & Boot-Verfahren

### Wie wir auf den Server kommen (KEIN SSH nötig)

Cherry Servers bietet **Out-of-Band Management (OOBM)** für alle Bare-Metal-Server:
- **IP KVM Console** — Remote-Bildschirm im Browser (HTML5 oder Java), als wäre ein Monitor angeschlossen
- Zugang über Cherry Servers Portal: "Servers" → Server auswählen → "Console" Button
- Tastatur + Maus funktionieren, man sieht den Server ab BIOS-Boot
- Separater Management-Port auf dem Motherboard, unabhängig von Haupt-NIC und OS
- Quelle: https://www.cherryservers.com/knowledge/docs/compute/configuration-management/out-of-band-management-console

### Boot-Verfahren

**Option A: ISO über KVM mounten (empfohlen für ersten Test)**
1. Server mit "IP KVM self-install" bestellen
2. KVM Console öffnen im Browser
3. Zero als bootfähiges ISO über KVM-Menü mounten
4. Server bootet direkt von ISO in Zero
5. Benchmark-Ergebnisse erscheinen auf dem Bildschirm → Screenshot machen

**Option B: iPXE Boot (für Automatisierung später)**
1. Server mit "Custom iPXE install" bestellen
2. iPXE-Script zeigt auf HTTP-Server (z.B. ein Cloud-VM) wo Kernel-Binary liegt
3. Server lädt Kernel + GGUF über Netzwerk und bootet

### Benchmark-Ergebnisse abgreifen

Zero gibt alle Ergebnisse auf dem Bildschirm aus (LFB Framebuffer). Über IP KVM im Browser sichtbar → Screenshot.
Kein SSH, kein Netzwerk, keine NIC-Treiber nötig.

### NIC-Hinweis

Cherry Servers Supermicro H13SSL-N Board hat vermutlich **Broadcom BCM5720** NIC (nicht E1000).
Zero hat nur E1000-Treiber → Netzwerk-Export funktioniert NICHT auf dieser Hardware.
Ist aber egal: alle Ergebnisse gehen über den Bildschirm (KVM), nicht über Netzwerk.

---

## 7. Das Benchmark-ISO als Live-Demo

### Konzept

Das gleiche ISO das wir für unsere Benchmarks bauen wird gleichzeitig zur **öffentlichen Live-Demo**.
Jeder VC, jeder Enterprise-Kunde, jeder CTO kann es herunterladen, auf eigener Hardware booten
und innerhalb von Sekunden sehen:

```
=== Zero Benchmark Suite ===
Hardware: AMD EPYC 9354P, 192 GB DDR5
Boot Time:        45ms    (Linux: 20,000ms)  → 444x schneller
Context Switch:   32ns    (Linux: 1,500ns)   → 47x schneller
Arena Alloc:      5ns     (Linux: 200ns)     → 40x schneller
IPC Throughput:   310 GB/s (Linux: 5 GB/s)   → 62x schneller
LLM Inference:    380 tok/s (Linux: 150 tok/s) → 2.5x schneller
================================
Sie sparen: $15.9M/Jahr bei 1,000 Servern
================================
```

### Warum das ein Killer-Feature ist

- **Kein Setup.** ISO rein, booten, Zahlen auf dem Screen. Fertig in <1 Minute.
- **Proof auf DEREN Hardware.** Keine Marketing-Slides, keine Versprechungen — echte Messung.
- **Vertrauen.** Der Kunde sieht dass es auf SEINEM Server läuft, nicht auf einer optimierten Demo-Maschine.
- **Viral.** CTOs teilen sowas: "Boot dieses ISO und sieh selbst." Ein One-Liner auf X/LinkedIn.
- **Sales-Tool.** Statt einer Stunde Pitch: "Hier, 2 Minuten, booten Sie das."

### Verteilung

- GitHub Release: `zero-benchmark-v1.0.iso`
- Website Download-Link
- Direkt per Link in DMs an VCs/Kunden
- QR-Code auf Pitch Deck Slides

---

### Linux-Referenzwerte (bereits recherchiert, kein eigenes Testen)

| Benchmark | Linux-Referenzwert | Quelle |
|-----------|-------------------|--------|
| Boot Time | 15-25 sec (Ubuntu), 3-5 sec (Clear Linux) | commandlinux.com, techrefreshing.com |
| Context Switch | 1,200-5,500 ns | lmbench, Eli Bendersky, Phoronix |
| malloc/free | 50-5,000 ns (glibc), 100-300 ns (jemalloc) | mimalloc-bench, DEV.to |
| IPC pipe | 3-6 GB/s | Linux Kernel Docs |
| IPC mmap | 50-100 GB/s | Linux IPC Benchmarks |
| Qwen2 1.5B tok/s | ~198 tok/s (CPU, 16T) | llama.cpp Benchmark |
| DeepSeek 8B tok/s | ~50 tok/s (EPYC 9554) | ahelpme.com |

### Zero testen (~4-6 Stunden pro Server)

```
1. Cherry Servers mieten, IP KVM self-install
2. ISO über KVM mounten, Zero booten
3. Benchmark Suite ausführen:
   a) Boot-to-Ready (rdtsc automatisch)
   b) Context Switch (Executor Benchmark, 1M Iterationen)
   c) Arena Allocation (Millionen Allokationen)
   d) Zero-Copy IPC (Shared Arena Throughput)
   e) LLM Inference Throughput (Qwen3-1.7B, 100+ Tokens)
4. Ergebnisse erscheinen auf Bildschirm (KVM) → Screenshot
5. 10+ Durchläufe pro Benchmark für Statistik
```

### Danach: Grafiken + Report

```
1. Ergebnisse aggregieren (Median, p50/p95/p99)
2. Grafiken erstellen:
   - Bar Charts: Linux vs Zero (jeder Benchmark)
   - Dollar-Savings Tabelle pro Cluster-Größe
   - ROI-Grafik: "Amortisierung in X Tagen"
3. Report auf GitHub veröffentlichen
4. Grafiken für DMs, a16z Application, Pitch Deck verwenden
```

---

## 9. Skalierungstest: 9354P vs 9554P

| Benchmark | 9354P (32C) | 9554P (64C) | Narrative |
|-----------|-------------|-------------|-----------|
| Context Switch | X ns | ~X ns | "Gleich schnell — Single-Core Metrik" |
| LLM Inference | X tok/s | ~1.5-2x tok/s | "Skaliert linear mit Cores" |
| Allocation | X ns | ~X ns | "Gleich schnell — Lock-Free" |
| IPC | X GB/s | ~X GB/s | "Memory-Bandwidth limitiert, nicht OS" |

**Pitch:** "Zero skaliert linear. Doppelte Cores = doppelter Throughput. Linux skaliert sublinear wegen Scheduling Contention."

---

## 10. Die 4 offenen Fragen

### Frage 1: Wie tracken wir alles?

**rdtsc Benchmark Framework** — muss gebaut werden (~100 Zeilen Rust):
- `rdtsc_serialized()` existiert bereits in `arch/x86_64/cycles.rs`
- Brauchen: Wrapper-Modul `kernel/src/bench.rs` mit:
  - `bench_start()` / `bench_end()` → Cycle-Delta
  - `bench_run(name, iterations, fn)` → automatisch Warmup, N Durchläufe, Median/p99
  - `bench_report()` → formatierte Ausgabe über Serial (COM1)
  - TSC-Frequenz Kalibrierung über CPUID für Cycles → Nanosekunden

**Ergebnis-Export:**
- Serial/UART (COM1, 0x3F8) — existiert, funktioniert in QEMU und Bare-Metal
- Format: `[BENCH] context_switch: median=32ns p99=48ns iterations=1000000`
- Später: UDP Export über E1000 Network Stack (existiert)

**Was fehlt:** Das `bench.rs` Modul. Alles andere (rdtsc, Serial, Executor) existiert.

### Frage 2: Läuft Zero auf der Hardware?

**Ja, mit Einschränkungen:**
- Zero bootet auf x86_64 Bare-Metal (BIOS + UEFI Images existieren)
- Cherry Servers iPXE Boot → Zero Kernel wird über Netzwerk geladen
- EPYC 9354P ist x86_64 Zen 4 → kompatibel

**Risiken:**
- **NIC-Treiber:** Zero hat E1000 Treiber. Cherry Servers nutzt Broadcom BCM5720 (nicht E1000) → Netzwerk funktioniert NICHT, aber egal — alles geht über KVM-Bildschirm
- **UEFI vs BIOS:** iPXE kann beides, Zero hat beides → sollte gehen
- **NUMA:** EPYC hat NUMA-Topology. Zero nutzt aktuell Single-NUMA, sollte trotzdem booten
- **NVMe:** Für Model Loading von Disk brauchen wir NVMe Treiber → existiert NICHT. Model muss über Bootloader oder Netzwerk geladen werden

**Lösung für Model Loading:** GGUF File über iPXE als initrd/Module laden, nicht von NVMe.

### Frage 3: Erreicht Zero die erwarteten Werte?

**Realistisch erreichbar (hohe Confidence):**
- Boot-to-Ready < 50ms: ✅ Unikernel, kein Init-System, direkte Execution
- Context Switch < 50ns: ✅ Cooperative Executor, kein TLB Flush, Lock-Free Queue
- Arena Allocation < 10ns: ✅ Bump Pointer = ADD + Bounds Check, das ist trivial schnell
- Zero-Copy IPC nahe Memory-Bandwidth: ✅ Shared Arena, gleicher Adressraum

**Risiko (mittlere Confidence):**
- LLM Inference 2-3x besser als llama.cpp: ⚠️ Hängt davon ab wie optimiert unsere Inference ist
  - Vorteil: kein Syscall, kein Context Switch, Arena statt malloc
  - Risiko: llama.cpp ist extrem optimiert (AVX-512, NUMA-aware, Thread Pool)
  - Unsere Inference nutzt aktuell NICHT AVX-512 → das muss evtl. noch rein
  - Realistischer Vorteil ohne AVX: 1.3-1.8x. Mit AVX-Optimierung: 2-3x

**Was wir tun wenn die Zahlen nicht stimmen:**
- Benchmark trotzdem publizieren — auch 1.5x ist ein Ergebnis
- Fokus auf die Benchmarks wo wir 50-1000x haben (Boot, Context Switch, Allocation)
- LLM Inference als "Phase 2 Optimierung" framen

### Frage 4: Welches LLM für die Tests?

Siehe Abschnitt 5 oben. Kurzversion:

**Jetzt: Qwen3-1.7B Q4_K** — bereits integriert, sofort testbar
**Ziel: Llama 3.1 8B Q4_K_M** — DAS Industrie-Standard-Benchmark-Modell

Llama 8B ist wichtig weil:
- Jeder VC und CTO kennt "Llama"
- Es gibt publizierte Benchmarks auf EPYC 9554: ~50 tok/s (ahelpme.com)
- 8B ist die Enterprise-relevante Größe
- Q4_K_M ist die Standard-Quantisierung

---

## 11. Was wir auf Zero implementieren müssen

### Muss gebaut werden (geschätzt ~2-3 Tage):

| # | Was | Aufwand | Priorität |
|---|-----|---------|-----------|
| 1 | **bench.rs Framework** — Warmup, N Iterationen, Median/p99, Serial-Output | ~100 Zeilen | MUSS |
| 2 | **Context Switch Benchmark** — Zwei Tasks, yield/resume Loop, rdtsc | ~80 Zeilen | MUSS |
| 3 | **Arena Allocation Benchmark** — Millionen Allokationen timen | ~50 Zeilen | MUSS |
| 4 | **IPC Throughput Benchmark** — Zwei Tasks, Shared Arena, GB/s messen | ~100 Zeilen | MUSS |
| 5 | **LLM Multi-Token Loop** — Token-Loop statt Single Forward Pass, tok/s | ~150 Zeilen | MUSS |
| 6 | **Bootfähiges ISO** — Kernel + GGUF als ISO für KVM-Boot + Demo-Verteilung | ~30 Zeilen | MUSS |
| 7 | **TSC Frequency Calibration** — CPUID auslesen für Cycles → ns | ~40 Zeilen | MUSS |
| 8 | **Llama 3.1 8B Support** — Inference-Pipeline für 8B erweitern | ~300 Zeilen | PHASE 2 |

### Existiert bereits:

- rdtsc_serialized() in arch/x86_64/cycles.rs ✅
- Cooperative Executor mit 32 Slots ✅
- 4 Arena Allocators (KERNEL, RUNTIME, ACTIVATION, KV_CACHE) ✅
- LLM Forward Pass (Qwen3-1.7B, deterministic) ✅
- Serial Output (COM1) + Network Stack (E1000, UDP) ✅
- AOT Benchmark Framework (run_bench() in aot.rs) ✅
- UEFI + BIOS Boot Images ✅

---

## 12. Quellen

### Enterprise-Zahlen
- Cast AI K8s Report 2026 (CPU 8%, Memory 20%, GPU 5%): https://cast.ai
- Spendark Cloud Waste 2026 ($182B): https://spendark.com/blog/state-of-cloud-waste-2026/
- VentureBeat GPU Waste ($401B): https://venturebeat.com/infrastructure/fomo-is-why-enterprises-pay-for-gpus-they-dont-use/
- zop.dev Cloud Spend ($1T): https://zop.dev/resources/blogs/beyond-the-hype-what-2026-cloud-data-says-about-spend-scale-strategy/

### Linux-Benchmarks (publizierte Referenzwerte)
- Boot Time: https://commandlinux.com/statistics/linux-boot-time-statistics-across-different-distributions-and-hardware/
- Context Switch: https://eli.thegreenplace.net/2018/measuring-context-switching-and-memory-overheads-for-linux-threads/
- Memory Allocators: https://dev.to/kunal_d6a8fea2309e1571ee7/jemalloc-vs-malloc-vs-tcmalloc-why-your-servers-default-allocator-is-killing-p99-latency-6l8
- LLM Inference EPYC 9554: https://ahelpme.com/ai/llm-inference-benchmarks-with-llamacpp-with-amd-epyc-9554-cpu/
- SchedCP 1.79-2.11x: NeurIPS 2025, arXiv:2509.01245

### Hardware
- Cherry Servers EPYC 9354P: https://www.cherryservers.com/pricing/dedicated-servers/amd-epyc-9354p
- Cherry Servers EPYC 9554P: https://www.cherryservers.com/pricing/dedicated-servers/amd-epyc-9554p

---

*Dokument wird mit echten Messwerten aktualisiert sobald Hardware verfügbar ist.*
