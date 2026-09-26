# NEPHILIM

> **Status — Research-active on-chain detection lab.** NEPHILIM is being narrowed toward evidence-first MEV pattern detection. Its ML classifier remains synthetic-data research and is **not a validated trading signal**. Deterministic detectors are being hardened to fail closed when pair/trace evidence is missing.


[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Web3.py](https://img.shields.io/badge/Web3.py-6.x-F16822?style=flat-square&logo=ethereum&logoColor=white)](https://web3py.readthedocs.io)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](docker-compose.yml)
[![GraphQL](https://img.shields.io/badge/GraphQL-Strawberry-E10098?style=flat-square&logo=graphql&logoColor=white)](https://strawberry.rocks)

> **Evidence-first on-chain MEV detection and entity-research lab for Ethereum-compatible chains.**

---

## The On-Chain Opacity Problem

On-chain activity is observable, but reliable attribution is harder than transaction decoding. A useful research system needs explicit evidence boundaries, reproducible detector behavior, and measurable false-positive/false-negative rates. NEPHILIM explores those primitives for questions such as:

- *Who just sandwiched a user on Uniswap V3?*
- *Is this new wallet cluster the same MEV bot that drained $3M from protocol X last week?*
- *Which addresses are acting as de-facto market makers for this token launch?*

NEPHILIM contains a live-ingestion architecture, deterministic pattern detectors, a synthetic-data ML classifier experiment, graph clustering, storage adapters, and a GraphQL surface. The deterministic detectors are the current validation focus. The ML classifier is not presented as production-validated attribution until it is benchmarked on independently labeled historical chain data.

---

## Detector evidence boundary

The sandwich detector no longer groups swaps by router address. For supported calldata it derives a canonical DEX pair key and preserves token direction. A detection now requires: front-run and victim on the same decoded pair and direction, a backrun in the exact reverse direction, the same attacker on front/back, and the configured gas-ordering evidence. Missing pair or direction evidence fails closed. Reported extracted value remains an explicit heuristic estimate until trace-level profit reconstruction is implemented.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         NEPHILIM Data Pipeline                           │
│                                                                          │
│  Ethereum / Arbitrum                                                     │
│  ┌─────────────┐    WebSocket     ┌───────────────────┐                 │
│  │  Alchemy /  │ ──────────────▶ │  BlockSubscriber  │                 │
│  │  QuickNode  │                  │  (async WS)       │                 │
│  └─────────────┘                  └────────┬──────────┘                 │
│                                            │ full tx receipts           │
│                                   ┌────────▼──────────┐                 │
│                                   │ TransactionDecoder │                 │
│                                   │ DEX ABI decode    │                 │
│                                   │ USD price cache   │                 │
│                                   └────────┬──────────┘                 │
│                                            │ decoded TxRecord           │
│                                   ┌────────▼──────────┐                 │
│                                   │   Kafka Topic:    │                 │
│                                   │  decoded_txs      │                 │
│                                   └──┬─────────────┬──┘                 │
│                                      │             │                    │
│                          ┌───────────▼──┐  ┌───────▼──────────┐        │
│                          │ MEV Classifier│  │  Pattern Detectors│       │
│                          │  (XGBoost)   │  │  Sandwich | JIT   │       │
│                          │  12 types    │  │  (block-level)    │       │
│                          └───────┬──────┘  └──────┬───────────┘        │
│                                  └────────┬────────┘                   │
│                                  ┌────────▼──────────┐                 │
│                                  │  Graph Builder    │                  │
│                                  │  (NetworkX)       │                  │
│                                  └────────┬──────────┘                 │
│                                  ┌────────▼──────────┐                 │
│                                  │  Entity Resolver  │                  │
│                                  │  (Louvain)        │                  │
│                                  └──┬─────────────┬──┘                 │
│                                     │             │                    │
│                          ┌──────────▼──┐  ┌───────▼──────────┐        │
│                          │    Neo4j    │  │   TimescaleDB    │        │
│                          │ Entity Graph│  │  Time-series     │        │
│                          └──────────┬──┘  └──────────────────┘        │
│                                     │                                  │
│                          ┌──────────▼──────────────┐                  │
│                          │  GraphQL API (Strawberry)│                  │
│                          │  + Telegram Alerter      │                  │
│                          └─────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## MEV Type Classification

NEPHILIM's research taxonomy contains **12 MEV categories**. The rule-based detectors cover a narrower subset; the XGBoost path is trained on synthetic labels and should be treated as an experiment rather than a validated classifier:

| # | Type | Description | Key Signal |
|---|------|-------------|------------|
| 1 | `sandwich_attack` | Front-run + victim swap + back-run triple on same token pair | Same block, same pair, gas escalation |
| 2 | `jit_liquidity` | Add liquidity → victim swap → remove liquidity in one block | LP position lifetime = 0 blocks |
| 3 | `cex_dex_arb` | Arbitrage between centralized price feed and AMM pool | Gas spike aligned with CEX feed move |
| 4 | `pure_arb` | Multi-hop on-chain arbitrage closing a price discrepancy | Profit > 0 ETH net, no victim |
| 5 | `liquidation` | Health factor breach triggers collateral seizure | Aave/Compound liquidation event log |
| 6 | `nft_sweep` | Floor sweep of NFT collection using flash loans or MEV | ERC-721 multi-transfer in one tx |
| 7 | `wash_trade` | Self-trade to inflate volume or manipulate price oracle | From/to same cluster, no net value |
| 8 | `bridge` | Cross-chain bridge transaction, often MEV-targeted | Bridge contract interaction |
| 9 | `organic_swap` | Normal retail swap with no MEV characteristics | Low gas, random block position |
| 10 | `flashloan_attack` | Flash loan used for price manipulation or reentrancy | Aave/Balancer flash loan + liquidation |
| 11 | `oracle_manipulation` | Deliberate manipulation of on-chain price oracle | TWAP deviation spike |
| 12 | `governance_attack` | Flash-loan-boosted governance vote or proposal spam | Governance token borrow + vote |

---

## Detector benchmark discipline

NEPHILIM includes a deterministic benchmark harness for labeled sandwich-detector cases:

```bash
pytest tests/test_sandwich_benchmark.py -q
```

The harness reports case-level TP / FP / TN / FN, precision, recall, false-positive rate, and accuracy. The repository currently uses synthetic/adversarial fixtures for regression: canonical sandwich, same-direction attacker traffic, missing direction evidence, equal-gas ordering, wrong-pair traffic, and wrong backrun sender.

A perfect score on this small synthetic corpus is **not** a production accuracy claim. The next validation gate is a versioned historical-chain corpus with independently checked labels, decoded traces, unsupported-router cases, and ambiguous multi-swap transactions.

### Current validation boundary

| Component | Current evidence | Missing before stronger claims |
|---|---|---|
| Pair decoding | Unit-tested V2 path + V3 exactInputSingle | More routers/selectors and trace reconciliation |
| Sandwich detector | Adversarial deterministic fixtures | Historical labeled corpus |
| Extracted value | Explicit heuristic estimate | Trace-level realized-profit reconstruction |
| ETH/USD enrichment | REAL/CACHED/UNAVAILABLE provenance | Dedicated source-quality/freshness policy |
| ML 12-class classifier | Synthetic training data | Real labeled dataset + held-out benchmark |
| Entity clustering | Reproducible graph algorithm | Ground-truth identity evaluation |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Alchemy or QuickNode API key (WebSocket endpoint)

### 1. Clone and configure

```bash
git clone https://github.com/kOs-tile/nephilim.git
cd nephilim
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start infrastructure

```bash
docker-compose up -d
```

This starts Neo4j, TimescaleDB, Redis, and Kafka. The app container waits for all services.

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Train the classifier

```bash
python nephilim/ml/train_classifier.py
# Generates synthetic labeled data and trains XGBoost model
# Saves model to models/mev_classifier.pkl
```

### 5. Seed the entity graph (demo)

```bash
python scripts/seed_graph.py
```

### 6. Run the engine

```bash
python -m nephilim
```

### 7. Replay demo blocks

```bash
python scripts/demo_replay.py
```

---

## GraphQL API

The GraphQL endpoint runs at `http://localhost:8000/graphql`.

### Query a wallet

```graphql
query {
  wallet(address: "0xDead...Beef") {
    address
    label
    cluster {
      id
      behavioralProfile
      totalMevExtractedEth
    }
    mevActivity {
      type
      count
      totalValueEth
    }
  }
}
```

### Get MEV summary for a block range

```graphql
query {
  mevSummary(fromBlock: 19500000, toBlock: 19500100) {
    totalExtractedEth
    byType {
      mevType
      count
      extractedEth
    }
    topActors {
      address
      extractedEth
    }
  }
}
```

### Cluster lookup

```graphql
query {
  cluster(id: "cluster-42") {
    id
    behavioralProfile
    memberCount
    members {
      address
      firstSeen
      lastSeen
    }
    mevActivity {
      type
      count
    }
  }
}
```

### Who touched a contract

```graphql
query {
  whoTouched(contract: "0xUniswapV3Pool...") {
    address
    interactionCount
    lastBlock
    cluster {
      id
      behavioralProfile
    }
  }
}
```

---

## Environment Variables

See `.env.example` for all required configuration. Key variables:

| Variable | Description |
|----------|-------------|
| `ALCHEMY_WS_URL` | Alchemy WebSocket URL (wss://eth-mainnet...) |
| `NEO4J_URI` | Neo4j Bolt URI |
| `TIMESCALE_DSN` | TimescaleDB connection string |
| `KAFKA_BROKERS` | Comma-separated Kafka broker list |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | Target chat ID for alerts |

---

## Project Structure

```
nephilim/
├── nephilim/
│   ├── __init__.py          # Package entry point + main runner
│   ├── config.py            # Pydantic Settings
│   ├── stream/
│   │   ├── block_subscriber.py    # Async WebSocket block ingestion
│   │   └── transaction_decoder.py # DEX ABI decoding + USD pricing
│   ├── classifier/
│   │   ├── mev_classifier.py      # XGBoost 12-type classifier
│   │   ├── sandwich_detector.py   # Block-level sandwich detection
│   │   └── jit_detector.py        # JIT liquidity detection
│   ├── clustering/
│   │   ├── graph_builder.py       # NetworkX transaction graph
│   │   └── entity_resolver.py     # Louvain community detection
│   ├── storage/
│   │   ├── neo4j_client.py        # Entity graph persistence
│   │   └── timescale_client.py    # Time-series MEV metrics
│   ├── alerts/
│   │   └── telegram_alerter.py    # Real-time Telegram notifications
│   ├── api/
│   │   ├── graphql_schema.py      # Strawberry GraphQL schema
│   │   └── server.py              # FastAPI application
│   └── ml/
│       └── train_classifier.py    # XGBoost training pipeline
├── scripts/
│   ├── demo_replay.py             # Historical block replay demo
│   └── seed_graph.py              # Seed Neo4j for demo
├── tests/
│   ├── test_sandwich_detector.py
│   ├── test_louvain_clustering.py
│   └── test_graphql.py
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Author

**Onur Kavi** — AI & Blockchain Systems Portfolio  
GitHub: [@onurkavi](https://github.com/kOs-tile)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
