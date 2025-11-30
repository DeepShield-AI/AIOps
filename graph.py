import json
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Span:
    source: str
    source_pods: List[str]
    target: str
    target_pods: List[str]
    count: int
    avg_latency: Optional[float]
    max_latency: Optional[float]
    avg_threshold: Optional[float]
    error_codes: Optional[List[str]] = None

    @staticmethod
    def from_json(obj: Dict[str, Any]):
        span = obj.get("span", {})
        msg = obj.get("message", {})
        latency = msg.get("latency", {})

        return Span(
            source=span.get("source"),
            source_pods=span.get("source_pods", []),
            target=span.get("target"),
            target_pods=span.get("target_pods", []),
            count=span.get("count", 0),
            avg_latency=latency.get("avg_latency_ms"),
            max_latency=latency.get("max_latency_ms"),
            avg_threshold=latency.get("avg_threshold_ms"),
            error_codes=msg.get("error_codes"),
        )

def load_spans(path: str) -> List[Span]:
    spans = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "span" in obj:
                spans.append(Span.from_json(obj))
    return spans

def build_graph(spans: List[Span]) -> nx.DiGraph:
    g = nx.DiGraph()

    for s in spans:
        if not g.has_node(s.source):
            g.add_node(s.source, pods=set(s.source_pods))
        else:
            g.nodes[s.source]["pods"].update(s.source_pods)

        if not g.has_node(s.target):
            g.add_node(s.target, pods=set(s.target_pods))
        else:
            g.nodes[s.target]["pods"].update(s.target_pods)

        record = {
            "count": s.count,
            "avg_latency": s.avg_latency,
            "max_latency": s.max_latency,
            "avg_threshold": s.avg_threshold,
            "error_codes": s.error_codes,
            "source_pods": s.source_pods,
            "target_pods": s.target_pods,
        }

        if not g.has_edge(s.source, s.target):
            g.add_edge(s.source, s.target, records=[record])
        else:
            g[s.source][s.target]["records"].append(record)

    for n in g.nodes:
        g.nodes[n]["pods"] = list(g.nodes[n]["pods"])

    return g


def print_graph_info(g: nx.DiGraph):
    print("\nNodes:")
    for n, data in g.nodes(data=True):
        print(f"  {n}: pods={data['pods']}")

    print("\nEdges:")
    for u, v, data in g.edges(data=True):
        print(f"  {u} -> {v}")
        for idx, rec in enumerate(data["records"], 1):
            print(f"    record {idx}: {rec}")

# def visualize_graph(g: nx.DiGraph):
#     plt.figure(figsize=(12, 8))

#     pos = nx.spring_layout(g, seed=42)

#     nx.draw(
#         g,
#         pos,
#         with_labels=True,
#         node_size=2500,
#         font_size=10,
#         arrows=True
#     )

#     edge_labels = {}
#     for u, v, data in g.edges(data=True):
#         records = data["records"]
#         total_count = sum(rec["count"] for rec in records)
#         span_count = len(records)

#         edge_labels[(u, v)] = f"total_cnt={total_count}, span_cnt={span_count}"

#     nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

#     plt.title("Service Call Graph (Aggregated)")
#     plt.tight_layout()
#     plt.savefig("service_call_graph.png")

def visualize_graph(g: nx.DiGraph):
    plt.figure(figsize=(14, 9))

    pos = nx.spring_layout(g, seed=42)

    node_labels = {}
    for n, data in g.nodes(data=True):
        pods = data.get("pods", [])
        node_labels[n] = f"{n}\npods: {','.join(pods)}"

    nx.draw_networkx_nodes(
        g, pos,
        node_size=2600,
        node_color="#A8D1FF"
    )

    nx.draw_networkx_labels(
        g, pos,
        labels=node_labels,
        font_size=9
    )

    nx.draw_networkx_edges(
        g, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        width=1.8
    )

    edge_labels = {}
    for u, v, data in g.edges(data=True):
        records = data["records"]
        total_count = sum(rec["count"] for rec in records)
        span_count = len(records)
        edge_labels[(u, v)] = f"count={total_count}\nspans={span_count}"

    nx.draw_networkx_edge_labels(
        g, pos,
        edge_labels=edge_labels,
        font_size=8
    )

    plt.title("Service Call Graph with Pods and Aggregated Edge Metrics")
    plt.tight_layout()
    plt.savefig("service_call_graph.png")

def main():
    TRACE_FILE = "trace_analysis_report.jsonl"
    LOG_FILE = "log_analysis_report.jsonl"
    spans = load_spans(TRACE_FILE)
    print(f"Loaded {len(spans)} spans.")

    g = build_graph(spans)

    print_graph_info(g)

    visualize_graph(g)

if __name__ == "__main__":
    main()
else:
    print("Please run this script directly.")
