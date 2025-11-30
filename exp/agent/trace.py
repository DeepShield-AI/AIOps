import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from exp.utils.input import load_parquet_by_hour

logger = logging.getLogger(__name__)


def detect_unbalanced_logs(spans: pd.DataFrame) -> List[Dict]:
    unbalanced = []
    for _, row in spans.iterrows():
        logs = row.get("logs")
        if logs.size == 0 or not isinstance(logs, np.ndarray):
            continue
        types = set()
        for log in logs:
            fields = log.get("fields")
            if isinstance(fields, np.ndarray):
                for f in fields:
                    if f.get("key") == "message.type":
                        types.add(f.get("value"))
        if not {"SENT", "RECEIVED"}.issubset(types):
            unbalanced.append(row.to_dict())
    return unbalanced


def get_operation_name(operation_name: str) -> str:
    return operation_name.removeprefix("hipstershop.").removeprefix("/hipstershop.")


# {trace_id: {spans: {span_id: span}, children: {parent_id: children_id}, roots: [span_id]}}
def build_trace(spans: pd.DataFrame) -> Dict[str, Dict]:
    traces = defaultdict(lambda: {'spans': {}, 'children': defaultdict(list), 'roots': []})
    for _, row in spans.iterrows():
        trace_id = str(row['traceID'])
        span_id: str = str(row['spanID'])
        span = row.to_dict()
        traces[trace_id]['spans'][span_id] = span
        refs = list(span.get('references', []))
        parents = [r['spanID'] for r in refs if r.get('refType') == 'CHILD_OF']
        if parents:
            for p in parents:
                traces[trace_id]['children'][p].append(span_id)
        else:
            traces[trace_id]['roots'].append(span_id)
    return traces


# single trace
def detect_trace_structure_signature(trace: Dict) -> str:
    def dfs(node, children_):
        # operation_name = get_operation_name(spans[node]['operationName'])
        # if node not in children_:
        #     return f"{operation_name}[{spans[node]['kind']}]"
        # return f"{operation_name}[{spans[node]['kind']}]({','.join(sorted([dfs(c, children_) for c in children_[node]]))})"
        op_name = get_operation_name(spans[node]['operationName'])
        kind = spans[node]['kind']
        entry = {
            "operation": op_name,
            "kind": kind
        }
        if node in children_:
            entry["children"] = [dfs(c, children_) for c in sorted(children_[node])]
        return entry

    roots = trace['roots'] or list(trace['spans'].keys())
    spans = trace['spans']
    children = trace['children']
    signatures = [dfs(r, children) for r in roots]
    logger.info(signatures)
    logger.info('|'.join(sorted(signatures)))
    # return '|'.join(sorted(signatures))
    return ""


def group_by_structure(traces: Dict[str, Dict]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for tid, trace in traces.items():
        signature = detect_trace_structure_signature(trace)
        groups[signature].append(tid)
    return groups


def analyze_trace_group_durations(traces: Dict[str, Dict[str, Dict]], trace_ids: List[str], threshold_sigma=3):
    durations = [sum([s['duration'] for s in traces[tid]['spans'].values()]) for tid in trace_ids]
    mean, std = np.mean(durations), np.std(durations)
    if std == 0:
        return []
    return [tid for tid, dur in zip(trace_ids, durations) if (dur - mean) > threshold_sigma * std]


def get_all_children(children_map: Dict) -> set:
    all_children = set()
    for children in children_map.values():
        all_children.update(children)
    return all_children


def detect_self_loops(span_map: Dict, children_map: Dict) -> List[List[str]]:
    visited = set()
    stack = []
    loops = []

    def dfs(span_id):
        if span_id in stack:
            loop_start = stack.index(span_id)
            loops.append(stack[loop_start:] + [span_id])
            return
        if span_id in visited:
            return
        visited.add(span_id)
        stack.append(span_id)
        for child in children_map.get(span_id, []):
            dfs(child)
        stack.pop()

    for root in [s for s in span_map if s not in get_all_children(children_map)]:
        dfs(root)
    return loops


def get_service_name(span_map: Dict[str, Dict[str, Dict]], span_id: str) -> Optional[str]:
    span = span_map.get(span_id)
    if not span:
        return None
    process = span.get('process')
    if not process:
        return None
    return process.get('serviceName')


def detect_service_self_calls(span_map: Dict[str, Dict[str, Dict]], children_map: Dict[str, List[str]]) -> List[str]:
    self_calls = []
    for parent_id, children in children_map.items():
        parent_service = get_service_name(span_map, parent_id)
        for child_id in children:
            child_service = get_service_name(span_map, child_id)
            if parent_service and child_service and parent_service == child_service:
                self_calls.append(parent_service)
    return self_calls


# traceID	e0b937776abecfa2d946dcd4b3f2f2cf
# spanID	c7a0a12ff9b0685e
# operationName	hipstershop.ProductCatalogService/ListProducts
# references	[{'refType': 'CHILD_OF', 'spanID': 'a97c04e2e6c86766', 'traceID': 'e0b937776abecfa2d946dcd4b3f2f2cf'}]
# startTimeMillis	1749142862303
# duration	65
# TODO: use status.code, rpc.method
# Does ip and peer need to be used?
# tags
# [{'key': 'rpc.system', 'type': 'string', 'value': 'grpc'}
#  {'key': 'rpc.service', 'type': 'string', 'value': 'hipstershop.ProductCatalogService'}
#  {'key': 'rpc.method', 'type': 'string', 'value': 'ListProducts'}
#  {'key': 'net.peer.ip', 'type': 'string', 'value': '10.233.77.230'}
#  {'key': 'net.peer.port', 'type': 'string', 'value': '33572'}
#  {'key': 'instrumentation.name', 'type': 'string', 'value': 'go.opentelemetry.io/otel/sdk/tracer'}
#  {'key': 'status.code', 'type': 'int64', 'value': '0'}
#  {'key': 'status.message', 'type': 'string', 'value': ''}
#  {'key': 'span.kind', 'type': 'string', 'value': 'server'}
#  {'key': 'internal.span.format', 'type': 'string', 'value': 'jaeger'}]
# TODO: how to use logs in span message
# logs
# [{'fields': array([{'key': 'message.type', 'type': 'string', 'value': 'RECEIVED'}, {'key': 'message.id', 'type': 'int64', 'value': '1'},{'key': 'message.uncompressed_size', 'type': 'int64', 'value': '0'}, {'key': 'name', 'type': 'string', 'value': 'message'}],dtype=object), 'timestamp': 1749142862303896}
# {'fields': array([{'key': 'message.type', 'type': 'string', 'value': 'SENT'}, {'key': 'message.id', 'type': 'int64', 'value': '1'}, {'key': 'message.uncompressed_size', 'type': 'int64', 'value': '2541'}, {'key': 'name', 'type': 'string', 'value': 'message'}], dtype=object), 'timestamp': 1749142862303934}                           ]
# TODO: use tags (name -> pod, node_name -> node, namespace -> namespace)
# Does ip need to be used?
# process
# {
# 'serviceName': 'productcatalogservice',
# 'tags': array([
#     {'key': 'exporter', 'type': 'string', 'value': 'jaeger'},
#     {'key': 'float', 'type': 'float64', 'value': '312.23'},
#     {'key': 'ip', 'type': 'string', 'value': '10.233.79.154'},
#     {'key': 'name', 'type': 'string', 'value': 'productcatalogservice-1'},
#     {'key': 'node_name', 'type': 'string', 'value': 'aiops-k8s-06'},
#     {'key': 'namespace', 'type': 'string', 'value': 'hipstershop'}
#     ],
#     dtype=object)}

class TraceAgent:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.fields = [
            "traceID", "spanID", "operationName", "references", "startTimeMillis", "duration", "tags", "logs",
            "process"]
        self.analysis_fields = [
            "traceID", "spanID", "operationName", "references", "start", "end", "duration", "tags", "logs", "namespace",
            "node", "pod", 'kind', 'code', 'process']

    def load_spans(self, start: datetime, end: datetime, max_workers=4):
        def callback(spans: pd.DataFrame) -> pd.DataFrame:
            def parse_process(process: Dict) -> pd.Series:
                t = {}
                tags = process.get('tags')
                if isinstance(tags, np.ndarray):
                    for tag in tags:
                        if isinstance(tag, dict) and "key" in tag and "value" in tag:
                            key = tag["key"]
                            if key in ("node_name", "namespace", "name"):
                                t[key] = tag["value"]
                return pd.Series([
                    t.get('node_name'),
                    t.get('namespace'),
                    t.get('name'),
                ])

            def parse_tags(tags: np.ndarray) -> pd.Series:
                t = {}
                for tag in tags:
                    if isinstance(tag, dict) and "key" in tag and "value" in tag:
                        key = tag["key"]
                        if key in ("span.kind", "status.code", "grpc.status_code"):
                            t[key] = tag["value"]

                return pd.Series([
                    str(t.get('span.kind')).lower(),
                    t.get("status.code") or t.get("grpc.status_code") or '0'
                ])

            spans['start'] = pd.to_datetime(spans["startTimeMillis"], unit="ms")
            spans['end'] = spans['start'] + pd.to_timedelta(spans['duration'], unit='ms')
            spans[['node', 'namespace', 'pod']] = spans['process'].apply(parse_process)
            spans[['kind', 'code']] = spans['tags'].apply(parse_tags)
            return spans

        return load_parquet_by_hour(
            start, end, self.root_path,
            file_pattern="{dataset}/{day}/trace-parquet/trace_jaeger-span_{day}_{hour}-00-00.parquet",
            load_fields=self.fields,
            return_fields=self.analysis_fields,
            filter_=None,
            callback=callback,
            max_workers=max_workers
        )

    def score(self, start_time: datetime, end_time: datetime, max_workers=4) -> Dict:
        """
        score method, generate analysis results.
        """
        all_spans = self.load_spans(start_time, end_time, max_workers=max_workers)
        if all_spans.empty:
            logger.warning(f"Didn't find any spans between {start_time} and {end_time}.")
            return []
        
        op_durations = all_spans.groupby('operationName')['duration']
        op_thresholds = (op_durations.mean() + 3 * op_durations.std()).to_dict()

        anomalous_links = defaultdict(lambda: {
            'count': 0,
            'error_codes': set(),
            'latent_durations': [],
            'latent_thresholds': [],
            'source_pods': set(),
            'target_pods': set()
        })

        for trace_id, group in all_spans.groupby('traceID'):
            trace_structure = build_trace(group)[trace_id]
            spans_map = trace_structure['spans']
            children_map = trace_structure['children']

            for parent_id, children_ids in children_map.items():
                parent_span = spans_map.get(parent_id)
                if not parent_span or parent_span.get('kind') != 'client':
                    continue

                for child_id in children_ids:
                    child_span = spans_map.get(child_id)
                    if not child_span or child_span.get('kind') != 'server':
                        continue
                    
                    is_error_code = child_span.get('code', '0') != '0'
                    
                    threshold = op_thresholds.get(parent_span['operationName'], float('inf'))
                    is_high_latency = parent_span['duration'] > (threshold * 1.05) # 增加5%容忍度

                    if is_error_code or is_high_latency:
                        source_service = parent_span['process']['serviceName']
                        source_pod = parent_span.get('pod', source_service)
                        target_service = child_span['process']['serviceName']
                        target_pod = child_span.get('pod', target_service)
                        
                        link_key = (source_service, target_service)
                        
                        anomalous_links[link_key]['count'] += 1
                        anomalous_links[link_key]['source_pods'].add(source_pod)
                        anomalous_links[link_key]['target_pods'].add(target_pod)

                        if is_error_code:
                            anomalous_links[link_key]['error_codes'].add(child_span.get('code'))
                        if is_high_latency:
                            anomalous_links[link_key]['latent_durations'].append(parent_span['duration'])
                            anomalous_links[link_key]['latent_thresholds'].append(threshold)

        if not anomalous_links:
            return []

        all_results = []
        for (source, target), data in anomalous_links.items():
            valid_source_pods = [p for p in data['source_pods'] if p]
            valid_target_pods = [p for p in data['target_pods'] if p]
            # source_pods_str = ", ".join(sorted(valid_source_pods))
            # target_pods_str = ", ".join(sorted(valid_target_pods))
            source_pods_list = sorted(valid_source_pods)
            target_pods_list = sorted(valid_target_pods)

             # span_str = f"[span]: {source}({source_pods_str}) -> {target}({target_pods_str}), [count]: {data['count']}"
            span_obj = {
                "source": f"{source}",
                "source_pods": source_pods_list,
                "target": f"{target}",
                "target_pods": target_pods_list,
                "count": data["count"],
            }
            
            # message_parts = []
            message_parts = {}
            if data['error_codes']:
                # message_parts.append(f"errors with code: {sorted(list(data['error_codes']))}")
                message_parts["error_codes"] = sorted(list(data['error_codes']))

            if data['latent_durations']:
                avg_latency_ms = np.mean(data['latent_durations']) / 1000
                max_latency_ms = np.max(data['latent_durations']) / 1000
                avg_threshold_ms = np.mean(data['latent_thresholds']) / 1000
                # message_parts.append(f"high latency (avg: {avg_latency_ms:.2f}ms, max: {max_latency_ms:.2f}ms, threshold ~{avg_threshold_ms:.2f}ms)")
                message_parts["latency"] = {
                    "avg_latency_ms": avg_latency_ms,
                    "max_latency_ms": max_latency_ms,
                    "avg_threshold_ms": avg_threshold_ms,
                }

            # if message_parts:
            #     message_str = f"[message]: {'; '.join(message_parts)}"
            # else:
            #      message_str = "[message]: Anomalous call detected with no specific error message."
            
            # formatted_results.append(span_str)
            # formatted_results.append(message_str)
            # formatted_results.append("-" * 20)
            formatted_results = {
                "span": span_obj,
                "message": message_parts
            }
            all_results.append(formatted_results)

        return all_results
# The span tag can be classified as
# [{'key': 'otel.library.name', 'type': 'string', 'value': 'OpenTelemetry.Instrumentation.StackExchangeRedis'}
#  {'key': 'otel.library.version', 'type': 'string', 'value': '1.0.0.10'}
#  {'key': 'db.system', 'type': 'string', 'value': 'redis'}
#  {'key': 'db.redis.flags', 'type': 'string', 'value': 'DemandMaster'}
#  {'key': 'db.statement', 'type': 'string', 'value': 'HMSET'}
#  {'key': 'net.peer.name', 'type': 'string', 'value': 'redis-cart'}
#  {'key': 'net.peer.port', 'type': 'int64', 'value': '6379'}
#  {'key': 'db.redis.database_index', 'type': 'int64', 'value': '0'}
#  {'key': 'peer.service', 'type': 'string', 'value': 'redis-cart:6379'}
#  {'key': 'span.kind', 'type': 'string', 'value': 'client'}
#  {'key': 'internal.span.format', 'type': 'string', 'value': 'otlp'}]

# [{'key': 'otel.library.name', 'type': 'string', 'value': 'OpenTelemetry.Instrumentation.AspNetCore'}
# {'key': 'otel.library.version', 'type': 'string', 'value': '1.0.0.0'}
# {'key': 'server.address', 'type': 'string', 'value': 'cartservice'}
# {'key': 'server.port', 'type': 'int64', 'value': '7070'}
# {'key': 'http.request.method', 'type': 'string', 'value': 'POST'}
# {'key': 'url.scheme', 'type': 'string', 'value': 'http'}
# {'key': 'url.path', 'type': 'string', 'value': '/hipstershop.CartService/GetCart'}
# {'key': 'network.protocol.version', 'type': 'string', 'value': '2'}
# {'key': 'user_agent.original', 'type': 'string', 'value': 'grpc-go/1.31.0'}
# {'key': 'grpc.method', 'type': 'string', 'value': '/hipstershop.CartService/GetCart'}
# {'key': 'grpc.status_code', 'type': 'string', 'value': '0'}
# {'key': 'http.route', 'type': 'string', 'value': '/hipstershop.CartService/GetCart'}
# {'key': 'http.response.status_code', 'type': 'int64', 'value': '200'}
# {'key': 'span.kind', 'type': 'string', 'value': 'server'}
# {'key': 'internal.span.format', 'type': 'string', 'value': 'otlp'}]

# [{'key': 'rpc.system', 'type': 'string', 'value': 'grpc'}
# {'key': 'rpc.service', 'type': 'string', 'value': 'hipstershop.RecommendationService'}
# {'key': 'rpc.method', 'type': 'string', 'value': 'ListRecommendations'}
# {'key': 'net.peer.ip', 'type': 'string', 'value': 'recommendationservice'}
# {'key': 'net.peer.port', 'type': 'string', 'value': '8080'}
# {'key': 'instrumentation.name', 'type': 'string', 'value': 'go.opentelemetry.io/otel/sdk/tracer'}
# {'key': 'status.code', 'type': 'int64', 'value': '0'}
# {'key': 'status.message', 'type': 'string', 'value': ''}
# {'key': 'span.kind', 'type': 'string', 'value': 'client'}
# {'key': 'internal.span.format', 'type': 'string', 'value': 'jaeger'}]
