"""
network_spatial_features.py

从 SUMO 网络文件中提取拓扑，计算每条边的上下游邻居，
然后为 edgedata 增强空间特征（上下游均值）。
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, Set, Tuple, List


class NetworkTopology:
    """解析 SUMO 网络，构建边的邻接关系"""
    
    def __init__(self, net_file: str):
        self.net_file = net_file
        self.edges_to = defaultdict(list)  # edge_id -> [out_edges]
        self.edges_from = defaultdict(list)  # edge_id -> [in_edges]
        self.edge_info = {}  # edge_id -> {'from': jid, 'to': jid, ...}
        self._parse_network()
    
    def _parse_network(self):
        """解析 SUMO .net.xml 文件"""
        print(f"[INFO] Parsing network: {self.net_file}")
        tree = ET.parse(self.net_file)
        root = tree.getroot()
        
        # 先收集所有边及其连接的路口
        for edge in root.findall('.//edge'):
            edge_id = edge.get('id')
            from_jid = edge.get('from')
            to_jid = edge.get('to')
            
            # 跳过内部边（交叉口内的边）
            if edge_id.startswith(':'):
                continue
            
            self.edge_info[edge_id] = {
                'from': from_jid,
                'to': to_jid
            }
        
        # 根据拓扑关系建立邻接表
        for edge_id, info in self.edge_info.items():
            to_jid = info['to']
            
            # 找所有从 to_jid 出发的边（下游）
            for other_id, other_info in self.edge_info.items():
                if other_info['from'] == to_jid:
                    self.edges_to[edge_id].append(other_id)
            
            # 找所有连接到 from_jid 的边（上游）
            from_jid = info['from']
            for other_id, other_info in self.edge_info.items():
                if other_info['to'] == from_jid:
                    self.edges_from[edge_id].append(other_id)
        
        print(f"[INFO] Loaded {len(self.edge_info)} edges")
        
        # 统计
        edges_with_upstream = sum(1 for e in self.edges_from.values() if len(e) > 0)
        edges_with_downstream = sum(1 for e in self.edges_to.values() if len(e) > 0)
        print(f"[INFO] Edges with upstream neighbors: {edges_with_upstream}")
        print(f"[INFO] Edges with downstream neighbors: {edges_with_downstream}")
    
    def get_neighbors(self, edge_id: str) -> Tuple[List[str], List[str]]:
        """获得某条边的上下游邻居"""
        upstream = self.edges_from.get(edge_id, [])
        downstream = self.edges_to.get(edge_id, [])
        return upstream, downstream


def aggregate_spatial_features(
    edgedata_dict: Dict,
    edge_list: List[str],
    topology: NetworkTopology,
    target_features: List[str] = None
) -> Dict:
    """
    为每条边增加上下游聚合特征
    
    Args:
        edgedata_dict: {edge_id: [(t, {feat_name: val}), ...]}
        edge_list: 所有边的 ID
        topology: NetworkTopology 对象
        target_features: 要聚合的特征名，默认 ['speed', 'density']
    
    Returns:
        增强后的 edgedata_dict
    """
    
    if target_features is None:
        target_features = ['speed', 'density']
    
    print(f"[INFO] Augmenting spatial features for {len(edge_list)} edges")
    print(f"[INFO] Aggregating features: {target_features}")
    
    edge_set = set(edge_list)
    
    for edge_id, timeseries in edgedata_dict.items():
        # 获得邻居
        upstream_ids, downstream_ids = topology.get_neighbors(edge_id)
        
        # 过滤存在的邻居（可能有些邻居在 edgedata 中不存在）
        upstream_ids = [e for e in upstream_ids if e in edge_set]
        downstream_ids = [e for e in downstream_ids if e in edge_set]
        
        if not upstream_ids and not downstream_ids:
            continue
        
        # 为每个时间步添加聚合特征
        for t, features_dict in enumerate(timeseries):
            # 上游聚合
            for feat_name in target_features:
                upstream_vals = []
                for up_id in upstream_ids:
                    if up_id in edgedata_dict and t < len(edgedata_dict[up_id]):
                        val = float(edgedata_dict[up_id][t].get(feat_name, 0))
                        upstream_vals.append(val)
                
                if upstream_vals:
                    features_dict[f'{feat_name}_upstream_mean'] = np.mean(upstream_vals)
                else:
                    features_dict[f'{feat_name}_upstream_mean'] = 0.0
            
            # 下游聚合
            for feat_name in target_features:
                downstream_vals = []
                for dn_id in downstream_ids:
                    if dn_id in edgedata_dict and t < len(edgedata_dict[dn_id]):
                        val = float(edgedata_dict[dn_id][t].get(feat_name, 0))
                        downstream_vals.append(val)
                
                if downstream_vals:
                    features_dict[f'{feat_name}_downstream_mean'] = np.mean(downstream_vals)
                else:
                    features_dict[f'{feat_name}_downstream_mean'] = 0.0
    
    print(f"[INFO] Spatial feature augmentation completed")
    return edgedata_dict


if __name__ == '__main__':
    # 测试
    net_file = r'D:\pro_and_data\SCM_DeepONet_code\net\shanghai_5km.net.xml'
    topo = NetworkTopology(net_file)
    
    # 示例：查看前 5 条边的邻居
    for i, edge_id in enumerate(list(topo.edge_info.keys())[:5]):
        upstream, downstream = topo.get_neighbors(edge_id)
        print(f"\nEdge {edge_id}:")
        print(f"  Upstream: {upstream}")
        print(f"  Downstream: {downstream}")
