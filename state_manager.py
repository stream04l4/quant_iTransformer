import threading
import queue

class GlobalState:
    def __init__(self):
        # 1. 基础并发与网络锁
        self.push_queue = queue.Queue()
        self.pos_lock = threading.Lock()
        self.event_lock = threading.Lock()
        
        # 2. 内存账本与网关缓存
        self.portfolio_memory = {'balance': {}, 'positions': [], 'open_stops': [], 'last_sync_ts': 0}
        self.active_pos_map = {}
        self.gate_markets = {}        
        self.dynamic_settings = {}
        self.recent_pushes = {} # {symbol: timestamp}
        self.forced_exit_locks = {} # {symbol: {"anchor_closed_ts": int, "trigger_ts": float, "reason": str}}




# 实例化全局唯一的单例，供其他所有模块调用
state = GlobalState()
