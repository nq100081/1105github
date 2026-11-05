# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🌍  智 能 旅 行 规 划 系 统                            ║
║                 Intelligent Travel Planning System                        ║
║                                                                            ║
║  架构: Flask + 多Agent协作 + SSE流式 + SQLite                             ║
║  启动: python app.py  →  http://localhost:5000                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import re
import json
import time
import random
import sqlite3
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from typing import Callable, Generator, Optional

from flask import Flask, request, jsonify, g, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

# ── 加载 .env ────────────────────────────────────────────────────────────
load_dotenv()

# ── 可选依赖 ──────────────────────────────────────────────────────────────
try:
    import requests as http_requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          ⚙️  配 置 区                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE = BASE_DIR / "travel_planner.db"

# LLM 配置
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

# 高德地图 API（可选）
AMAP_API_KEY = os.getenv("AMAP_API_KEY", "")

# 服务端口
PORT = int(os.getenv("PORT", 5000))

# 预算等级映射
BUDGET_LEVELS = {
    "budget": "经济型 (人均<2000元)",
    "standard": "舒适型 (人均2000-5000元)",
    "luxury": "豪华型 (人均>5000元)",
}

# 预算等级 → 每日人均花费估算
BUDGET_DAILY = {
    "budget": {"transport": 50, "hotel": 150, "food": 80, "tickets": 60, "other": 40},
    "standard": {"transport": 100, "hotel": 350, "food": 180, "tickets": 120, "other": 80},
    "luxury": {"transport": 200, "hotel": 800, "food": 400, "tickets": 200, "other": 200},
}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     🗄️  数 据 库 初 始 化                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DATABASE))
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db

def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(str(DATABASE))
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE,
            password_hash TEXT  NOT NULL DEFAULT '',
            avatar      TEXT    DEFAULT '🧳',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS trips (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL DEFAULT 1,
            destination     TEXT    NOT NULL,
            days            INTEGER NOT NULL DEFAULT 3,
            budget          TEXT    NOT NULL DEFAULT 'standard',
            preferences     TEXT    DEFAULT '',
            companions      TEXT    DEFAULT '独自',
            itinerary_json  TEXT    NOT NULL DEFAULT '{}',
            budget_json     TEXT    DEFAULT '{}',
            weather_json    TEXT    DEFAULT '[]',
            status          TEXT    DEFAULT 'completed',
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL DEFAULT 1,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            trip_id     INTEGER,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

    """)
    # 兼容旧库：添加缺失的列
    for col, col_def in [("avatar", "TEXT DEFAULT '🧳'"), ("password_hash", "TEXT DEFAULT ''")]:
        try:
            db.execute(f"ALTER TABLE users ADD COLUMN {col} {col_def}")
        except:
            pass
    # 创建默认用户（密码为空的旧用户需重新设置密码）
    db.execute("INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (1, '旅行达人', '')")
    db.commit()
    db.close()
    print("[DB] 数据库初始化完成")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   🤖  LLM  客 户 端 封 装                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def call_llm(messages: list, temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
    """
    统一的 LLM 调用接口（OpenAI 兼容协议）
    支持: OpenAI / 通义千问 / DeepSeek / 任意兼容端点
    无 API Key 时返回 None → 触发降级
    """
    if not LLM_API_KEY or not HAS_REQUESTS:
        return None

    try:
        resp = http_requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"[LLM] API 错误 {resp.status_code}: {resp.text[:300]}")
            return None
    except Exception as e:
        print(f"[LLM] 调用异常: {e}")
        return None


def has_llm() -> bool:
    """检查 LLM 是否可用"""
    return bool(LLM_API_KEY) and HAS_REQUESTS

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     📦  数 据 加 载 层                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_attractions_data = None
def load_attractions() -> dict:
    """加载景点/酒店模拟数据"""
    global _attractions_data
    if _attractions_data is None:
        path = DATA_DIR / "attractions.json"
        with open(path, "r", encoding="utf-8") as f:
            _attractions_data = json.load(f)
    return _attractions_data


def get_cities() -> list:
    """获取所有支持的城市列表"""
    data = load_attractions()
    return [
        {"name": name, "province": info["province"], "count": len(info["attractions"])}
        for name, info in data["cities"].items()
    ]


def get_attractions(city: str, preferences: str = "") -> list:
    """获取指定城市的景点列表（优先本地精选数据，否则高德POI搜索）"""
    # 优先使用本地精选数据
    data = load_attractions()
    city_data = data["cities"].get(city)
    if city_data:
        attractions = city_data["attractions"].copy()
        if preferences:
            prefs = [p.strip() for p in preferences.replace("，", ",").split(",")]
            def score(a):
                s = 0
                for p in prefs:
                    if p in a["category"] or p in a["name"]:
                        s += 2
                    if p in a.get("tips", ""):
                        s += 1
                return -s
            attractions.sort(key=score)
        return attractions

    # 本地没有该城市 → 高德 POI 兜底
    if AMAP_API_KEY and HAS_REQUESTS:
        amap_result = _amap_get_attractions(city, preferences)
        if amap_result:
            return amap_result

    return []


def get_hotels(budget_level: str, city: str = "") -> list:
    """根据预算等级获取酒店列表（优先高德POI，否则本地数据）"""
    if city and AMAP_API_KEY and HAS_REQUESTS:
        amap_hotels = _amap_get_hotels(city, budget_level)
        if amap_hotels:
            return amap_hotels

    data = load_attractions()
    return data["hotels"].get(budget_level, data["hotels"]["standard"])


# ── 高德 POI 搜索 ─────────────────────────────────────────────────────────

def _amap_poi_search(city: str, keywords: str, types: str = "", offset: int = 20) -> list:
    """高德 POI 搜索通用函数"""
    try:
        resp = http_requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={
                "key": AMAP_API_KEY,
                "keywords": keywords,
                "city": city,
                "types": types,
                "offset": offset,
                "extensions": "all",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "1" and data.get("pois"):
                return data["pois"]
    except Exception as e:
        print(f"[Amap] POI搜索异常: {e}")
    return []


def _amap_get_attractions(city: str, preferences: str = "") -> list:
    """通过高德 POI 搜索景点，返回与本地数据相同格式的列表"""
    # 根据偏好调整搜索关键词
    keywords = "景点|公园|博物馆|古镇|寺庙"
    if preferences:
        pref_keywords = {"美食": "美食街|夜市", "历史文化": "古迹|博物馆|寺庙",
                         "自然风光": "山|湖|公园|峡谷", "拍照打卡": "网红|打卡|文创",
                         "亲子": "游乐园|动物园|海洋馆", "购物": "步行街|商圈",
                         "探险户外": "登山|漂流|徒步"}
        for k, v in pref_keywords.items():
            if k in preferences:
                keywords = v + "|" + keywords

    pois = _amap_poi_search(city, keywords, types="110000|140000|060000", offset=15)
    if not pois:
        return []

    attractions = []
    for p in pois:
        biz = p.get("biz_ext", {}) or {}
        rating = float(biz.get("rating", "0")) if biz.get("rating") else 0
        if not rating:
            rating = round(3.5 + (hash(p.get("name", "")) % 50) / 10, 1)  # fallback rating
        cost = biz.get("cost", "") if biz else ""

        attractions.append({
            "name": p.get("name", ""),
            "category": p.get("type", "").split(";")[0] if p.get("type") else "景点",
            "address": p.get("address", ""),
            "ticket": int(cost) if cost and cost.isdigit() else 0,
            "duration": "2-3小时",
            "rating": rating,
            "tips": f'位于{p.get("adname", "")}，{p.get("business_area", "") if p.get("business_area") else "交通便利"}',
        })

    # 去重 + 按评分排序
    seen = set()
    unique = []
    for a in attractions:
        if a["name"] not in seen:
            seen.add(a["name"])
            unique.append(a)
    unique.sort(key=lambda x: x["rating"], reverse=True)

    # 按偏好加权排序
    if preferences:
        prefs = [p.strip() for p in preferences.replace("，", ",").split(",")]
        def score(a):
            s = 0
            for p in prefs:
                if p in a["category"] or p in a["name"]:
                    s += 2
                if p in a.get("tips", ""):
                    s += 1
            return -s
        unique.sort(key=score)

    return unique


def _amap_get_hotels(city: str, budget_level: str) -> list:
    """通过高德 POI 搜索酒店，返回与本地数据相同格式的列表"""
    # 统一搜"酒店"，再由价格筛选，避免关键词过窄
    pois = _amap_poi_search(city, "酒店|宾馆|民宿", types="100000", offset=15)
    if not pois:
        return []

    price_range = {"budget": (0, 250), "standard": (250, 550), "luxury": (550, 9999)}
    lo, hi = price_range.get(budget_level, (200, 500))
    level_name = {"budget": "经济型", "standard": "舒适型", "luxury": "豪华型"}.get(budget_level, "舒适型")

    hotels = []
    for p in pois:
        biz = p.get("biz_ext", {}) or {}
        rating = float(biz.get("rating", "0")) if biz.get("rating") else 0
        cost = biz.get("cost", "") if biz else ""
        price = int(cost) if cost and cost.isdigit() else 0

        # 按价格区间筛选；无价格信息的随机分配
        if price == 0:
            price = random.randint(lo, hi)
        elif not (lo <= price <= hi):
            continue

        hotels.append({
            "name": p.get("name", ""),
            "price_per_night": price,
            "level": level_name,
            "amenities": "WiFi/空调/24小时热水",
            "address": p.get("address", ""),
            "rating": rating if rating else round(3.5 + (hash(p.get("name", "")) % 40) / 10, 1),
        })

    seen = set()
    unique = []
    for h in hotels:
        if h["name"] not in seen:
            seen.add(h["name"])
            unique.append(h)
    # 如果筛选后少于2家，兜底：无价格信息的也加入
    if len(unique) < 2:
        for p in pois:
            if len(unique) >= 6:
                break
            biz = p.get("biz_ext", {}) or {}
            name = p.get("name", "")
            if name in seen:
                continue
            rating = float(biz.get("rating", "0")) if biz.get("rating") else round(3.5 + (hash(name) % 40) / 10, 1)
            unique.append({
                "name": name,
                "price_per_night": random.randint(lo, hi),
                "level": level_name,
                "amenities": "WiFi/空调/24小时热水",
                "address": p.get("address", ""),
                "rating": rating,
            })

    unique.sort(key=lambda x: x["rating"], reverse=True)
    return unique[:6]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   🎯  多 Agent 协 作 架 构                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TravelAgent:
    """Agent 基类"""
    def __init__(self, name: str, icon: str, description: str):
        self.name = name
        self.icon = icon
        self.description = description

    def emit(self, status: str, message: str, data: dict = None):
        """发送进度事件到回调队列"""
        if hasattr(self, "_callback") and self._callback:
            self._callback({
                "agent": self.name,
                "icon": self.icon,
                "status": status,
                "message": message,
                "data": data or {},
            })

    def run(self, context: dict, callback: Callable = None) -> dict:
        """运行 Agent，子类重写"""
        self._callback = callback
        return {}


class IntentAgent(TravelAgent):
    """用户意图Agent：解析自然语言输入，提取结构化旅行参数"""

    def __init__(self):
        super().__init__("IntentAgent", "🎯", "理解用户需求，提取旅行参数")

    def run(self, context: dict, callback: Callable = None) -> dict:
        super().run(context, callback)
        self.emit("running", "🔍 正在分析您的旅行需求...")

        raw_input = context.get("raw_input", "")
        # 如果有表单数据，直接使用
        destination = context.get("destination", "")
        days = context.get("days", 3)
        budget = context.get("budget", "standard")
        preferences = context.get("preferences", "")
        companions = context.get("companions", "独自")

        # 如果从原始输入中提取参数
        if raw_input and not destination:
            extracted = self._extract_from_text(raw_input)
            destination = extracted.get("destination", destination)
            days = extracted.get("days", days)
            budget = extracted.get("budget", budget)
            preferences = extracted.get("preferences", preferences)
            companions = extracted.get("companions", companions)

        # 如果有 LLM，进行更精准的意图识别
        if raw_input and has_llm():
            llm_extracted = self._llm_extract(raw_input)
            if llm_extracted:
                destination = llm_extracted.get("destination") or destination
                days = llm_extracted.get("days") or days
                budget = llm_extracted.get("budget") or budget
                preferences = llm_extracted.get("preferences") or preferences
                companions = llm_extracted.get("companions") or companions

        # 标准化 days
        try:
            days = int(days)
        except (ValueError, TypeError):
            days = 3
        days = max(1, min(days, 30))

        # 标准化 budget
        if budget not in BUDGET_LEVELS:
            budget = "standard"

        result = {
            "destination": destination,
            "days": days,
            "budget": budget,
            "budget_label": BUDGET_LEVELS.get(budget, ""),
            "preferences": preferences,
            "companions": companions,
        }

        self.emit("done",
            f"✅ 需求分析完成：{destination}，{days}天，{BUDGET_LEVELS.get(budget, '')}，偏好：{preferences or '无'}",
            result)
        return result

    def _extract_from_text(self, text: str) -> dict:
        """本地规则提取结构化参数"""
        result = {}
        # 城市提取：优先匹配已知城市，否则尝试提取「XX市/XX城/去XX」等模式
        data = load_attractions()
        for city in data["cities"]:
            if city in text:
                result["destination"] = city
                break
        if "destination" not in result:
            # 匹配 "信阳站/去信阳/信阳市/信阳城/在信阳" 等模式
            m = re.search(r"(?:去|在|到|想去|玩转?)\s*([一-鿿]{2,4}(?:市|城|县|镇|区)?)", text)
            if not m:
                m = re.search(r"([一-鿿]{2,4})(?:玩|旅游|旅行|游|待|住|出发)", text)
            if m:
                city = m.group(1)
                if not city.endswith(("市", "城", "县", "镇", "区")):
                    # 去除常见非地名词
                    skip = {"想去", "喜欢", "预算", "准备", "计划", "安排", "希望"}
                    if city not in skip and len(city) >= 2:
                        result["destination"] = city
        # 天数提取
        day_patterns = [r"(\d+)天", r"(\d+)日游", r"玩(\d+)天"]
        for pat in day_patterns:
            m = re.search(pat, text)
            if m:
                result["days"] = int(m.group(1))
                break
        # 预算提取
        if any(w in text for w in ["穷游", "省钱", "经济", "便宜", "预算少"]):
            result["budget"] = "budget"
        elif any(w in text for w in ["豪华", "奢侈", "享受", "高端", "五星"]):
            result["budget"] = "luxury"
        # 偏好提取
        prefs = []
        pref_keywords = {
            "美食": ["美食", "好吃", "小吃", "火锅", "餐厅"],
            "历史文化": ["历史", "文化", "古迹", "博物馆", "传统"],
            "自然风光": ["自然", "风景", "山水", "海", "湖", "山", "公园"],
            "拍照打卡": ["拍照", "打卡", "网红", "摄影", "出片"],
            "购物": ["购物", "逛街", "买", "商场", "商业街"],
            "亲子": ["亲子", "孩子", "带娃", "乐园", "动物"],
        }
        for pref, keywords in pref_keywords.items():
            if any(kw in text for kw in keywords):
                prefs.append(pref)
        if prefs:
            result["preferences"] = "、".join(prefs)
        # 同伴提取
        for kw, val in [("情侣", "情侣"), ("闺蜜", "朋友"), ("带娃", "亲子"), ("家人", "家庭"), ("一个人", "独自")]:
            if kw in text:
                result["companions"] = val
                break
        return result

    def _llm_extract(self, text: str) -> dict:
        """使用 LLM 提取旅行参数"""
        prompt = f"""你是一个旅行规划助手。从用户的自然语言描述中提取旅行参数。

预算等级：budget(经济)/standard(舒适)/luxury(豪华)

用户输入："{text}"

请只返回一个 JSON 对象（不要其他文字）：
{{"destination": "用户想去的城市名(直接提取城市名,如信阳/郑州/武汉等)", "days": 天数(数字), "budget": "预算等级", "preferences": "偏好关键词用中文逗号分隔", "companions": "同行人类型"}}"""

        messages = [
            {"role": "system", "content": "你是一个精准的参数提取工具，只输出JSON，不输出其他内容。"},
            {"role": "user", "content": prompt},
        ]
        content = call_llm(messages, temperature=0.1, max_tokens=300)
        if content:
            # 清理可能的 markdown 包裹
            content = re.sub(r"^```(?:json)?\s*", "", content.strip())
            content = re.sub(r"\s*```$", "", content.strip())
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"[IntentAgent] LLM返回解析失败: {content[:200]}")
        return {}


class ExploreAgent(TravelAgent):
    """景点探索Agent：搜索目的地景点并排序"""

    def __init__(self):
        super().__init__("ExploreAgent", "🔍", "搜索景点并智能排序推荐")

    def run(self, context: dict, callback: Callable = None) -> dict:
        super().run(context, callback)
        destination = context.get("destination", "")
        days = context.get("days", 3)
        preferences = context.get("preferences", "")

        self.emit("running", f"🔍 正在搜索「{destination}」热门景点...")

        # 获取本地数据
        attractions = get_attractions(destination, preferences)

        # 如果有 LLM，增强景点推荐
        if has_llm() and attractions:
            llm_enhanced = self._llm_enhance(attractions, destination, preferences, days)
            if llm_enhanced:
                attractions = llm_enhanced

        # 如果没有找到该城市
        if not attractions:
            self.emit("error", f"⚠️ 未找到「{destination}」的景点信息，请尝试换个目的地或检查高德Key是否配置")
            return {"attractions": [], "city_found": False}

        # 根据天数筛选合适数量的景点（每天3-4个）
        max_attractions = days * 4
        selected = attractions[:max_attractions]

        self.emit("done",
            f"✅ 在「{destination}」发现 {len(attractions)} 个景点，已优选 {len(selected)} 个",
            {"attractions": selected, "total_found": len(attractions)})
        return {"attractions": selected, "city_found": True, "destination": destination}

    def _llm_enhance(self, attractions: list, city: str, preferences: str, days: int) -> list:
        """使用 LLM 为景点生成增强描述"""
        try:
            # 只增强前6个景点，减少token消耗
            subset = attractions[:6]
            names = "、".join([a["name"] for a in subset])
            prompt = f"""你在为旅行规划「{city}」{days}日游推荐景点。
偏好：{preferences or '综合体验'}
现有景点：{names}

请对每个景点用一句话推荐（融入偏好），返回JSON数组：
[{{"name": "景点名", "enhanced_tip": "一句话推荐(含实用小贴士)"}}]"""

            messages = [
                {"role": "system", "content": "你是资深旅行达人，输出简洁实用的推荐。只输出JSON数组。"},
                {"role": "user", "content": prompt},
            ]
            content = call_llm(messages, temperature=0.7, max_tokens=600)
            if content:
                content = re.sub(r"^```(?:json)?\s*", "", content.strip())
                content = re.sub(r"\s*```$", "", content.strip())
                enhanced_map = {}
                for item in json.loads(content):
                    enhanced_map[item["name"]] = item.get("enhanced_tip", "")
                # 合并回原始数据
                for a in attractions:
                    if a["name"] in enhanced_map and enhanced_map[a["name"]]:
                        a["enhanced_tip"] = enhanced_map[a["name"]]
        except Exception as e:
            print(f"[ExploreAgent] LLM增强失败: {e}")
        return attractions


class HotelAgent(TravelAgent):
    """酒店推荐Agent：根据预算推荐住宿"""

    def __init__(self):
        super().__init__("HotelAgent", "🏨", "根据预算智能匹配酒店")

    def run(self, context: dict, callback: Callable = None) -> dict:
        super().run(context, callback)
        budget = context.get("budget", "standard")
        days = context.get("days", 3)
        destination = context.get("destination", "")

        self.emit("running", f"🏨 正在为您匹配{BUDGET_LEVELS.get(budget, '')}酒店...")

        hotels = get_hotels(budget, city=destination)
        # 推荐2-3家酒店
        selected = random.sample(hotels, min(3, len(hotels)))

        self.emit("done",
            f"✅ 已为您匹配 {len(selected)} 家{BUDGET_LEVELS.get(budget, '')}酒店",
            {"hotels": selected})
        return {"hotels": selected}


class WeatherAgent(TravelAgent):
    """天气查询Agent：获取目的地天气预报"""

    def __init__(self):
        super().__init__("WeatherAgent", "🌤️", "查询目的地未来天气")

    def run(self, context: dict, callback: Callable = None) -> dict:
        super().run(context, callback)
        destination = context.get("destination", "")
        days = context.get("days", 3)

        self.emit("running", f"🌤️ 正在查询「{destination}」未来{days}天天气...")

        weather_data = self._get_weather(destination, days)

        self.emit("done",
            f"✅ 天气查询完成：{weather_data[0]['weather']}，{weather_data[0]['temp']}",
            {"weather": weather_data})
        return {"weather": weather_data}

    def _get_weather(self, city: str, days: int) -> list:
        """获取天气（优先高德API，否则模拟）"""
        # 尝试高德天气 API
        if AMAP_API_KEY and HAS_REQUESTS:
            try:
                # 先获取城市编码
                geo_resp = http_requests.get(
                    "https://restapi.amap.com/v3/geocode/geo",
                    params={"key": AMAP_API_KEY, "address": city},
                    timeout=10,
                )
                if geo_resp.status_code == 200:
                    geo_data = geo_resp.json()
                    if geo_data.get("geocodes"):
                        adcode = geo_data["geocodes"][0]["adcode"]
                        weather_resp = http_requests.get(
                            "https://restapi.amap.com/v3/weather/weatherInfo",
                            params={"key": AMAP_API_KEY, "city": adcode, "extensions": "all"},
                            timeout=10,
                        )
                        if weather_resp.status_code == 200:
                            wdata = weather_resp.json()
                            if wdata.get("forecasts"):
                                casts = wdata["forecasts"][0]["casts"][:days]
                                return [
                                    {
                                        "date": c["date"],
                                        "week": c.get("week", ""),
                                        "weather": c.get("dayweather", "晴"),
                                        "temp": f"{c.get('nighttemp', '15')}°C ~ {c.get('daytemp', '25')}°C",
                                        "tip": self._weather_tip(c.get("dayweather", "")),
                                    }
                                    for c in casts
                                ]
            except Exception as e:
                print(f"[Weather] 高德API失败: {e}")

        # 模拟天气数据
        return self._simulate_weather(days)

    def _simulate_weather(self, days: int) -> list:
        """模拟天气数据"""
        weathers = ["晴 ☀️", "多云 ⛅", "阴天 ☁️", "小雨 🌧️", "晴转多云 🌤️"]
        temps = ["18°C ~ 28°C", "20°C ~ 30°C", "16°C ~ 26°C", "15°C ~ 24°C", "22°C ~ 32°C"]
        tips_map = {
            "晴 ☀️": "阳光充足，适合户外活动，记得涂防晒霜",
            "多云 ⛅": "天气舒适，非常适合出行游览",
            "阴天 ☁️": "阴天光线柔和，适合拍照，带把伞以防万一",
            "小雨 🌧️": "小雨不影响出行，带雨伞即可，别有一番意境",
            "晴转多云 🌤️": "上午天气好，建议户外景点安排在上午",
        }

        today = datetime.now()
        result = []
        for i in range(days):
            date = today + timedelta(days=i)
            w = random.choice(weathers)
            t = random.choice(temps)
            result.append({
                "date": date.strftime("%Y-%m-%d"),
                "week": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][date.weekday()],
                "weather": w,
                "temp": t,
                "tip": tips_map.get(w, "适合出行"),
            })
        return result

    def _weather_tip(self, weather: str) -> str:
        tips = {
            "晴": "阳光充足，适合户外活动，记得防晒",
            "多云": "天气舒适，非常适合出行",
            "阴": "光线柔和适合拍照，带伞以防万一",
            "雨": "带好雨具，室内景点也是好选择",
            "雪": "注意保暖防滑，雪景别有一番风味",
        }
        for k, v in tips.items():
            if k in weather:
                return v
        return "适合出行，祝旅途愉快"


class PlannerAgent(TravelAgent):
    """行程编排Agent：将所有信息整合为完整行程方案"""

    def __init__(self):
        super().__init__("PlannerAgent", "📋", "编排完整行程与预算")

    def run(self, context: dict, callback: Callable = None) -> dict:
        super().run(context, callback)
        self.emit("running", "📋 正在编排每日行程...")

        destination = context.get("destination", "")
        days = context.get("days", 3)
        budget_level = context.get("budget", "standard")
        preferences = context.get("preferences", "")
        attractions = context.get("attractions", [])
        hotels = context.get("hotels", [])
        weather = context.get("weather", [])

        # 编排每日行程
        daily_schedule = self._build_daily_schedule(
            attractions, days, destination, preferences, weather
        )

        # 计算预算
        budget_breakdown = self._calculate_budget(
            daily_schedule, hotels, budget_level, days
        )

        # 如果有 LLM，优化行程
        if has_llm():
            optimized = self._llm_optimize(
                daily_schedule, budget_breakdown, destination, days, preferences
            )
            if optimized:
                daily_schedule = optimized.get("schedule", daily_schedule)
                budget_breakdown = optimized.get("budget", budget_breakdown)

        self.emit("done",
            f"✅ 行程编排完成！{days}天行程 + 预算明细已生成",
            {"days": days})

        return {
            "daily_schedule": daily_schedule,
            "budget_breakdown": budget_breakdown,
            "hotel_recommendation": hotels[0] if hotels else None,
        }

    def _build_daily_schedule(self, attractions: list, days: int,
                               city: str, preferences: str,
                               weather: list) -> list:
        """构建每日行程"""
        schedule = []
        # 均匀分配景点到每一天
        per_day = max(2, len(attractions) // days) if attractions else 2
        # 确保每天至少有2个景点
        per_day = max(2, min(per_day, 4))

        time_slots = ["09:00-11:30", "13:00-15:30", "16:00-18:00", "19:00-20:30"]
        # 根据偏好调整上午/下午的景点类型
        if "美食" in preferences:
            time_slots = ["09:30-11:30", "12:00-14:00(午餐)", "15:00-17:00", "18:00-20:00(晚餐)"]

        for day_idx in range(days):
            start = day_idx * per_day
            end = start + per_day
            day_attractions = attractions[start:end] if start < len(attractions) else []

            # 如果景点不够，重复最后一个
            while len(day_attractions) < per_day and len(day_attractions) < 3:
                if day_idx > 0 and schedule:
                    # 从其他天借一个
                    pass
                break

            day_items = []
            for i, attr in enumerate(day_attractions):
                slot = time_slots[min(i, len(time_slots) - 1)]
                day_items.append({
                    "time": slot,
                    "name": attr["name"],
                    "category": attr.get("category", "景点"),
                    "address": attr.get("address", ""),
                    "duration": attr.get("duration", "1-2小时"),
                    "ticket": attr.get("ticket", 0),
                    "tips": attr.get("enhanced_tip") or attr.get("tips", ""),
                    "rating": attr.get("rating", 4.0),
                })

            # 添加用餐建议
            meal_tips = self._meal_suggestions(city, day_idx)

            # 天气信息
            day_weather = weather[day_idx] if day_idx < len(weather) else None

            schedule.append({
                "day": day_idx + 1,
                "title": f"第{day_idx + 1}天",
                "date": (datetime.now() + timedelta(days=day_idx)).strftime("%m月%d日"),
                "weather": day_weather,
                "items": day_items,
                "meal_tips": meal_tips,
                "summary": f"共{len(day_items)}个景点，节奏{'轻松' if len(day_items) <= 2 else '适中' if len(day_items) <= 3 else '充实'}",
            })

        return schedule

    def _meal_suggestions(self, city: str, day_idx: int) -> dict:
        """根据城市生成用餐建议"""
        food_map = {
            "北京": {"早餐": "豆汁焦圈/炒肝包子", "午餐": "炸酱面/卤煮", "晚餐": "北京烤鸭/涮羊肉"},
            "上海": {"早餐": "生煎包/小馄饨", "午餐": "小笼包/排骨年糕", "晚餐": "本帮菜/蟹粉豆腐"},
            "西安": {"早餐": "肉丸胡辣汤/甑糕", "午餐": "biangbiang面/凉皮", "晚餐": "羊肉泡馍/葫芦鸡"},
            "成都": {"早餐": "担担面/抄手", "午餐": "冒菜/钟水饺", "晚餐": "火锅/串串香"},
            "杭州": {"早餐": "葱包烩/定胜糕", "午餐": "西湖醋鱼/龙井虾仁", "晚餐": "东坡肉/叫花鸡"},
            "广州": {"早餐": "肠粉/虾饺", "午餐": "煲仔饭/云吞面", "晚餐": "白切鸡/烧鹅"},
            "重庆": {"早餐": "重庆小面/油茶", "午餐": "酸辣粉/豆花饭", "晚餐": "重庆火锅/毛血旺"},
            "南京": {"早餐": "鸭血粉丝汤/小笼包", "午餐": "盐水鸭/皮肚面", "晚餐": "金陵菜/狮子头"},
            "厦门": {"早餐": "沙茶面/面线糊", "午餐": "海蛎煎/土笋冻", "晚餐": "姜母鸭/海鲜大排档"},
            "昆明": {"早餐": "过桥米线/烧饵块", "午餐": "汽锅鸡/宜良烤鸭", "晚餐": "野生菌火锅/傣味菜"},
            "三亚": {"早餐": "海南粉/椰丝包", "午餐": "文昌鸡/加积鸭", "晚餐": "海鲜大餐/东山羊"},
            "长沙": {"早餐": "长沙米粉/糖油粑粑", "午餐": "臭豆腐/剁椒鱼头", "晚餐": "口味虾/辣椒炒肉"},
        }
        return food_map.get(city, {"早餐": "当地特色早餐", "午餐": "当地特色午餐", "晚餐": "当地特色晚餐"})

    def _calculate_budget(self, daily_schedule: list, hotels: list,
                           budget_level: str, days: int) -> dict:
        """计算详细预算"""
        daily_base = BUDGET_DAILY.get(budget_level, BUDGET_DAILY["standard"])

        # 门票总计
        total_tickets = 0
        for day in daily_schedule:
            for item in day["items"]:
                total_tickets += item.get("ticket", 0)

        # 酒店费用
        hotel_price = hotels[0]["price_per_night"] if hotels else daily_base["hotel"]
        total_hotel = hotel_price * max(1, days - 1)  # N天行程住N-1晚

        # 交通（往返大交通 + 市内）
        total_transport = daily_base["transport"] * days + random.randint(300, 1500)

        # 餐饮
        total_food = daily_base["food"] * days

        # 其他
        total_other = daily_base["other"] * days

        hotel_price_per = hotel_price

        return {
            "transport": {"name": "交通", "amount": total_transport, "icon": "🚄",
                         "detail": f"含往返大交通({random.choice(['高铁','飞机','自驾'])}估算) + {days}天市内交通"},
            "hotel": {"name": "住宿", "amount": total_hotel, "icon": "🏨",
                     "detail": f"{hotel_price_per}元/晚 × {max(1, days-1)}晚"},
            "food": {"name": "餐饮", "amount": total_food, "icon": "🍜",
                    "detail": f"{daily_base['food']}元/天 × {days}天"},
            "tickets": {"name": "门票", "amount": total_tickets, "icon": "🎫",
                       "detail": "各景点门票合计"},
            "other": {"name": "其他", "amount": total_other, "icon": "💡",
                     "detail": "购物/纪念品/意外支出预留"},
            "total": 0,
            "per_person": 0,
            "budget_level": budget_level,
            "budget_label": BUDGET_LEVELS.get(budget_level, ""),
        }

    def _llm_optimize(self, schedule: list, budget: dict, city: str, days: int, preferences: str) -> dict:
        """使用 LLM 优化行程"""
        # 构建简化的行程描述
        schedule_text = "\n".join([
            f"第{d['day']}天: " + " → ".join([i['name'] for i in d['items']])
            for d in schedule
        ])

        prompt = f"""优化以下{city}{days}日游行程：
偏好：{preferences or '综合体验'}

当前行程：
{schedule_text}

请优化行程并返回JSON（保持相同结构）：
{{
  "schedule": [
    {{"day": 1, "optimized_order": ["景点1", "景点2"], "pace_tip": "节奏建议一句话"}}
  ],
  "budget_tip": "预算优化建议一句话"
}}"""

        messages = [
            {"role": "system", "content": "你是资深旅行规划师，输出简洁JSON。"},
            {"role": "user", "content": prompt},
        ]
        content = call_llm(messages, temperature=0.5, max_tokens=800)
        if content:
            content = re.sub(r"^```(?:json)?\s*", "", content.strip())
            content = re.sub(r"\s*```$", "", content.strip())
            try:
                optimized = json.loads(content)
                # 应用优化建议到行程
                for s in schedule:
                    for opt in optimized.get("schedule", []):
                        if opt.get("day") == s["day"]:
                            if opt.get("pace_tip"):
                                s["summary"] = opt["pace_tip"]
                if optimized.get("budget_tip"):
                    budget["optimization_tip"] = optimized["budget_tip"]
                return {"schedule": schedule, "budget": budget}
            except json.JSONDecodeError:
                pass
        return {"schedule": schedule, "budget": budget}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                  🔄  Agent  编 排 器 (Orchestrator)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TravelOrchestrator:
    """
    旅行规划编排器：按顺序调度各Agent，通过回调函数发送SSE进度事件。
    工作流: IntentAgent → ExploreAgent → HotelAgent → WeatherAgent → PlannerAgent
    """

    def __init__(self):
        self.agents = {
            "intent": IntentAgent(),
            "explore": ExploreAgent(),
            "hotel": HotelAgent(),
            "weather": WeatherAgent(),
            "planner": PlannerAgent(),
        }

    def run_workflow(self, form_data: dict, callback: Callable = None) -> dict:
        """
        执行完整的旅行规划工作流
        返回: {context, itinerary, budget, errors}
        """
        context = {**form_data}
        errors = []

        # ── Step 1: 意图分析 ──
        intent_result = self.agents["intent"].run(context, callback)
        context.update(intent_result)
        if not context.get("destination"):
            errors.append("未能识别目的地，请明确告知想去哪个城市")
            return {"context": context, "itinerary": None, "error": "; ".join(errors)}

        # ── Step 2: 景点探索 ──
        explore_result = self.agents["explore"].run(context, callback)
        context.update(explore_result)
        if not context.get("attractions"):
            errors.append(f"未找到「{context['destination']}」的景点信息")
            return {"context": context, "itinerary": None, "error": "; ".join(errors)}

        # ── Step 3: 酒店推荐 ──
        hotel_result = self.agents["hotel"].run(context, callback)
        context.update(hotel_result)

        # ── Step 4: 天气查询 ──
        weather_result = self.agents["weather"].run(context, callback)
        context.update(weather_result)

        # ── Step 5: 行程编排 ──
        planner_result = self.agents["planner"].run(context, callback)
        context.update(planner_result)

        # ── 汇总结果 ──
        itinerary = {
            "destination": context["destination"],
            "days": context["days"],
            "budget_level": context["budget"],
            "budget_label": BUDGET_LEVELS.get(context["budget"], ""),
            "preferences": context.get("preferences", ""),
            "companions": context.get("companions", "独自"),
            "daily_schedule": planner_result.get("daily_schedule", []),
            "hotel": planner_result.get("hotel_recommendation"),
            "weather": context.get("weather", []),
        }

        budget = planner_result.get("budget_breakdown", {})
        # 计算总计
        total = sum(v["amount"] if isinstance(v, dict) else 0
                    for k, v in budget.items() if k not in ("total", "per_person", "budget_level", "budget_label", "optimization_tip"))
        budget["total"] = total
        budget["per_person"] = total  # 简化：单人预算

        return {
            "context": context,
            "itinerary": itinerary,
            "budget": budget,
            "error": None,
        }


    def modify_itinerary(self, trip_data: dict, user_message: str,
                          callback: Callable = None) -> dict:
        """
        多轮对话：根据用户反馈修改行程
        """
        if callback:
            callback({
                "agent": "PlannerAgent", "icon": "📋",
                "status": "running", "message": "🔄 正在根据您的反馈调整行程...", "data": {}
            })

        itinerary = trip_data.get("itinerary_json", {})
        if isinstance(itinerary, str):
            itinerary = json.loads(itinerary)

        budget_json = trip_data.get("budget_json", {})
        if isinstance(budget_json, str):
            budget_json = json.loads(budget_json)

        # 构建当前行程摘要
        schedule_summary = ""
        for day in itinerary.get("daily_schedule", []):
            items_str = " → ".join([i["name"] for i in day.get("items", [])])
            schedule_summary += f"第{day['day']}天: {items_str}\n"

        if has_llm():
            modified = self._llm_modify(
                itinerary["destination"], itinerary["days"],
                schedule_summary, user_message, itinerary.get("preferences", "")
            )
            if modified:
                # 合并修改到原行程
                itinerary, budget_json = self._apply_modifications(
                    itinerary, budget_json, modified, callback
                )

        if callback:
            callback({
                "agent": "PlannerAgent", "icon": "📋",
                "status": "done", "message": "✅ 行程已根据您的反馈调整完毕！", "data": {}
            })

        return {"itinerary": itinerary, "budget": budget_json}

    def _llm_modify(self, city: str, days: int, schedule: str,
                     feedback: str, preferences: str) -> dict:
        """使用 LLM 解析修改请求"""
        prompt = f"""当前是{city}{days}日游行程：
{schedule}

用户反馈："{feedback}"
偏好：{preferences}

请返回JSON修改方案：
{{
  "adjustments": [
    {{"day": 1, "action": "add/remove/reorder/replace", "target": "景点名", "detail": "修改说明"}}
  ],
  "summary": "一句话总结修改内容"
}}"""

        messages = [
            {"role": "system", "content": "你是旅行行程调整助手，只输出JSON。"},
            {"role": "user", "content": prompt},
        ]
        content = call_llm(messages, temperature=0.5, max_tokens=500)
        if content:
            content = re.sub(r"^```(?:json)?\s*", "", content.strip())
            content = re.sub(r"\s*```$", "", content.strip())
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        return {}

    def _apply_modifications(self, itinerary: dict, budget: dict,
                              modified: dict, callback) -> tuple:
        """应用LLM的修改方案到行程"""
        adjustments = modified.get("adjustments", [])
        if not adjustments:
            return itinerary, budget

        daily = itinerary.get("daily_schedule", [])
        for adj in adjustments:
            day_num = adj.get("day", 1)
            action = adj.get("action", "")
            target = adj.get("target", "")

            for day in daily:
                if day["day"] == day_num:
                    if action == "remove":
                        day["items"] = [i for i in day["items"] if i["name"] != target]
                        day["summary"] = f"已移除「{target}」，行程更轻松"
                    elif action == "add":
                        day["items"].append({
                            "time": "灵活安排",
                            "name": target,
                            "category": "推荐",
                            "address": "请参考地图",
                            "duration": "1-2小时",
                            "ticket": 0,
                            "tips": adj.get("detail", "根据您的需求新增"),
                        })
                        day["summary"] = f"已新增「{target}」"
                    elif action == "replace":
                        for item in day["items"]:
                            if item["name"] == target:
                                item["name"] = adj.get("detail", item["name"])
                                item["tips"] = "根据您的要求替换"
                    break

        itinerary["daily_schedule"] = daily
        if modified.get("summary"):
            itinerary["modification_note"] = modified["summary"]

        return itinerary, budget

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     🌐  Flask  路 由 层                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

app = Flask(__name__, static_folder=None, template_folder=str(BASE_DIR / "templates"))
CORS(app)
app.teardown_appcontext(close_db)

orchestrator = TravelOrchestrator()


@app.route("/")
def index():
    """首页：返回 Vue3 SPA"""
    from flask import send_from_directory
    return send_from_directory(str(BASE_DIR / "templates"), "index.html")


# ── 用户注册 / 登录 / 查询 ──────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def register():
    """注册新用户"""
    data = request.get_json(force=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or len(username) < 1:
        return jsonify({"ok": False, "error": "请输入用户名"}), 400
    if len(username) > 15:
        return jsonify({"ok": False, "error": "用户名最多15个字符"}), 400
    if not password or len(password) < 4:
        return jsonify({"ok": False, "error": "密码至少4位"}), 400

    conn = get_db()
    exists = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if exists:
        return jsonify({"ok": False, "error": "用户名已被占用"}), 409

    pw_hash = generate_password_hash(password)
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, avatar) VALUES (?, ?, ?)",
        (username, pw_hash, "🧳"),
    )
    conn.commit()
    return jsonify({"ok": True, "user": {"id": cur.lastrowid, "username": username, "avatar": "🧳"}})


@app.route("/api/login", methods=["POST"])
def login():
    """用户登录：验证用户名和密码"""
    data = request.get_json(force=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"ok": False, "error": "请输入用户名和密码"}), 400

    conn = get_db()
    row = conn.execute(
        "SELECT id, username, avatar, password_hash FROM users WHERE username=?",
        (username,),
    ).fetchone()

    if not row:
        return jsonify({"ok": False, "error": "用户不存在，请先注册"}), 404

    if not check_password_hash(row[3], password):
        return jsonify({"ok": False, "error": "密码错误"}), 401

    return jsonify({"ok": True, "user": {"id": row[0], "username": row[1], "avatar": row[2]}})


@app.route("/api/user/<int:user_id>", methods=["GET"])
def get_user(user_id):
    """根据ID查询用户信息（用于恢复登录态）"""
    conn = get_db()
    row = conn.execute("SELECT id, username, avatar FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "用户不存在"}), 404
    return jsonify({"ok": True, "user": {"id": row[0], "username": row[1], "avatar": row[2]}})


# ── SSE 流式规划 ──────────────────────────────────────────────────────────

@app.route("/api/plan/stream", methods=["POST"])
def plan_stream():
    """
    SSE 流式旅行规划接口
    接收表单数据 → 实时推送每个Agent的进度 → 最终返回完整行程
    """
    data = request.get_json(force=True) or {}

    def generate():
        events = []
        def collect_event(event: dict):
            events.append(event)

        try:
            # 发送初始事件
            init_event = json.dumps({
                "type": "start",
                "message": "🚀 旅行规划引擎启动中...",
                "agents": ["IntentAgent", "ExploreAgent", "HotelAgent", "WeatherAgent", "PlannerAgent"],
            }, ensure_ascii=False)
            yield f"data: {init_event}\n\n"

            # 运行工作流
            result = orchestrator.run_workflow(data, collect_event)

            # 发送所有Agent的进度事件
            for evt in events:
                payload = json.dumps({
                    "type": "progress",
                    **evt,
                }, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                time.sleep(0.3)  # 视觉进度间隔

            # 保存到数据库
            if result["itinerary"]:
                trip_id = _save_trip(data.get("user_id", 1), result)
                result["trip_id"] = trip_id

            # 发送最终结果
            final = json.dumps({
                "type": "result",
                "ok": result["error"] is None,
                "trip_id": result.get("trip_id"),
                "itinerary": result["itinerary"],
                "budget": result["budget"],
                "error": result.get("error"),
            }, ensure_ascii=False)
            yield f"data: {final}\n\n"

            # 完成事件
            yield f"data: {json.dumps({'type': 'done', 'message': '规划完成！'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_event = json.dumps({
                "type": "error",
                "message": f"规划出错: {str(e)}",
            }, ensure_ascii=False)
            yield f"data: {error_event}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── 对话模式 ──────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    对话式旅行规划接口
    支持自然语言输入，系统理解后自动提取参数或执行修改
    """
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", 1)
    trip_id = data.get("trip_id")

    if not user_message:
        return jsonify({"ok": False, "error": "请输入消息"}), 400

    db = get_db()

    # 保存用户消息
    db.execute(
        "INSERT INTO chat_history (user_id, role, content, trip_id) VALUES (?, 'user', ?, ?)",
        (user_id, user_message, trip_id),
    )
    db.commit()

    # 判断消息意图
    is_modification = False
    if trip_id:
        # 已存在行程 → 可能是修改请求
        modify_keywords = ["改", "换", "调整", "去掉", "增加", "加", "少", "太多", "不够",
                          "太满", "太累", "轻松", "不要", "想换成", "修改", "变了"]
        if any(kw in user_message for kw in modify_keywords):
            is_modification = True

    if is_modification and trip_id:
        # ── 修改行程 ──
        trip = db.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
        if not trip:
            return jsonify({"ok": False, "error": "行程不存在"}), 404

        trip_data = dict(trip)
        events = []
        result = orchestrator.modify_itinerary(trip_data, user_message, lambda e: events.append(e))

        # 更新数据库
        db.execute(
            "UPDATE trips SET itinerary_json=?, budget_json=? WHERE id=?",
            (json.dumps(result["itinerary"], ensure_ascii=False),
             json.dumps(result["budget"], ensure_ascii=False),
             trip_id),
        )
        db.commit()

        ai_reply = f"已根据「{user_message}」调整行程。{result['itinerary'].get('modification_note', '请查看更新后的行程～')}"

        db.execute(
            "INSERT INTO chat_history (user_id, role, content, trip_id) VALUES (?, 'assistant', ?, ?)",
            (user_id, ai_reply, trip_id),
        )
        db.commit()

        return jsonify({
            "ok": True,
            "type": "modification",
            "reply": ai_reply,
            "itinerary": result["itinerary"],
            "budget": result["budget"],
            "trip_id": trip_id,
        })

    else:
        # ── 新建行程：从对话中提取参数 ──
        context = {"raw_input": user_message}
        events = []
        intent_result = orchestrator.agents["intent"].run(context, lambda e: events.append(e))

        destination = intent_result.get("destination", "")
        days = intent_result.get("days", 3)

        if not destination:
            # 无法提取目的地，引导用户
            cities = ", ".join([c["name"] for c in get_cities()])
            ai_reply = f"请问您想去哪个城市呢？目前支持：{cities}等热门目的地。您也可以直接告诉我如「想去西安玩3天，预算中等，喜欢美食和历史」～"

            db.execute(
                "INSERT INTO chat_history (user_id, role, content, trip_id) VALUES (?, 'assistant', ?, ?)",
                (user_id, ai_reply, trip_id),
            )
            db.commit()

            return jsonify({
                "ok": True,
                "type": "clarification",
                "reply": ai_reply,
                "extracted": intent_result,
                "available_cities": get_cities(),
            })

        # 参数完整，提示用户确认
        ai_reply = f"我理解您想去「{destination}」玩{days}天，预算{BUDGET_LEVELS.get(intent_result.get('budget', 'standard'), '舒适型')}。偏好：{intent_result.get('preferences', '综合体验')}。正在为您生成行程..."

        db.execute(
            "INSERT INTO chat_history (user_id, role, content, trip_id) VALUES (?, 'assistant', ?, ?)",
            (user_id, ai_reply, trip_id),
        )
        db.commit()

        return jsonify({
            "ok": True,
            "type": "ready_to_plan",
            "reply": ai_reply,
            "extracted": intent_result,
            "can_proceed": True,
        })


# ── 对话后执行规划（供前端在确认后调用） ────────────────────────────────

@app.route("/api/chat/plan", methods=["POST"])
def chat_plan():
    """对话确认后，执行完整规划（SSE流式）"""
    data = request.get_json(force=True)
    # 提取的参数来自对话
    extracted = data.get("extracted", {})
    form_data = {
        "destination": extracted.get("destination", ""),
        "days": extracted.get("days", 3),
        "budget": extracted.get("budget", "standard"),
        "preferences": extracted.get("preferences", ""),
        "companions": extracted.get("companions", "独自"),
        "user_id": data.get("user_id", 1),
    }

    def generate():
        events = []
        def cb(evt):
            events.append(evt)

        try:
            yield f"data: {json.dumps({'type': 'start', 'message': '开始生成行程...'}, ensure_ascii=False)}\n\n"

            result = orchestrator.run_workflow(form_data, cb)

            for evt in events:
                payload = json.dumps({"type": "progress", **evt}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                time.sleep(0.3)

            if result["itinerary"]:
                trip_id = _save_trip(data.get("user_id", 1), result)
                result["trip_id"] = trip_id

            final = json.dumps({
                "type": "result",
                "ok": result["error"] is None,
                "trip_id": result.get("trip_id"),
                "itinerary": result["itinerary"],
                "budget": result["budget"],
                "error": result.get("error"),
            }, ensure_ascii=False)
            yield f"data: {final}\n\n"

            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── 行程管理 ────────────────────────────────────────────────────────────────

@app.route("/api/trips", methods=["GET"])
def list_trips():
    """获取所有旅行计划"""
    user_id = request.args.get("user_id", 1)
    db = get_db()
    rows = db.execute(
        "SELECT * FROM trips WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
        (user_id,),
    ).fetchall()
    trips = []
    for row in rows:
        trip = dict(row)
        # 解析JSON字段
        for field in ("itinerary_json", "budget_json", "weather_json"):
            if isinstance(trip.get(field), str):
                try:
                    trip[field] = json.loads(trip[field])
                except json.JSONDecodeError:
                    trip[field] = {}
        trips.append(trip)
    return jsonify({"ok": True, "trips": trips})


@app.route("/api/trips/<int:trip_id>", methods=["GET"])
def get_trip(trip_id):
    """获取单个旅行计划详情"""
    db = get_db()
    row = db.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "行程不存在"}), 404
    trip = dict(row)
    for field in ("itinerary_json", "budget_json", "weather_json"):
        if isinstance(trip.get(field), str):
            try:
                trip[field] = json.loads(trip[field])
            except json.JSONDecodeError:
                trip[field] = {}
    return jsonify({"ok": True, "trip": trip})


@app.route("/api/trips/<int:trip_id>", methods=["DELETE"])
def delete_trip(trip_id):
    """删除旅行计划"""
    db = get_db()
    db.execute("DELETE FROM trips WHERE id=?", (trip_id,))
    db.commit()
    return jsonify({"ok": True, "message": "已删除"})


# ── 城市列表 ────────────────────────────────────────────────────────────────

@app.route("/api/cities", methods=["GET"])
def list_cities():
    """获取支持的城市列表"""
    return jsonify({"ok": True, "cities": get_cities()})


# ── 对话历史 ────────────────────────────────────────────────────────────────

@app.route("/api/chat/history", methods=["GET"])
def chat_history():
    """获取对话历史"""
    trip_id = request.args.get("trip_id")
    user_id = request.args.get("user_id", 1)
    db = get_db()
    if trip_id:
        rows = db.execute(
            "SELECT * FROM chat_history WHERE user_id=? AND trip_id=? ORDER BY created_at ASC",
            (user_id, trip_id),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM chat_history WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
            (user_id,),
        ).fetchall()
    return jsonify({
        "ok": True,
        "messages": [dict(r) for r in rows],
    })


# ── 系统状态 ────────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def system_status():
    """系统状态检查"""
    return jsonify({
        "ok": True,
        "llm_available": has_llm(),
        "llm_provider": LLM_PROVIDER if has_llm() else "local",
        "llm_model": LLM_MODEL if has_llm() else "N/A",
        "amap_available": bool(AMAP_API_KEY),
        "cities_count": len(get_cities()),
    })


# ── 辅助函数 ────────────────────────────────────────────────────────────────

def _save_trip(user_id: int, result: dict) -> int:
    """保存旅行计划到数据库"""
    db = get_db()
    itinerary = result["itinerary"]
    budget = result["budget"]
    cur = db.execute(
        """INSERT INTO trips (user_id, destination, days, budget, preferences, companions,
           itinerary_json, budget_json, weather_json, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed')""",
        (
            user_id,
            itinerary["destination"],
            itinerary["days"],
            itinerary["budget_level"],
            itinerary.get("preferences", ""),
            itinerary.get("companions", "独自"),
            json.dumps(itinerary, ensure_ascii=False),
            json.dumps(budget, ensure_ascii=False),
            json.dumps(itinerary.get("weather", []), ensure_ascii=False),
        ),
    )
    db.commit()
    return cur.lastrowid


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       🚀  启 动 入 口                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # 修复 Windows 终端编码
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  [Travel] 智能旅行规划系统")
    print("  Multi-Agent Travel Planning System")
    print("=" * 60)
    print()

    # 初始化数据库
    init_db()

    # 加载景点数据
    data = load_attractions()
    cities_count = len(data["cities"])
    print(f"[Data] 已加载 {cities_count} 个城市的景点数据")

    # LLM 状态
    if has_llm():
        print(f"[LLM] {LLM_PROVIDER} / {LLM_MODEL} 已就绪")
    else:
        print("[LLM] 未配置API Key，使用本地规则引擎 + 模拟数据")
        print("      设置 LLM_API_KEY 环境变量以启用AI增强")

    # 地图状态
    if AMAP_API_KEY:
        print("[Map] 高德地图API已配置，使用真实天气数据")
    else:
        print("[Map] 未配置高德API Key，使用模拟天气数据")

    # Agent 状态
    print(f"[Agent] 多Agent协作架构已就绪:")
    print(f"       IntentAgent → ExploreAgent → HotelAgent → WeatherAgent → PlannerAgent")
    print()

    print(f"[Run] 启动服务器: http://localhost:{PORT}")
    print("      按 Ctrl+C 停止")
    print("-" * 60)

    app.run(host="0.0.0.0", port=PORT, debug=False)
