"""Autocomplete backend: a trie served over HTTP on localhost:5000.

Run:   python server.py        (Python 3.8+, no packages to install)
Needs: data/words.txt next to this file.
API:
  GET    /api/search?q=app        top 5 suggestions for a prefix
  GET    /api/benchmark?q=app     trie vs linear scan timing for a prefix
  GET    /api/stats               word and node counts
  GET    /api/tree?q=app&depth=3  subtree under a prefix (for the visualizer)
  POST   /api/words  {"word":"x"} add a word (in memory only, lost on restart)
  DELETE /api/words?word=x        remove a word and prune dead nodes
"""
import heapq
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HOST, PORT, TOP_K = "127.0.0.1", 5000, 5
WORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "words.txt")
WORD_RE = re.compile(r"^[a-z]{1,40}$")
LOCK = threading.RLock()        # the server is threaded and words can change at runtime
FREQ = {}                       # word -> frequency; mirrors the trie, used by the linear-scan baseline


class Node:
    __slots__ = ("kids", "is_word", "freq")

    def __init__(self):
        self.kids = {}          # hashmap: letter -> child node (only letters that exist)
        self.is_word = False
        self.freq = 0


class Item:
    """Heap entry. The heap's smallest item is the WORST suggestion, so it is the one evicted."""
    __slots__ = ("word", "freq")

    def __init__(self, word, freq):
        self.word, self.freq = word, freq

    def __lt__(self, other):    # "worse than": lower frequency, or same frequency and later alphabetically
        if self.freq != other.freq:
            return self.freq < other.freq
        return self.word > other.word


class Trie:
    def __init__(self):
        self.root = Node()
        self.nodes = 1

    def insert(self, word, freq=1):
        word = word.lower()
        if not word:
            return
        cur = self.root
        for ch in word:
            nxt = cur.kids.get(ch)
            if nxt is None:
                nxt = cur.kids[ch] = Node()
                self.nodes += 1
            cur = nxt
        cur.is_word = True
        cur.freq += freq

    def delete(self, word):
        """Unmark the word, then prune upward every node that no longer leads to any word."""
        path, cur = [], self.root
        for ch in word.lower():
            nxt = cur.kids.get(ch)
            if nxt is None:
                return False
            path.append((cur, ch))
            cur = nxt
        if not cur.is_word:
            return False
        cur.is_word, cur.freq = False, 0
        for parent, ch in reversed(path):
            child = parent.kids[ch]
            if child.kids or child.is_word:
                break
            del parent.kids[ch]
            self.nodes -= 1
        return True

    def suggest(self, prefix, k=TOP_K, stats=None):
        prefix = prefix.lower()
        node, visited = self.root, 1
        for ch in prefix:                       # walk down the typed letters
            node = node.kids.get(ch)
            if node is None:
                if stats is not None:
                    stats["nodes"] = visited
                return []
            visited += 1
        heap = []                               # min-heap of the best k words seen so far
        stack = [(node, prefix)]                # depth-first walk of everything below the prefix
        while stack:
            cur, word = stack.pop()
            if cur.is_word:
                heapq.heappush(heap, Item(word, cur.freq))
                if len(heap) > k:
                    heapq.heappop(heap)         # drop the worst
            for ch, child in cur.kids.items():
                visited += 1
                stack.append((child, word + ch))
        if stats is not None:
            stats["nodes"] = visited
        heap.sort(key=lambda it: (-it.freq, it.word))   # best first
        return [it.word for it in heap]

    def subtree(self, prefix, depth, budget=150):
        """Path root->prefix plus the nodes below it, `depth` levels deep, capped at `budget` nodes."""
        node, path = self.root, [{"ch": "root", "end": False}]
        for ch in prefix.lower():
            node = node.kids.get(ch)
            if node is None:
                return None
            path.append({"ch": ch, "end": node.is_word})
        left = [budget]

        def walk(n, ch, d):
            out = {"ch": ch, "end": n.is_word, "kids": [], "more": 0}
            if d == 0:
                out["more"] = len(n.kids)
                return out
            for i, (c, child) in enumerate(sorted(n.kids.items())):
                if left[0] <= 0:
                    out["more"] = len(n.kids) - i
                    break
                left[0] -= 1
                out["kids"].append(walk(child, c, d - 1))
            return out

        return path, walk(node, path[-1]["ch"], depth)


trie = Trie()


def add_word(word, freq):
    word = word.lower()
    trie.insert(word, freq)
    FREQ[word] = FREQ.get(word, 0) + freq


def remove_word(word):
    word = word.lower()
    FREQ.pop(word, None)
    return trie.delete(word)


def scan(prefix, k=TOP_K):
    """Baseline: check every word in the dictionary."""
    prefix = prefix.lower()
    found = [(w, f) for w, f in FREQ.items() if w.startswith(prefix)]
    best = heapq.nsmallest(k, found, key=lambda x: (-x[1], x[0]))
    return [w for w, _ in best], len(found)


def benchmark(prefix, reps=20):
    stats = {}
    results = trie.suggest(prefix, stats=stats)
    t0 = time.perf_counter()
    for _ in range(reps):
        trie.suggest(prefix)
    trie_us = (time.perf_counter() - t0) / reps * 1e6
    t0 = time.perf_counter()
    for _ in range(reps):
        _, matches = scan(prefix)
    scan_us = (time.perf_counter() - t0) / reps * 1e6
    return {"results": results, "matches": matches, "trie_us": round(trie_us, 1), "scan_us": round(scan_us, 1),
            "speedup": round(scan_us / max(trie_us, 0.01), 1), "trie_nodes": stats["nodes"], "scan_checked": len(FREQ)}


def load_words(path):
    with open(path, encoding="utf-8") as f:
        words = [line.split()[0] for line in f if line.strip()]
    n = len(words)
    for i, w in enumerate(words):               # earlier in the file = more common = higher frequency
        add_word(w, n - i)
    return n


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")   # lets the HTML page call this server
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        q = qs.get("q", [""])[0].strip()[:100]
        with LOCK:
            if url.path == "/api/search":
                return self._send(200, {"results": trie.suggest(q) if q else []})
            if url.path == "/api/benchmark":
                if not q:
                    return self._send(400, {"error": "Enter a prefix first."})
                return self._send(200, benchmark(q))
            if url.path == "/api/stats":
                return self._send(200, {"words": len(FREQ), "nodes": trie.nodes})
            if url.path == "/api/tree":
                try:
                    depth = max(1, min(4, int(qs.get("depth", ["3"])[0])))
                except ValueError:
                    depth = 3
                out = trie.subtree(q, depth)
                if out is None:
                    return self._send(200, {"found": False})
                return self._send(200, {"found": True, "path": out[0], "tree": out[1]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/words":
            return self._send(404, {"error": "not found"})
        word = str(self._body().get("word", "")).strip().lower()
        if not WORD_RE.match(word):
            return self._send(400, {"error": "Use 1 to 40 letters (a-z) only."})
        with LOCK:
            if word in FREQ:
                return self._send(409, {"error": "'%s' is already in the trie." % word})
            add_word(word, max(FREQ.values(), default=0) + 1)   # top frequency, so it ranks first
            self._send(200, {"ok": True, "words": len(FREQ), "nodes": trie.nodes})

    def do_DELETE(self):
        url = urlparse(self.path)
        if url.path != "/api/words":
            return self._send(404, {"error": "not found"})
        word = parse_qs(url.query).get("word", [""])[0].strip().lower()
        with LOCK:
            if word not in FREQ:
                return self._send(404, {"error": "'%s' is not in the trie." % word})
            remove_word(word)
            self._send(200, {"ok": True, "words": len(FREQ), "nodes": trie.nodes})

    def log_message(self, fmt, *args):
        print("%s  %s" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    try:
        count = load_words(WORDS)
    except OSError:
        raise SystemExit("Could not read %s. Put server.py next to the data folder." % WORDS)
    print("Loaded %d words. Serving on http://localhost:%d  (Ctrl+C to stop)" % (count, PORT))
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()